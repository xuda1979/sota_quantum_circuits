"""频域 LMMSE 基线：TS 联合估计、梳状导频跟踪、逐子载波检测。

原文 VII-A 前只说明上述流程，未公开完整估计器。这里采用已知 PDP 的
联合贝叶斯 LMMSE 和一阶 Jakes 相关近似；具体假设见 LMMSE_BASELINE.md。
本模块的 B1--B6 是项目说明文档编号，不是原论文公式编号。
"""

from dataclasses import dataclass

import numpy as np

from tqrcnet_sim.mimo_ofdm.channel import epa_delays_in_samples, epa_profile, epa_frequency_basis
from tqrcnet_sim.mimo_ofdm.config import OfdmConfig
from tqrcnet_sim.mimo_ofdm.frame import remove_cp_fft


@dataclass(frozen=True)
class LmmseSettings:
    """固定的估计和判决口径，不根据测试 BER 调参。"""

    gain_normalization: bool = True


def channel_basis(cfg: OfdmConfig) -> tuple[np.ndarray, np.ndarray]:
    """返回 [Nsc,Npath] 延迟响应与每条链路的路径功率先验。"""
    if cfg.channel_time_evolution_mode.lower() != "symbol_block":
        raise ValueError("当前 LMMSE 基线只支持符号内不变信道，不忽略 ICI")
    if cfg.channel_model.lower() == "epa":
        delays = epa_delays_in_samples(cfg.sample_rate_hz)
        powers = epa_profile()[2]
    elif cfg.channel_model.lower() == "simplified":
        delays = np.arange(cfg.channel_taps, dtype=float)
        powers = np.exp(-delays)
        powers /= powers.sum()
    else:
        raise ValueError("不支持的 LMMSE 信道先验")
    if int(np.ceil(delays.max())) > cfg.ncp:
        raise ValueError("CP 短于信道记忆长度，不能使用无 ISI 的频域模型")
    if cfg.mimo_power_normalization.lower() == "total":
        powers = powers / cfg.nt
    if cfg.channel_model.lower() == "epa":
        # FIR 模式下 H 是相应频域对角响应；真实线性滤波的 CP 尾部残差
        # 仍保留在观测中，由实验单独量化，而不是宣称完全没有 ISI/ICI。
        return epa_frequency_basis(cfg), powers

    # 对应原文公式 (4) 的多径卷积；这里必须匹配实际信道的一阶分数延迟，
    # 不能直接用理想 exp(-j*omega*delay) 代替。说明文档公式 (B1)。
    integer = np.floor(delays)
    fraction = delays - integer
    omega = 2.0 * np.pi * np.arange(cfg.nsc)[:, None] / cfg.nsc
    basis = np.exp(-1j * omega * integer) * (
        (1.0 - fraction) + fraction * np.exp(-1j * omega)
    )
    return basis, powers


def frequency_noise_variance(time_variance: float, cfg: OfdmConfig) -> float:
    """酉 FFT 保持复噪声方差；原生 FFT 将方差放大 Nsc 倍。"""
    if not np.isfinite(time_variance) or time_variance < 0.0:
        raise ValueError("噪声方差必须为有限非负数")
    return float(time_variance) * (
        cfg.nsc if cfg.transform_normalization.lower() == "matlab_native" else 1
    )


def perfect_frequency_channel(coefficients: np.ndarray, cfg: OfdmConfig) -> np.ndarray:
    """仅用于诊断的理想 CSI，输出 [Nsym,Nsc,Nr,Nt]。"""
    basis, _ = channel_basis(cfg)
    if coefficients.ndim != 4 or coefficients.shape[1:] != (
        cfg.nr, cfg.nt, basis.shape[1]
    ):
        raise ValueError("信道系数维度必须为 [Nsym,Nr,Nt,Npath]")
    return np.einsum("srtp,kp->skrt", coefficients, basis)


class LmmseChannelEstimator:
    """联合估计全部 Tx 链路，只接收已知 TS 和 pilot，不接收数据标签。

    白化路径状态 u 的维度 D=Nt*Npath，均值 [D,Nr]，后验协方差 [D,D]。
    对低空间相关，各接收天线复用相同协方差，但有独立后验均值。
    """

    def __init__(self, cfg: OfdmConfig, noise_variance: float, *, prior_basis=None, prior_powers=None):
        self.cfg = cfg
        basis, powers = channel_basis(cfg)
        if (prior_basis is None) != (prior_powers is None):
            raise ValueError("显式先验必须同时提供响应基和路径功率")
        if prior_basis is not None:
            basis, powers = np.asarray(prior_basis), np.asarray(prior_powers, dtype=float)
            if (basis.shape != (cfg.nsc, powers.size) or powers.ndim != 1
                    or not np.all(np.isfinite(basis)) or not np.all(np.isfinite(powers))
                    or np.any(powers <= 0)):
                raise ValueError("显式先验的维度或统计功率无效")
        self.weighted_basis = basis * np.sqrt(powers)[None, :]
        self.noise_variance = frequency_noise_variance(noise_variance, cfg)
        dimension = cfg.nt * powers.size
        self.mean = np.zeros((dimension, cfg.nr), dtype=np.complex128)
        self.covariance = np.eye(dimension, dtype=np.complex128)
        # 说明文档 (B4)：rho=J0(2*pi*fD*Ts)。角度积分避免额外 scipy 依赖。
        angles = 2.0 * np.pi * (np.arange(256) + 0.5) / 256
        argument = 2.0 * np.pi * cfg.doppler_hz * cfg.resolved_channel_evolution_interval_s
        self.rho = float(np.mean(np.cos(argument * np.cos(angles))))
        if cfg.channel_model.lower() == "simplified":
            self.rho = float(np.sqrt(1.0 - cfg.channel_drift))
        self.fitted = False

    def _design(self, known: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """说明文档 (B2)：每行同时包含所有 Tx 的已知符号与路径响应。"""
        if known.ndim != 3 or known.shape[1:] != (self.cfg.nt, indices.size):
            raise ValueError("已知符号必须为 [Nsymbol,Nt,Ncarrier]")
        return np.einsum(
            "stk,kp->sktp", known, self.weighted_basis[indices]
        ).reshape(-1, self.mean.shape[0])

    def fit_ts(self, received_ts: np.ndarray, known_ts: np.ndarray) -> None:
        """TS 内准静态近似，联合使用所有占用子载波；只允许 TS 数量输入。"""
        cfg = self.cfg
        if known_ts.shape != (cfg.n_ts, cfg.nt, cfg.nsc):
            raise ValueError("TS 输入必须为 [Nts,Nt,Nsc]")
        if received_ts.shape != (cfg.n_ts, cfg.nr, cfg.nsc):
            raise ValueError("TS 接收必须为 [Nts,Nr,Nsc]")
        indices = np.flatnonzero(np.any(np.abs(known_ts) > 0.0, axis=(0, 1)))
        design = self._design(known_ts[:, :, indices], indices)
        target = received_ts[:, :, indices].transpose(0, 2, 1).reshape(-1, cfg.nr)
        identity = np.eye(self.mean.shape[0], dtype=np.complex128)
        # 说明文档 (B3)：u_hat=(A^H A+sigma_f^2 I)^-1 A^H y。
        # 白化先验为 I，即使多个 EPA 分数路径线性相关也不会反演奇异 PDP。
        if self.noise_variance == 0.0:
            inverse = np.linalg.pinv(design)
            self.mean = inverse @ target
            self.covariance = identity - inverse @ design
        else:
            gram = design.conj().T @ design + self.noise_variance * identity
            self.mean = np.linalg.solve(gram, design.conj().T @ target)
            self.covariance = self.noise_variance * np.linalg.solve(gram, identity)
        self.covariance = (self.covariance + self.covariance.conj().T) / 2.0
        self.fitted = True

    def update_pilots(self, received_pilots: np.ndarray, known_pilots: np.ndarray) -> None:
        """当前 OFDM 符号的 4 个混合 pilot 联合更新，不读取数据子载波。"""
        if not self.fitted:
            raise RuntimeError("必须先使用 TS 初始化")
        cfg = self.cfg
        indices = np.asarray(cfg.pilot_indices)
        if received_pilots.shape != (cfg.nr, indices.size):
            raise ValueError("接收 pilot 必须为 [Nr,Npilot]")
        design = self._design(known_pilots[None, :, :], indices)
        identity = np.eye(self.mean.shape[0], dtype=np.complex128)
        predicted_mean = self.rho * self.mean
        predicted_cov = self.rho**2 * self.covariance + (1.0 - self.rho**2) * identity
        cross = predicted_cov @ design.conj().T
        innovation_cov = design @ cross + self.noise_variance * np.eye(indices.size)
        gain = np.linalg.lstsq(innovation_cov, cross.conj().T, rcond=None)[0].conj().T
        # 说明文档 (B5)：条件 LMMSE，同时完成梳状导频到数据频点的 Wiener 插值。
        residual = received_pilots.T - design @ predicted_mean
        self.mean = predicted_mean + gain @ residual
        # Joseph 形式减少反复更新造成的非 Hermitian/负特征值数值误差。
        remainder = identity - gain @ design
        self.covariance = (
            remainder @ predicted_cov @ remainder.conj().T
            + self.noise_variance * gain @ gain.conj().T
        )
        self.covariance = (self.covariance + self.covariance.conj().T) / 2.0

    def frequency_channel(self) -> np.ndarray:
        """输出 [Nsc,Nr,Nt]，其中每个子载波是一张 Nr x Nt 信道矩阵。"""
        if not self.fitted:
            raise RuntimeError("必须先使用 TS 初始化")
        paths = self.mean.reshape(self.cfg.nt, -1, self.cfg.nr)
        return np.einsum("tpr,kp->krt", paths, self.weighted_basis)


def equalize_lmmse(
    received_frequency: np.ndarray,
    channel: np.ndarray,
    noise_variance: float,
    gain_normalization: bool = True,
) -> np.ndarray:
    """说明文档 (B6)：每个频点执行 W=(H^H H+sigma_f^2 I)^-1 H^H。

    输入 [Nr,Nsc] 和 [Nsc,Nr,Nt]，输出 [Nt,Nsc]。16-QAM 平均 Es=1。
    可选的逐流有效增益校正避免 LMMSE 的幅度收缩改变 QAM 判决门限。
    """
    if channel.ndim != 3 or received_frequency.shape != (
        channel.shape[1], channel.shape[0]
    ):
        raise ValueError("接收信号/信道的频域维度不一致")
    if not np.isfinite(noise_variance) or noise_variance < 0.0:
        raise ValueError("噪声方差必须为有限非负数")
    adjoint = channel.conj().transpose(0, 2, 1)
    if noise_variance == 0.0:
        weights = np.linalg.pinv(channel)
    else:
        gram = adjoint @ channel + noise_variance * np.eye(channel.shape[2])
        weights = np.linalg.solve(gram, adjoint)
    detected = np.einsum("ktr,rk->kt", weights, received_frequency)
    if gain_normalization:
        gain = np.einsum("ktr,krt->kt", weights, channel).real
        detected /= np.maximum(gain, np.finfo(float).eps)
    return detected.T


def detect_estimated_csi(
    received_time: np.ndarray,
    known_ts: np.ndarray,
    known_pilots: np.ndarray,
    cfg: OfdmConfig,
    time_noise_variance: float,
    track_pilots: bool = True,
    settings: LmmseSettings | None = None,
) -> np.ndarray:
    """实用基线入口；函数不接收真实信道和数据标签，输出 [Ndata,Nt,Nsc]。"""
    settings = settings or LmmseSettings()
    if received_time.shape != (cfg.n_ts + cfg.n_data, cfg.nr, cfg.ntd):
        raise ValueError("接收帧维度不一致")
    if known_pilots.shape != (cfg.n_data, cfg.nt, len(cfg.pilot_indices)):
        raise ValueError("已知 pilot 必须为 [Ndata,Nt,Npilot]")
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    estimator = LmmseChannelEstimator(cfg, time_noise_variance)
    estimator.fit_ts(received[:cfg.n_ts], known_ts)
    detected = []
    for index, symbol in enumerate(received[cfg.n_ts:]):
        if track_pilots:
            estimator.update_pilots(symbol[:, cfg.pilot_indices], known_pilots[index])
        detected.append(equalize_lmmse(
            symbol, estimator.frequency_channel(), estimator.noise_variance,
            settings.gain_normalization,
        ))
    return np.asarray(detected)
