from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .config import OfdmConfig


# 3GPP TS 36.101 表 B.2.1-2：Extended Pedestrian A（EPA）7 径模型。
EPA_DELAYS_NS = np.array([0.0, 30.0, 70.0, 90.0, 110.0, 190.0, 410.0])
EPA_POWERS_DB = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])


@dataclass(frozen=True)
class ChannelRealization:
    """一次固定的信道与单位噪声 realization。

    `unit_noise` 满足 E[|n|^2]=1。改变 Eb/N0 时只缩放该数组，从而让
    同一 trial 的所有 SNR 点和所有检测算法复用相同信道及噪声轨迹。
    """

    noiseless_received: np.ndarray
    coefficients: np.ndarray
    unit_noise: np.ndarray


def epa_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 EPA 时延、相对功率和归一化线性功率。

    返回：
        delays_ns: 7 径超额时延，单位 ns。
        powers_db: 3GPP 表中的相对功率，单位 dB。
        powers_linear: 总和归一化为 1 的线性功率。
    """
    powers_linear = 10.0 ** (EPA_POWERS_DB / 10.0)
    powers_linear /= powers_linear.sum()
    return EPA_DELAYS_NS.copy(), EPA_POWERS_DB.copy(), powers_linear


def epa_delays_in_samples(sample_rate_hz: float) -> np.ndarray:
    """将 EPA 的 ns 时延转换为当前采样率下的分数采样时延。"""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz 必须大于 0")
    return EPA_DELAYS_NS * 1e-9 * sample_rate_hz


@lru_cache(maxsize=1)
def matlab_epa_fir() -> tuple[np.ndarray, np.ndarray]:
    """读取延迟补偿后的实测路径 FIR；运行时不依赖 MATLAB 安装。"""
    path = Path(__file__).with_name("data") / "matlab_r2018b_epa_20mhz_fir.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    lags = np.asarray(payload["lags"], dtype=int)
    coefficients = np.asarray(payload["coefficients"], dtype=float)
    if (payload["sample_rate_hz"] != 20e6 or coefficients.shape != (7, lags.size)
            or not np.array_equal(lags, np.arange(lags[0], lags[-1] + 1))
            or not np.all(np.isfinite(coefficients))):
        raise ValueError("MATLAB EPA FIR 校准文件无效")
    lags.setflags(write=False)
    coefficients.setflags(write=False)
    return lags, coefficients


def epa_frequency_basis(cfg: OfdmConfig) -> np.ndarray:
    """按实际延迟滤波器构造 [Nsc,7] 响应；仅提供统计结构，不含真实增益。"""
    if cfg.fractional_delay_mode.lower() == "matlab_fir":
        lags, coefficients = matlab_epa_fir()
        kernel = np.exp(-2j * np.pi * np.arange(cfg.nsc)[:, None] * lags[None] / cfg.nsc)
        return kernel @ coefficients.T
    delays = epa_delays_in_samples(cfg.sample_rate_hz)
    integer, fraction = np.floor(delays), delays - np.floor(delays)
    omega = 2.0 * np.pi * np.arange(cfg.nsc)[:, None] / cfg.nsc
    return np.exp(-1j * omega * integer) * ((1.0 - fraction) + fraction * np.exp(-1j * omega))


def _fir_delay(signal, lags, coefficients):
    """原文公式 (4) 的线性卷积；负 lag 表示已补偿实现延迟后的前游抽头。

    不把帧改为循环卷积，也不添加人工循环后缀。FIR 超出 CP 的小尾部会
    保留在实际波形中，并在诊断中报告，不能为得到理想频域模型而删除。
    """
    convolved = np.convolve(signal, coefficients, mode="full")
    positions = np.arange(signal.size) - int(lags[0])
    output = np.zeros(signal.size, dtype=np.complex128)
    valid = (positions >= 0) & (positions < convolved.size)
    output[valid] = convolved[positions[valid]]
    return output


def generate_epa_channel(
    cfg: OfdmConfig,
    n_symbols: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成符号块或逐样本时变的 MIMO-EPA 信道系数。

    `symbol_block` 输出 [Nsym, Nr, Nt, 7]，对应原文明确采用的符号内
    不变假设。`sample_continuous` 输出 [Nsym, Ntd, Nr, Nt, 7]，用于
    P2-1 审计 MATLAB `lteFadingChannel` 的逐样本 PathGains 语义。
    两种模式复用同一组和振荡器频率与初相位，且每个符号首样本相等。
    """
    if n_symbols <= 0:
        raise ValueError("n_symbols 必须大于 0")
    if cfg.jakes_sinusoids <= 0:
        raise ValueError("jakes_sinusoids 必须大于 0")
    if cfg.doppler_hz < 0.0:
        raise ValueError("doppler_hz 不能为负数")
    if rng is None:
        rng = np.random.default_rng(cfg.seed + 1000)

    _, _, path_powers = epa_profile()
    n_paths = path_powers.size
    n_osc = cfg.jakes_sinusoids

    # 经典 Jakes 谱可由 f_m = f_D*cos(theta_m) 的复指数叠加近似。
    # 每条 MIMO 链路、每条路径使用独立角度偏移和初始相位。
    base_angles = 2.0 * np.pi * (np.arange(n_osc) + 0.5) / n_osc
    angle_offsets = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=(cfg.nr, cfg.nt, n_paths, 1),
    )
    oscillator_angles = base_angles[None, None, None, :] + angle_offsets
    oscillator_frequencies = cfg.doppler_hz * np.cos(oscillator_angles)
    initial_phases = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=(cfg.nr, cfg.nt, n_paths, n_osc),
    )

    mode = cfg.channel_time_evolution_mode.lower()
    if mode == "symbol_block":
        sample_times = (
            np.arange(n_symbols)
            * cfg.resolved_channel_evolution_interval_s
        )
    elif mode == "sample_continuous":
        sample_times = (
            np.arange(n_symbols * cfg.ntd)
            * cfg.resolved_channel_sample_interval_s
        )
    else:
        # OfdmConfig 已验证该字段；保留防御分支以支持外部兼容配置对象。
        raise ValueError(
            "channel_time_evolution_mode 必须为 "
            "'symbol_block' 或 'sample_continuous'"
        )

    # 逐样本论文帧包含 8000 个时间点。分块计算避免同时创建
    # [Nr,Nt,Npath,Nosc,Nsample] 的大型临时复数数组。
    fading = np.empty(
        (sample_times.size, cfg.nr, cfg.nt, n_paths),
        dtype=np.complex128,
    )
    chunk_size = 512
    for start in range(0, sample_times.size, chunk_size):
        stop = min(start + chunk_size, sample_times.size)
        phases = (
            initial_phases[..., None]
            + 2.0
            * np.pi
            * oscillator_frequencies[..., None]
            * sample_times[None, None, None, None, start:stop]
        )
        chunk = np.exp(1j * phases).sum(axis=3) / np.sqrt(n_osc)
        fading[start:stop] = np.moveaxis(chunk, -1, 0)

    # EPA 表只规定单条链路的相对路径功率，论文没有说明 MIMO 总功率口径。
    # total：除以 Nt，使接收端聚合功率不随发射天线数增加；
    # per_link：每条 Tx-Rx 链路的 7 径功率和为 1。
    power_divisor = (
        cfg.nt if cfg.mimo_power_normalization.lower() == "total" else 1.0
    )
    path_scale = np.sqrt(path_powers / power_divisor)
    fading *= path_scale[None, None, None, :]
    if mode == "sample_continuous":
        return fading.reshape(
            n_symbols,
            cfg.ntd,
            cfg.nr,
            cfg.nt,
            n_paths,
        )
    return fading


def _fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """用一阶分数延迟插值计算 x[n-delay]，帧外历史样本补零。"""
    if delay_samples < 0.0:
        raise ValueError("路径时延不能为负数")

    source_positions = np.arange(signal.size, dtype=float) - delay_samples
    left_indices = np.floor(source_positions).astype(int)
    fractions = source_positions - left_indices
    delayed = np.zeros(signal.size, dtype=np.complex128)

    left_valid = (left_indices >= 0) & (left_indices < signal.size)
    delayed[left_valid] += (
        1.0 - fractions[left_valid]
    ) * signal[left_indices[left_valid]]

    right_indices = left_indices + 1
    right_valid = (right_indices >= 0) & (right_indices < signal.size)
    delayed[right_valid] += (
        fractions[right_valid]
    ) * signal[right_indices[right_valid]]
    return delayed


def _apply_time_varying_tapped_channel(
    x_time: np.ndarray,
    channel_coefficients: np.ndarray,
    delays_samples: np.ndarray,
    path_filters: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """对连续 MIMO-OFDM 帧施加符号块或逐样本时变抽头信道。

    输入符号先按时间串接，因此当前符号开头的延迟分量可以来自上一符号
    的尾部，保留论文公式 (4) 中线性卷积导致的跨符号影响。
    """
    n_symbols, nt, ntd = x_time.shape
    if channel_coefficients.ndim == 4:
        expected_prefix = (n_symbols,)
        nr = channel_coefficients.shape[1]
        coefficient_nt = channel_coefficients.shape[2]
    elif channel_coefficients.ndim == 5:
        expected_prefix = (n_symbols, ntd)
        nr = channel_coefficients.shape[2]
        coefficient_nt = channel_coefficients.shape[3]
    else:
        raise ValueError(
            "信道系数必须具有 [Nsym,Nr,Nt,Npath] 或 "
            "[Nsym,Ntd,Nr,Nt,Npath] 维度"
        )
    if (
        channel_coefficients.shape[: len(expected_prefix)] != expected_prefix
        or coefficient_nt != nt
    ):
        raise ValueError("信道系数的 Nsym、Ntd 或 Nt 与发送信号不一致")
    if channel_coefficients.shape[-1] != delays_samples.size:
        raise ValueError("信道路径数必须与时延数量相同")

    serial = np.transpose(x_time, (1, 0, 2)).reshape(nt, -1)
    total_samples = serial.shape[1]
    delayed_signals = np.empty(
        (delays_samples.size, nt, total_samples),
        dtype=np.complex128,
    )
    for path_index, delay in enumerate(delays_samples):
        for tx_index in range(nt):
            if path_filters is None:
                delayed_signals[path_index, tx_index] = _fractional_delay(serial[tx_index], float(delay))
            else:
                lags, filters = path_filters
                delayed_signals[path_index, tx_index] = _fir_delay(serial[tx_index], lags, filters[path_index])

    if channel_coefficients.ndim == 5:
        sample_coefficients = channel_coefficients.reshape(
            total_samples,
            nr,
            nt,
            delays_samples.size,
        )
        y_serial = np.einsum(
            "srtp,pts->rs",
            sample_coefficients,
            delayed_signals,
        )
        return y_serial.reshape(nr, n_symbols, ntd).transpose(1, 0, 2)

    y = np.zeros((n_symbols, nr, ntd), dtype=np.complex128)
    for symbol_index in range(n_symbols):
        start = symbol_index * ntd
        stop = start + ntd
        delayed_block = delayed_signals[:, :, start:stop]
        y[symbol_index] = np.einsum(
            "rtp,ptn->rn",
            channel_coefficients[symbol_index],
            delayed_block,
        )
    return y


def replay_channel_realization(x_time, cfg, coefficients, unit_noise) -> ChannelRealization:
    """对明确给定的路径增益回放信道，用于同 realization 的滤波器对照。

    系数只能由信道仿真侧提供；接收机只能访问 noisy received。
    MATLAB 参考已按其 ChannelFilterDelay 对齐，不能在这里再次移位。
    """
    if cfg.channel_model.lower() != "epa":
        raise ValueError("参考回放只支持 EPA")
    filters = matlab_epa_fir() if cfg.fractional_delay_mode.lower() == "matlab_fir" else None
    received = _apply_time_varying_tapped_channel(
        x_time, coefficients, epa_delays_in_samples(cfg.sample_rate_hz), filters
    )
    if received.shape != unit_noise.shape:
        raise ValueError("回放的噪声与接收波形维度不同")
    return ChannelRealization(received, coefficients, unit_noise)


def _initial_simplified_channel(
    cfg: OfdmConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成旧版指数功率延迟信道，仅用于消融对照。"""
    h = (
        rng.normal(size=(cfg.nr, cfg.nt, cfg.channel_taps))
        + 1j * rng.normal(size=(cfg.nr, cfg.nt, cfg.channel_taps))
    )
    power = np.exp(-np.arange(cfg.channel_taps))
    h *= np.sqrt(power[None, None, :] / power.sum())
    power_divisor = (
        cfg.nt if cfg.mimo_power_normalization.lower() == "total" else 1.0
    )
    return h / np.sqrt(2 * power_divisor)


def _generate_simplified_channel(
    cfg: OfdmConfig,
    n_symbols: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成旧版 AR 漂移信道序列，便于和 EPA 做消融实验。"""
    h = _initial_simplified_channel(cfg, rng)
    sequence = np.empty(
        (n_symbols, cfg.nr, cfg.nt, cfg.channel_taps),
        dtype=np.complex128,
    )
    sequence[0] = h
    for symbol_index in range(1, n_symbols):
        innovation = _initial_simplified_channel(cfg, rng)
        h = (
            np.sqrt(1.0 - cfg.channel_drift) * h
            + np.sqrt(cfg.channel_drift) * innovation
        )
        sequence[symbol_index] = h
    return sequence


def payload_energy_per_bit(x_time: np.ndarray, cfg: OfdmConfig) -> float:
    """计算数据 OFDM 符号的实际发射能量/有效载荷比特数。

    该定义计入数据符号中的 pilot 与 CP 发射能量，但不把 TS 计入
    payload Eb。Nt 条发送流的能量和比特数同时累加，因此适用于 MIMO。
    """
    if x_time.ndim != 3 or x_time.shape[1:] != (cfg.nt, cfg.ntd):
        raise ValueError("x_time 维度必须与 OfdmConfig 的 Nt、Ntd 一致")
    n_payload_symbols = x_time.shape[0] - cfg.n_ts
    if n_payload_symbols <= 0:
        raise ValueError("计算 Eb 至少需要一个数据 OFDM 符号")

    payload_time = x_time[cfg.n_ts :]
    payload_energy = float(np.sum(np.abs(payload_time) ** 2))
    payload_bits = (
        n_payload_symbols
        * cfg.nt
        * cfg.n_data_subcarriers
        * cfg.bits_per_qam_symbol
    )
    if payload_bits <= 0 or payload_energy <= 0.0:
        raise ValueError("payload 比特数和发射能量必须大于 0")
    return payload_energy / payload_bits


def noise_variance_from_ebn0(
    x_time: np.ndarray,
    cfg: OfdmConfig,
    ebn0_db: float | None = None,
) -> float:
    """由实际 payload Eb 计算每个复基带噪声采样的方差 N0。

    对复 AWGN 使用 E[|n|^2]=N0，即实部和虚部方差均为 N0/2。
    """
    if ebn0_db is None:
        ebn0_db = cfg.ebn0_db
    ebn0_linear = 10.0 ** (float(ebn0_db) / 10.0)
    return payload_energy_per_bit(x_time, cfg) / ebn0_linear


def _add_awgn(
    signal: np.ndarray,
    noise_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """按给定复噪声方差 N0 添加高斯白噪声。"""
    if noise_variance < 0.0:
        raise ValueError("noise_variance 不能为负数")
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape)
    )
    return signal + noise


def generate_channel_realization(
    x_time: np.ndarray,
    cfg: OfdmConfig,
) -> ChannelRealization:
    """生成一次固定的 MIMO 多径信道和单位复高斯噪声。

    默认 `channel_model="epa"`，采用 3GPP EPA 7 径功率时延谱和论文的
    20 Hz Doppler 设置。时间粒度由 `channel_time_evolution_mode` 控制。

    输入形状：[Nsym, Nt, Ntd]
    无噪声接收信号和单位噪声形状：[Nsym, Nr, Ntd]
    """
    if x_time.ndim != 3:
        raise ValueError("x_time 必须具有 [Nsym, Nt, Ntd] 三个维度")
    if x_time.shape[1:] != (cfg.nt, cfg.ntd):
        raise ValueError("x_time 的 Nt 或 Ntd 与 OfdmConfig 不一致")

    n_symbols = x_time.shape[0]
    rng = np.random.default_rng(cfg.seed + 1000)
    channel_model = cfg.channel_model.lower()

    if channel_model == "epa":
        coefficients = generate_epa_channel(cfg, n_symbols, rng=rng)
        delays_samples = epa_delays_in_samples(cfg.sample_rate_hz)
    elif channel_model == "simplified":
        coefficients = _generate_simplified_channel(cfg, n_symbols, rng)
        delays_samples = np.arange(cfg.channel_taps, dtype=float)
    else:
        raise ValueError(
            f"未知信道模型 {cfg.channel_model!r}，可选值为 'epa' 或 'simplified'"
        )

    faded = _apply_time_varying_tapped_channel(
        x_time,
        coefficients,
        delays_samples,
        matlab_epa_fir() if cfg.fractional_delay_mode.lower() == "matlab_fir" else None,
    )
    unit_noise = (
        rng.normal(size=faded.shape) + 1j * rng.normal(size=faded.shape)
    ) / np.sqrt(2.0)
    return ChannelRealization(
        noiseless_received=faded,
        coefficients=coefficients,
        unit_noise=unit_noise,
    )


def observe_channel_realization(
    realization: ChannelRealization,
    x_time: np.ndarray,
    cfg: OfdmConfig,
    ebn0_db: float | None = None,
) -> np.ndarray:
    """在指定 Eb/N0 下观测已有信道 realization。"""
    if realization.noiseless_received.shape != realization.unit_noise.shape:
        raise ValueError("无噪声接收信号与单位噪声维度必须一致")

    # 按有效载荷 Eb/N0 标定，而不是把 Eb/N0 直接当作接收采样 SNR。
    noise_variance = noise_variance_from_ebn0(x_time, cfg, ebn0_db)
    return (
        realization.noiseless_received
        + np.sqrt(noise_variance) * realization.unit_noise
    )


def transmit_through_channel(
    x_time: np.ndarray,
    cfg: OfdmConfig,
    return_channel: bool = False,
):
    """通过时变 MIMO 多径信道并添加 AWGN。

    这是兼容原有调用的便捷接口。需要在多个 Eb/N0 之间复用同一条
    信道与噪声轨迹时，应组合调用 `generate_channel_realization` 和
    `observe_channel_realization`。
    """
    realization = generate_channel_realization(x_time, cfg)
    received = observe_channel_realization(realization, x_time, cfg)
    if return_channel:
        return received, realization.coefficients
    return received
