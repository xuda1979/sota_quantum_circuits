import numpy as np

from .config import OfdmConfig
from .frame import ofdm_fft, ofdm_ifft


def pilot_time_training_tuple(y_symbol: np.ndarray, x_freq_symbol: np.ndarray, cfg: OfdmConfig):
    """从数据 OFDM 符号中构造论文 Algorithm 1 的 pilot 时域训练元组。

    输入：
        y_symbol: 当前接收 OFDM 符号，形状为 [Nr, Ntd]，包含 CP。
        x_freq_symbol: 当前发送频域 OFDM 符号，形状为 [Nt, Nsc]。

    输出：
        y_pilot_time: 训练输入 I_i，形状为 [Nr, Ntd]。
        x_pilot_time: 训练标签 L_i，形状为 [Nt, Ntd]。

    对应论文公式：
        公式 (24)：I_{i,ncp}^r = q_{i,ncp}^r = F^H Σ F y_{i,ncp}^r
        公式 (25)：I_{i,cp}^r = q_{i,cp}^r = q_{i,ncp}^r[末尾 Ncp 个采样]
        公式 (45)：L_{i,ncp}^t = F^H P_i^t,
                  L_{i,cp}^t = L_{i,ncp}^t[末尾 Ncp 个采样]

    注意：
        公式 (25) 依赖论文公式 (23) 的 pilot pattern。该 pattern 在
        frame.generate_frame 中设置。
    """
    mask = np.zeros(cfg.nsc, dtype=bool)
    mask[list(cfg.pilot_indices)] = True

    # 取非 CP 部分 y_{i,ncp}^r，形状 [Nr, Nsc]。
    y_ncp = y_symbol[:, cfg.ncp :]

    # 论文公式 (24)：q_{i,ncp}^r = F^H Σ F y_{i,ncp}^r。
    # 变换幅度统一由 OfdmConfig 控制，避免输入与标签使用不同口径。
    y_freq = ofdm_fft(
        y_ncp,
        normalization=cfg.transform_normalization,
    )
    y_pilot_freq = np.zeros_like(y_freq)
    y_pilot_freq[:, mask] = y_freq[:, mask]
    y_pilot_ncp = ofdm_ifft(
        y_pilot_freq,
        normalization=cfg.transform_normalization,
    )

    # 论文公式 (25)：CP 部分直接复制 q_{i,ncp}^r 的末尾 Ncp 个采样。
    y_pilot_cp = y_pilot_ncp[:, -cfg.ncp :]
    y_pilot_time = np.concatenate([y_pilot_cp, y_pilot_ncp], axis=-1)

    # 构造频域 pilot P_i^t：仅保留 pilot 子载波，数据子载波置零。
    x_pilot_freq = np.zeros_like(x_freq_symbol)
    x_pilot_freq[:, mask] = x_freq_symbol[:, mask]

    # 论文公式 (45)：L_{i,ncp}^t = F^H P_i^t。
    x_pilot_ncp = ofdm_ifft(
        x_pilot_freq,
        normalization=cfg.transform_normalization,
    )

    # 论文公式 (45)：L_{i,cp}^t = L_{i,ncp}^t 的末尾 Ncp 个采样。
    x_pilot_cp = x_pilot_ncp[:, -cfg.ncp :]
    x_pilot_time = np.concatenate([x_pilot_cp, x_pilot_ncp], axis=-1)
    return y_pilot_time, x_pilot_time
