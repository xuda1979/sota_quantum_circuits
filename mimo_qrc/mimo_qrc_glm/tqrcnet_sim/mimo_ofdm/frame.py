import numpy as np

from .config import OfdmConfig
from .qam import bits_to_16qam


# IEEE 802.11-2012 公式 (20-11) 的 20 MHz legacy L-LTF 序列。
# lower 对应子载波 -26...-1，upper 对应 +1...+26。
WIFI_LTF_LOWER = np.array(
    [
        1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
        1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
    ],
    dtype=np.complex128,
)
WIFI_LTF_UPPER = np.array(
    [
        1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1,
        -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1,
    ],
    dtype=np.complex128,
)

# IEEE 802.11-2012 公式 (20-27) 的 HT-LTF 正交映射矩阵。
WIFI_HT_LTF_MAPPING = np.array(
    [
        [1, -1, 1, 1],
        [1, 1, -1, 1],
        [1, 1, 1, -1],
        [-1, 1, 1, 1],
    ],
    dtype=np.complex128,
)


def ofdm_ifft(
    x_freq: np.ndarray,
    normalization: str = "unitary",
) -> np.ndarray:
    """按指定幅度口径执行 OFDM IFFT。"""
    normalized_mode = normalization.lower()
    x_time = np.fft.ifft(x_freq, axis=-1)
    if normalized_mode == "unitary":
        return x_time * np.sqrt(x_freq.shape[-1])
    if normalized_mode == "matlab_native":
        return x_time
    raise ValueError(
        "normalization 必须为 'unitary' 或 'matlab_native'"
    )


def ofdm_fft(
    x_time: np.ndarray,
    normalization: str = "unitary",
) -> np.ndarray:
    """按与 `ofdm_ifft` 配对的幅度口径执行 OFDM FFT。"""
    normalized_mode = normalization.lower()
    x_freq = np.fft.fft(x_time, axis=-1)
    if normalized_mode == "unitary":
        return x_freq / np.sqrt(x_time.shape[-1])
    if normalized_mode == "matlab_native":
        return x_freq
    raise ValueError(
        "normalization 必须为 'unitary' 或 'matlab_native'"
    )


def ifft_with_cp(
    x_freq: np.ndarray,
    ncp: int,
    normalization: str = "unitary",
) -> np.ndarray:
    """对频域 OFDM 符号做 IFFT 并添加 CP。"""
    x_time = ofdm_ifft(x_freq, normalization=normalization)
    # Python 中 -0 等于 0；显式处理 Ncp=0，避免把完整符号重复一次。
    cp = x_time[..., -ncp:] if ncp > 0 else x_time[..., :0]
    return np.concatenate([cp, x_time], axis=-1)


def remove_cp_fft(
    x_time_cp: np.ndarray,
    ncp: int,
    normalization: str = "unitary",
) -> np.ndarray:
    """去除 CP 后做 FFT。"""
    x_time = x_time_cp[..., ncp:]
    return ofdm_fft(x_time, normalization=normalization)


def subcarrier_masks(cfg: OfdmConfig):
    """构造互斥的 data、pilot、null 三类子载波掩码。"""
    data_mask = np.zeros(cfg.nsc, dtype=bool)
    pilot_mask = np.zeros(cfg.nsc, dtype=bool)
    data_mask[list(cfg.resolved_data_indices)] = True
    pilot_mask[list(cfg.pilot_indices)] = True
    null_mask = ~(data_mask | pilot_mask)
    return data_mask, pilot_mask, null_mask


def _initial_mimo_pilots(cfg: OfdmConfig) -> np.ndarray:
    """构造不同发送流之间正交的初始 pilot。

    论文只规定公式 (23) 的跨符号递推，没有公开初始 pilot 数值。
    dft 采用 Npilot 点 DFT 的前 Nt 行；wifi_bpsk 复用 IEEE
    802.11-2012 公式 (20-27) 的 HT-LTF 正交 BPSK 矩阵。后者仅用于
    检查未公开初值对结果的敏感性，不能据此认定论文采用了该矩阵。
    """
    n_pilots = len(cfg.pilot_indices)
    if cfg.nt > n_pilots:
        raise ValueError("正交 MIMO pilot 要求 Nt 不大于 pilot 子载波数量")

    mode = cfg.initial_mimo_pilot_mode.lower()
    if mode == "wifi_bpsk":
        if n_pilots != WIFI_HT_LTF_MAPPING.shape[1]:
            raise ValueError("wifi_bpsk 初始 MIMO pilot 仅支持 4 个 pilot")
        return WIFI_HT_LTF_MAPPING[: cfg.nt].copy()

    if mode != "dft":
        raise ValueError(
            "initial_mimo_pilot_mode 必须为 'dft' 或 'wifi_bpsk'"
        )
    stream = np.arange(cfg.nt)[:, None]
    pilot = np.arange(n_pilots)[None, :]
    return np.exp(1j * 2.0 * np.pi * stream * pilot / n_pilots)


def wifi_ltf_training_symbols(cfg: OfdmConfig) -> np.ndarray:
    """构造论文 4×4、NTS=8 配置的 Wi-Fi LTF 复现假设。

    legacy Wi-Fi 的 52 个占用子载波承载标准 L-LTF BPSK 序列；发送流
    之间采用 IEEE HT-LTF 的正交映射。标准 4 流 HT-LTF 只有 4 个
    OFDM 符号，而论文设置 NTS=8，因此每个映射列连续重复两次。
    输出形状为 [NTS, Nt, 64]。
    """
    if cfg.training_sequence_mode.lower() != "wifi_ltf":
        raise ValueError("该函数仅用于 training_sequence_mode='wifi_ltf'")

    base = np.zeros(64, dtype=np.complex128)
    base[38:64] = WIFI_LTF_LOWER
    base[1:27] = WIFI_LTF_UPPER

    n_ltf = 1 if cfg.nt == 1 else 2 if cfg.nt == 2 else 4
    repeats = cfg.n_ts // n_ltf
    column_indices = np.repeat(np.arange(n_ltf), repeats)
    stream_codes = WIFI_HT_LTF_MAPPING[
        : cfg.nt,
        column_indices,
    ].T
    return stream_codes[:, :, None] * base[None, None, :]


def generate_frame(cfg: OfdmConfig):
    """生成一个 MIMO-OFDM 帧。

    `random_qam` 使用固定种子的已知 16-QAM TS；`wifi_ltf` 使用
    `wifi_ltf_training_symbols` 中记录的标准序列组合假设。论文未公开
    4×4、8-symbol TS 的具体矩阵，因此两种模式均保留用于配对消融。
    """
    rng = np.random.default_rng(cfg.seed)
    x_freq = np.zeros((cfg.n_total, cfg.nt, cfg.nsc), dtype=np.complex128)
    data_bits = {}
    data_mask, pilot_mask, null_mask = subcarrier_masks(cfg)
    pilot_indices = np.asarray(cfg.pilot_indices)

    # 论文公式 (23)：P_i = Δ^{-1} P_{i-1}。
    # 其中 diag(Δ) = exp(-j*2π*Ncp*k/Nsc)，所以 Δ^{-1} 的递推因子为
    # exp(+j*2π*Ncp*k/Nsc)。只有 pilot 子载波需要设置该相位递推。
    pilot_step = np.exp(1j * 2 * np.pi * cfg.ncp * pilot_indices / cfg.nsc)
    pilot_values = _initial_mimo_pilots(cfg)
    initial_pilot_values = pilot_values.copy()
    ts_mode = cfg.training_sequence_mode.lower()
    pilot_pattern_mode = cfg.pilot_pattern_mode.lower()

    for i in range(cfg.n_total):
        for t in range(cfg.nt):
            # 两种 TS 模式都消耗相同随机数，使数据段在配对实验中完全一致。
            bits = rng.integers(
                0,
                2,
                size=cfg.n_data_subcarriers * cfg.bits_per_qam_symbol,
                dtype=np.int8,
            )
            if i >= cfg.n_ts or ts_mode == "random_qam":
                x_freq[i, t, data_mask] = bits_to_16qam(bits)
            if i >= cfg.n_ts:
                data_bits[(i, t)] = bits

            if i >= cfg.n_ts or ts_mode == "random_qam":
                # 论文公式 (23)：数据 OFDM 符号中的 pilot 逐符号递推。
                x_freq[i, t, pilot_indices] = pilot_values[t]

        pilot_is_present = i >= cfg.n_ts or ts_mode == "random_qam"
        if pilot_is_present and (
            i < cfg.n_ts or pilot_pattern_mode == "paper_change"
        ):
            # random-QAM TS 在两种 Fig. 6 对照中保持完全一致；fixed
            # 只从首个数据符号之后停止公式 (23) 的跨符号递推。
            pilot_values = pilot_values * pilot_step[None, :]

    if ts_mode == "wifi_ltf":
        x_freq[: cfg.n_ts] = wifi_ltf_training_symbols(cfg)

    x_time = ifft_with_cp(
        x_freq,
        cfg.ncp,
        normalization=cfg.transform_normalization,
    )
    return {
        "x_freq": x_freq,
        "x_time": x_time,
        "data_bits": data_bits,
        "data_mask": data_mask,
        "pilot_mask": pilot_mask,
        "null_mask": null_mask,
        "initial_pilot_values": initial_pilot_values,
        "initial_mimo_pilot_mode": cfg.initial_mimo_pilot_mode.lower(),
        "training_sequence_mode": ts_mode,
    }


def flatten_time_symbols(symbols: np.ndarray) -> np.ndarray:
    """将 [Nsym, Nant, Ntd] 展开为 [Nsym*Ntd, Nant]。"""
    return np.transpose(symbols, (0, 2, 1)).reshape(-1, symbols.shape[1])
