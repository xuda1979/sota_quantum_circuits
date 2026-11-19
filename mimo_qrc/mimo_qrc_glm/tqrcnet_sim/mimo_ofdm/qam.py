import numpy as np


def bits_to_16qam(bits: np.ndarray) -> np.ndarray:
    """将比特映射为归一化 16-QAM 符号。

    每 4 个比特对应 1 个复数符号。这里使用 Gray-like 映射，平均功率归一化为 1。
    """
    bits = np.asarray(bits, dtype=np.int8)
    if bits.size % 4 != 0:
        raise ValueError("16-QAM 调制要求比特数是 4 的整数倍")

    b = bits.reshape(-1, 4)
    levels = np.array([-3, -1, 3, 1], dtype=float)
    i_idx = b[:, 0] * 2 + b[:, 1]
    q_idx = b[:, 2] * 2 + b[:, 3]
    symbols = levels[i_idx] + 1j * levels[q_idx]
    return symbols / np.sqrt(10.0)


def qam16_to_bits(symbols: np.ndarray) -> np.ndarray:
    """将 16-QAM 符号硬判决为比特。"""
    symbols = np.asarray(symbols) * np.sqrt(10.0)
    levels = np.array([-3, -1, 1, 3], dtype=float)
    # 对实部和虚部分别寻找最近星座点。
    i_level = levels[np.argmin(np.abs(symbols.real[:, None] - levels[None, :]), axis=1)]
    q_level = levels[np.argmin(np.abs(symbols.imag[:, None] - levels[None, :]), axis=1)]

    # 与 bits_to_16qam 中的 Gray-like 映射保持一致。
    inv = {
        -3.0: (0, 0),
        -1.0: (0, 1),
        3.0: (1, 0),
        1.0: (1, 1),
    }
    out = np.empty((symbols.size, 4), dtype=np.int8)
    for idx, (iv, qv) in enumerate(zip(i_level, q_level)):
        out[idx, :2] = inv[float(iv)]
        out[idx, 2:] = inv[float(qv)]
    return out.reshape(-1)

