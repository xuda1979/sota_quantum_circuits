import numpy as np

from .frame import remove_cp_fft
from .qam import qam16_to_bits


def detect_data_bits(
    x_hat_time: np.ndarray,
    data_bits: dict,
    data_mask: np.ndarray,
    ncp: int,
    symbol_offset: int = 0,
    transform_normalization: str = "unitary",
):
    """将检测到的时域符号转为数据子载波比特。"""
    x_hat_freq = remove_cp_fft(
        x_hat_time,
        ncp,
        normalization=transform_normalization,
    )
    pred = {}
    for i in range(x_hat_freq.shape[0]):
        for t in range(x_hat_freq.shape[1]):
            symbols = x_hat_freq[i, t, data_mask]
            pred[(i + symbol_offset, t)] = qam16_to_bits(symbols)
    return pred


def bit_error_counts(pred_bits: dict, true_bits: dict) -> tuple[int, int]:
    """返回错误比特数与参与统计的总比特数。"""
    errors = 0
    total = 0
    for key, truth in true_bits.items():
        pred = pred_bits[key]
        n = min(pred.size, truth.size)
        errors += int(np.sum(pred[:n] != truth[:n]))
        total += n
    return errors, total


def bit_error_rate(pred_bits: dict, true_bits: dict) -> float:
    """计算 BER。"""
    errors, total = bit_error_counts(pred_bits, true_bits)
    return errors / max(total, 1)
