from abc import ABC, abstractmethod

import numpy as np


class Reservoir(ABC):
    """reservoir 后端统一接口。"""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """输出特征维度。"""

    @abstractmethod
    def reset_state(self):
        """重置内部状态。"""

    @abstractmethod
    def step(self, input_t: np.ndarray) -> np.ndarray:
        """输入单个时间步，返回 reservoir 特征。"""

    def forward_sequence(self, inputs: np.ndarray) -> np.ndarray:
        """处理完整序列，输入形状 [T, D]，输出形状 [T, Nz]。"""
        feats = []
        for input_t in inputs:
            feats.append(self.step(input_t))
        return np.asarray(feats)

