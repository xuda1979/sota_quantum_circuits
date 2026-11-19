import numpy as np

from .base import Reservoir


class ESNReservoir(Reservoir):
    """经典 ESN reservoir，对应 T-RCNet 中的 RC 模块。"""

    def __init__(
        self,
        input_dim: int,
        n_state: int = 64,
        spectral_radius: float = 0.2,
        input_scale: float = 0.5,
        input_distribution: str = "symmetric",
        recurrent_distribution: str = "symmetric",
        recurrent_connectivity: float = 1.0,
        activation: str = "tanh",
        seed: int = 1,
    ):
        valid_distributions = ("symmetric", "positive")
        normalized_input_distribution = input_distribution.lower()
        normalized_recurrent_distribution = recurrent_distribution.lower()
        if normalized_input_distribution not in valid_distributions:
            raise ValueError(
                "input_distribution 必须为 'symmetric' 或 'positive'"
            )
        if normalized_recurrent_distribution not in valid_distributions:
            raise ValueError(
                "recurrent_distribution 必须为 'symmetric' 或 'positive'"
            )
        if (
            not np.isfinite(recurrent_connectivity)
            or recurrent_connectivity <= 0.0
            or recurrent_connectivity > 1.0
        ):
            raise ValueError("recurrent_connectivity 必须位于 (0, 1]")
        valid_activations = ("tanh", "linear", "relu", "sigmoid")
        normalized_activation = activation.lower()
        if normalized_activation not in valid_activations:
            raise ValueError(
                "activation 必须为 'tanh'、'linear'、"
                "'relu' 或 'sigmoid'"
            )
        self.input_dim = input_dim
        self.n_state = n_state
        self.input_distribution = normalized_input_distribution
        self.recurrent_distribution = normalized_recurrent_distribution
        self.recurrent_connectivity = float(recurrent_connectivity)
        self.activation = normalized_activation
        self.rng = np.random.default_rng(seed)
        # 论文只说明 W_in 来自均匀分布，没有公开区间。两种模式使用
        # 同一底层随机数，只改变从单位均匀分布到目标区间的映射。
        input_low = (
            -input_scale
            if self.input_distribution == "symmetric"
            else 0.0
        )
        self.w_in = self.rng.uniform(
            input_low,
            input_scale,
            size=(n_state, input_dim),
        )
        # 论文实验设置明确说明输入权重和状态转移权重均由均匀分布生成。
        recurrent_low = (
            -1.0
            if self.recurrent_distribution == "symmetric"
            else 0.0
        )
        dense_w = self.rng.uniform(
            recurrent_low,
            1.0,
            size=(n_state, n_state),
        )
        # 所有连接率复用同一稠密权重和掩码底图；降低阈值只删除连接。
        # connectivity=1 时掩码全真，历史稠密基线逐元素保持不变。
        mask_draws = self.rng.random((n_state, n_state))
        mask = mask_draws < self.recurrent_connectivity
        w = dense_w * mask
        eig = np.max(np.abs(np.linalg.eigvals(w)))
        if eig <= 1e-12:
            raise ValueError(
                "当前 recurrent_connectivity 生成了零谱半径矩阵，"
                "请提高连接率或更换 model seed"
            )
        self.w = w * (spectral_radius / max(eig, 1e-12))
        self.state = np.zeros(n_state)

    @property
    def feature_dim(self) -> int:
        return self.n_state + self.input_dim

    def reset_state(self):
        self.state = np.zeros(self.n_state)

    def _activate(self, preactivation: np.ndarray) -> np.ndarray:
        """计算论文公式 (7) 中未公开具体形式的状态激活函数 f。"""
        if self.activation == "tanh":
            return np.tanh(preactivation)
        if self.activation == "linear":
            return preactivation
        if self.activation == "relu":
            return np.maximum(preactivation, 0.0)
        # 用 tanh 等价式稳定计算 logistic sigmoid，避免指数溢出。
        return 0.5 * (1.0 + np.tanh(0.5 * preactivation))

    def step(self, input_t: np.ndarray) -> np.ndarray:
        # ESN 内部权重固定，只把状态和输入拼接成 readout 特征。
        preactivation = self.w @ self.state + self.w_in @ input_t
        self.state = self._activate(preactivation)
        return np.concatenate([self.state, input_t])
