from itertools import combinations

import numpy as np

from .base import Reservoir


QRC_INPUT_ENCODINGS = ("pca_angle", "pca_dense_angle")


class DTCQRCReservoir(Reservoir):
    """按 DTC-QRC 论文构造的理想状态向量 reservoir。

    每个 T-RCNet 滑动窗口视为一个经典向量样本。编码器只用当前层 TS
    拟合 PCA，再按论文公式 (1)--(5) 做 angle 或 dense-angle 编码；
    随后按公式 (10) 演化，并按公式 (6)--(8) 在四个演化分段读取
    所有单体 Z_i 与两体 Z_i Z_j 期望值。

    这是将论文静态向量分类流程迁移到 MIMO-OFDM 窗口样本的明确假设。
    论文的标量时序注入使用密度矩阵替换，不直接定义 32 维窗口的注入，
    因而不能把本适配写成原文已经验证过的通信接收算法。
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 6,
        cycles: int = 12,
        measurement_count: int = 4,
        g: float = 0.84,
        seed: int = 2,
        input_encoding: str = "pca_dense_angle",
        input_scale: float = 1.0,
        include_input: bool = True,
        simulation_chunk_size: int = 512,
    ):
        if input_dim <= 0:
            raise ValueError("QRC 输入维度必须大于 0")
        if n_qubits <= 0:
            raise ValueError("QRC qubit 数必须大于 0")
        if n_qubits > 10:
            raise ValueError("状态向量仿真成本较高，当前限制 n_qubits <= 10")
        if cycles <= 0 or measurement_count <= 0:
            raise ValueError("Floquet 周期数和测量分段数必须大于 0")
        if cycles % measurement_count:
            raise ValueError("Floquet 周期数必须能被测量分段数整除")
        if not np.isfinite(g) or g < 0.0 or g > 1.0:
            raise ValueError("DTC 不完美翻转参数 g 必须位于 [0, 1]")
        if not np.isfinite(input_scale) or input_scale <= 0.0:
            raise ValueError("QRC 输入角度尺度必须为有限正数")
        normalized_encoding = input_encoding.lower()
        if normalized_encoding not in QRC_INPUT_ENCODINGS:
            raise ValueError(
                "input_encoding 必须为 'pca_angle' 或 "
                "'pca_dense_angle'"
            )
        if simulation_chunk_size <= 0:
            raise ValueError("QRC 状态向量批处理块大小必须大于 0")

        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.cycles = int(cycles)
        self.measurement_count = int(measurement_count)
        self.cycles_per_measurement = self.cycles // self.measurement_count
        self.g = float(g)
        self.input_encoding = normalized_encoding
        self.input_scale = float(input_scale)
        self.include_input = bool(include_input)
        self.simulation_chunk_size = int(simulation_chunk_size)
        self.rng = np.random.default_rng(seed)
        self.dim = 2**self.n_qubits

        # 论文公式 (10)：相互作用角与无序场在给定区间随机抽取。
        self.phi = self.rng.uniform(
            -1.5 * np.pi,
            -0.5 * np.pi,
            size=self.n_qubits - 1,
        )
        self.h = self.rng.uniform(-np.pi, np.pi, size=self.n_qubits)
        self._build_operators()

        self.encoder_component_count = (
            self.n_qubits
            if self.input_encoding == "pca_angle"
            else 2 * self.n_qubits
        )
        self.encoder_mean = None
        self.encoder_components = None
        self.encoder_scales = None
        self.explained_variance_ratio = None
        self.input_loading_norms = None
        self.reset_state()

    @property
    def measurement_feature_dim(self) -> int:
        """单个测量时刻的 Z_i 与全部 Z_i Z_j 特征数。"""
        return self.n_qubits + len(self.pair_indices)

    @property
    def feature_dim(self) -> int:
        """返回 QRC 测量特征与可选 T-RCNet 直接输入的总维度。"""
        return self.quantum_feature_dim + (
            self.input_dim if self.include_input else 0
        )

    @property
    def quantum_feature_dim(self) -> int:
        """论文公式 (8) 拼接多个测量分段后的纯量子特征维度。"""
        return self.measurement_count * self.measurement_feature_dim

    @property
    def explained_variance_ratio_sum(self) -> float:
        """返回 TS-PCA 保留的总方差比例；拟合前返回 NaN。"""
        if self.explained_variance_ratio is None:
            return float("nan")
        return float(np.sum(self.explained_variance_ratio))

    def reset_state(self):
        """重置到 |0...0>；每个向量样本编码时还会再次制备初态。"""
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0

    def _build_operators(self):
        """预计算公式 (6)、(7)、(10) 所需的对角量和单比特门。"""
        basis = np.arange(self.dim)
        self.z_diag = []
        for qubit in range(self.n_qubits):
            bit = (basis >> (self.n_qubits - qubit - 1)) & 1
            self.z_diag.append((1 - 2 * bit).astype(float))

        # 论文使用单体和两体局域可观测量；全部 i<j 配对随 qubit 数
        # 多项式增长，不能只保留相邻 qubit 的 n-1 个相关量。
        self.pair_indices = tuple(combinations(range(self.n_qubits), 2))
        self.zz_diag = [
            self.z_diag[left] * self.z_diag[right]
            for left, right in self.pair_indices
        ]
        self.measurement_diagonals = np.asarray(
            self.z_diag + self.zz_diag,
            dtype=float,
        )

        # 公式 (10)：U_flip=exp[-i*pi*g/2*sum_i X_i]。
        theta = self.g * np.pi
        self.rx_gate = np.asarray(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=np.complex128,
        )

        # 公式 (10) 的相互作用系数是 -i*phi/4，而 onsite 是 -i*h/2。
        phase = np.zeros(self.dim, dtype=float)
        for qubit in range(self.n_qubits - 1):
            phase += (
                -0.25
                * self.phi[qubit]
                * self.z_diag[qubit]
                * self.z_diag[qubit + 1]
            )
        for qubit in range(self.n_qubits):
            phase += -0.5 * self.h[qubit] * self.z_diag[qubit]
        self.diagonal_phase = np.exp(1j * phase)

    def fit_input_encoder(self, inputs: np.ndarray) -> None:
        """仅用当前层 TS 窗口拟合论文输入端的线性 PCA 映射。"""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("QRC PCA 输入必须具有 [T, input_dim] 维度")
        if inputs.shape[0] < 2:
            raise ValueError("QRC PCA 至少需要两个 TS 窗口样本")
        if not np.all(np.isfinite(inputs)):
            raise ValueError("QRC PCA 输入必须全部为有限数")

        self.encoder_mean = np.mean(inputs, axis=0)
        centered = inputs - self.encoder_mean[None, :]
        _, singular_values, vh = np.linalg.svd(
            centered,
            full_matrices=False,
        )
        available = min(self.encoder_component_count, vh.shape[0])
        components = np.zeros(
            (self.encoder_component_count, self.input_dim),
            dtype=float,
        )
        components[:available] = vh[:available]

        # SVD 的主轴符号任意；固定最大绝对载荷为正，保证跨平台可追溯。
        for index in range(available):
            pivot = int(np.argmax(np.abs(components[index])))
            if components[index, pivot] < 0.0:
                components[index] *= -1.0
        projected = centered @ components.T
        scales = np.sqrt(np.mean(np.square(projected), axis=0))
        scales = np.where(scales > 1e-12, scales, 1.0)

        total_variance = float(np.sum(np.square(singular_values)))
        explained = np.zeros(self.encoder_component_count, dtype=float)
        if total_variance > 0.0:
            explained[:available] = (
                np.square(singular_values[:available]) / total_variance
            )

        self.encoder_components = components
        self.encoder_scales = scales
        self.explained_variance_ratio = explained
        # 每个原始窗口分量对保留主成分的标准化载荷范数，用于审计
        # “后 26 维是否仍参与编码”，而不是只检查矩阵形状。
        standardized_components = components / scales[:, None]
        self.input_loading_norms = np.linalg.norm(
            standardized_components,
            axis=0,
        )

    def _encode_components(self, inputs: np.ndarray) -> np.ndarray:
        if self.encoder_components is None:
            raise RuntimeError("需要先用 TS 调用 fit_input_encoder")
        centered = inputs - self.encoder_mean[None, :]
        components = centered @ self.encoder_components.T
        components = components / self.encoder_scales[None, :]
        return np.clip(components, -1.0, 1.0) * self.input_scale

    def _prepare_product_states(
        self,
        encoded_components: np.ndarray,
    ) -> np.ndarray:
        """按论文公式 (4) 或理想态公式 (5) 批量制备乘积初态。"""
        theta = encoded_components[:, : self.n_qubits]
        if self.input_encoding == "pca_dense_angle":
            azimuth = encoded_components[
                :,
                self.n_qubits : 2 * self.n_qubits,
            ]
        else:
            azimuth = np.zeros_like(theta)

        local_zero = np.cos(theta / 2)
        local_one = np.exp(1j * azimuth) * np.sin(theta / 2)
        states = np.ones(
            (encoded_components.shape[0], 1),
            dtype=np.complex128,
        )
        for qubit in range(self.n_qubits):
            local = np.stack(
                [local_zero[:, qubit], local_one[:, qubit]],
                axis=1,
            )
            states = (
                states[:, :, None] * local[:, None, :]
            ).reshape(encoded_components.shape[0], -1)
        return states

    def _apply_single_qubit_gate_batch(
        self,
        states: np.ndarray,
        gate: np.ndarray,
        qubit: int,
    ) -> np.ndarray:
        """以 O(2^n) 张量收缩应用单比特门，避免构造 2^n 方阵。"""
        left_dim = 2**qubit
        right_dim = 2 ** (self.n_qubits - qubit - 1)
        tensor = states.reshape(
            states.shape[0],
            left_dim,
            2,
            right_dim,
        )
        updated = np.einsum(
            "ab,tlbr->tlar",
            gate,
            tensor,
            optimize=True,
        )
        return updated.reshape(states.shape)

    def _apply_floquet_cycle_batch(self, states: np.ndarray) -> np.ndarray:
        """批量执行论文公式 (10) 的一个 Floquet 周期。"""
        for qubit in range(self.n_qubits):
            states = self._apply_single_qubit_gate_batch(
                states,
                self.rx_gate,
                qubit,
            )
        states *= self.diagonal_phase[None, :]
        norms = np.linalg.norm(states, axis=1)
        return states / norms[:, None]

    def _measure_features_batch(self, states: np.ndarray) -> np.ndarray:
        probabilities = np.abs(states) ** 2
        return probabilities @ self.measurement_diagonals.T

    def _forward_chunk(
        self,
        inputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        encoded = self._encode_components(inputs)
        states = self._prepare_product_states(encoded)
        feature_segments = []
        for cycle_index in range(1, self.cycles + 1):
            states = self._apply_floquet_cycle_batch(states)
            if cycle_index % self.cycles_per_measurement == 0:
                feature_segments.append(
                    self._measure_features_batch(states)
                )
        quantum_features = np.concatenate(feature_segments, axis=1)
        if self.include_input:
            # T-RCNet 的扩展状态定义为 z=[s,i]。保留 i 可确保两种后端
            # 只替换 reservoir 状态映射；纯 DTC-QRC 论文公式 (9) 则可
            # 通过 include_input=False 运行。
            features = np.concatenate(
                [quantum_features, inputs],
                axis=1,
            )
        else:
            features = quantum_features
        return features, states[-1].copy()

    def forward_sequence(self, inputs: np.ndarray) -> np.ndarray:
        """批量处理独立窗口样本，返回 [T, feature_dim] 量子特征。"""
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("QRC 输入必须具有 [T, input_dim] 维度")
        if inputs.shape[0] == 0:
            return np.empty((0, self.feature_dim), dtype=float)
        chunks = []
        for start in range(0, inputs.shape[0], self.simulation_chunk_size):
            stop = min(start + self.simulation_chunk_size, inputs.shape[0])
            features, last_state = self._forward_chunk(inputs[start:stop])
            chunks.append(features)
            self.state = last_state
        return np.concatenate(chunks, axis=0)

    def step(self, input_t: np.ndarray) -> np.ndarray:
        """处理一个窗口；向量编码按论文公式 (4)、(5) 重新制备初态。"""
        input_t = np.asarray(input_t, dtype=float)
        if input_t.shape != (self.input_dim,):
            raise ValueError("单步 QRC 输入维度与 input_dim 不一致")
        return self.forward_sequence(input_t[None, :])[0]
