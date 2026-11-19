import numpy as np


GAW_ERROR_POWER_MODES = ("sum", "mean")
GAW_WEIGHT_GRANULARITIES = ("global", "per_stream")


def gaw_error_power(
    error: np.ndarray,
    mode: str = "sum",
) -> float:
    """计算送入 GAW 权重函数的预测误差功率。

    `sum` 对应论文原式 ``||e(n)||_2^2``。`mean` 是 P1-2 中用于审计
    作者未公开实现的输出维度归一化分支，不应被解释为论文明确规定。
    """
    normalized_mode = mode.lower()
    if normalized_mode not in GAW_ERROR_POWER_MODES:
        raise ValueError("GAW 误差功率模式必须为 'sum' 或 'mean'")

    error = np.asarray(error)
    if error.size == 0:
        raise ValueError("GAW 误差向量不能为空")
    power = float(np.vdot(error, error).real)
    if normalized_mode == "mean":
        power /= error.size
    return power


def gaw_weight_from_error(
    error: np.ndarray,
    alpha: float = 27.0,
    beta: float = 15.0,
    error_power_scale: float = 1.0,
    error_power_mode: str = "sum",
) -> float:
    """根据预测误差计算论文中的 GAW-RLS 样本权重。

    论文实验部分给出的权重函数为：
        omega(n) = 1 / (1 + exp(alpha + beta * log(||e(n)||_2^2)))

    其中 alpha=27、beta=15。误差越大，当前训练样本的权重越小，
    从而减弱强噪声或异常 pilot 样本对 readout 的影响。

    `error_power_scale` 用于审计误差功率单位：
        omega(n) = 1 / (1 + exp(alpha
                   + beta * log(scale * ||e(n)||_2^2)))

    默认 `scale=1` 且 `error_power_mode="sum"` 严格对应论文原式。
    `mean` 与非 1 尺度只作为误差功率口径消融参数，不会缩放公式 (11)
    中真正用于更新输出权重的预测误差。
    """
    if not np.isfinite(error_power_scale) or error_power_scale <= 0.0:
        raise ValueError("GAW 误差功率尺度必须为有限正数")

    error_power = error_power_scale * gaw_error_power(
        error,
        mode=error_power_mode,
    )
    if error_power <= 0.0:
        return 1.0

    logit = alpha + beta * np.log(error_power)

    # 分段计算 sigmoid(-logit)，避免 exp 在极端误差下溢出或溢出。
    if logit >= 0.0:
        exp_negative = np.exp(-logit)
        return float(exp_negative / (1.0 + exp_negative))
    exp_positive = np.exp(logit)
    return float(1.0 / (1.0 + exp_positive))


def gaw_midpoint_error_power(
    alpha: float = 27.0,
    beta: float = 15.0,
    error_power_scale: float = 1.0,
) -> float:
    """返回使公式 (14) 的 GAW 权重等于 0.5 的原始误差功率。

    由 `alpha + beta*log(scale*||e||^2)=0` 可得：
        ||e||^2_mid = exp(-alpha / beta) / scale
    """
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("GAW alpha 和 beta 必须为有限数")
    if beta == 0.0:
        raise ValueError("计算 GAW 中点要求 beta 非零")
    if not np.isfinite(error_power_scale) or error_power_scale <= 0.0:
        raise ValueError("GAW 误差功率尺度必须为有限正数")
    return float(np.exp(-alpha / beta) / error_power_scale)


def complex_to_real_matrix(x: np.ndarray) -> np.ndarray:
    """将 [T, D] 复数序列转成 [T, 2D] 实数序列。"""
    return np.concatenate([x.real, x.imag], axis=-1)


def real_to_complex_matrix(x: np.ndarray) -> np.ndarray:
    """将 [T, 2D] 实数序列还原为 [T, D] 复数序列。"""
    half = x.shape[-1] // 2
    return x[..., :half] + 1j * x[..., half:]


def make_windows(x: np.ndarray, window: int) -> np.ndarray:
    """构造线性移位寄存器窗口，当前样本在前，帧外历史补零。"""
    if window <= 0:
        raise ValueError("滑动窗口长度必须大于 0")
    if window <= 1:
        return x
    out = np.zeros((x.shape[0], x.shape[1] * window), dtype=x.dtype)
    for n in range(x.shape[0]):
        parts = []
        for k in range(window):
            idx = n - k
            if idx >= 0:
                parts.append(x[idx])
            else:
                parts.append(np.zeros(x.shape[1], dtype=x.dtype))
        out[n] = np.concatenate(parts)
    return out


def _readout_design(z: np.ndarray, include_bias: bool) -> np.ndarray:
    """按配置构造 readout 设计矩阵。"""
    if include_bias:
        return np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
    return z


def _regularized_gram_pseudoinverse(
    design: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """由 Z 的 SVD 稳定构造 `(Z^T Z + ridge I)^-1`。

    直接计算 `pinv(Z^T Z)` 会把 Z 的条件数平方，并可能让理论上应为
    半正定的 Ψ^-1 出现较大的负特征值。这里直接使用 Z 的奇异值：

        Z = U diag(s) V^T
        (Z^T Z)^dagger = V diag((s^2)^dagger) V^T

    `ridge>0` 时，零空间方向也按 `1/ridge` 保留。
    """
    _, singular_values, vh = np.linalg.svd(
        design,
        full_matrices=True,
    )
    n_features = vh.shape[0]
    gram_eigenvalues = np.zeros(n_features, dtype=float)
    gram_eigenvalues[: singular_values.size] = singular_values**2

    if ridge > 0.0:
        inverse_eigenvalues = 1.0 / (gram_eigenvalues + ridge)
    else:
        inverse_eigenvalues = np.zeros(n_features, dtype=float)
        if singular_values.size:
            # 与公式 (9) 所用 np.linalg.pinv 的默认相对阈值保持一致。
            cutoff = 1e-15 * singular_values[0]
            retained = singular_values > cutoff
            retained_indices = np.flatnonzero(retained)
            inverse_eigenvalues[retained_indices] = (
                1.0 / singular_values[retained_indices] ** 2
            )

    right_vectors = vh.conj().T
    p_matrix = (
        right_vectors * inverse_eigenvalues[None, :]
    ) @ right_vectors.conj().T
    # 消除浮点舍入造成的非对称分量，保持 RLS 相关矩阵的结构。
    return (p_matrix + p_matrix.conj().T) / 2.0


def train_ridge_readout(
    z: np.ndarray,
    labels: np.ndarray,
    ridge: float = 1e-6,
    include_bias: bool = True,
):
    """训练线性 readout。

    当 `ridge=0` 且 `include_bias=False` 时，严格对应论文公式 (9)：
    W_out = L Z^dagger。本项目按“样本为行”存储，因此实现为
    `pinv(Z) @ L`。
    """
    if ridge < 0.0:
        raise ValueError("ridge 不能为负数")

    design = _readout_design(z, include_bias)
    if ridge == 0.0:
        return np.linalg.pinv(design) @ labels

    gram = design.T @ design
    reg = ridge * np.eye(gram.shape[0])
    weights = np.linalg.solve(gram + reg, design.T @ labels)
    return weights


def apply_readout(
    z: np.ndarray,
    weights: np.ndarray,
    include_bias: bool = True,
) -> np.ndarray:
    """使用线性 readout 进行预测。"""
    return _readout_design(z, include_bias) @ weights


class RLSReadout:
    """支持标准 RLS 与 GAW-RLS 的 readout 更新器。

    启用偏置时权重形状为 [Nz + 1, No]，否则为 [Nz, No]。
    `adaptive_weighting=True` 时，实现论文公式 (14) 的加权损失；
    令所有样本权重 omega=1 时，GAW-RLS 退化为标准 RLS。
    """

    def __init__(
        self,
        weights: np.ndarray,
        p_matrix: np.ndarray,
        forgetting: float = 0.9995,
        adaptive_weighting: bool = True,
        gaw_alpha: float = 27.0,
        gaw_beta: float = 15.0,
        gaw_error_power_scale: float = 1.0,
        gaw_error_power_mode: str = "sum",
        gaw_weight_granularity: str = "global",
        include_bias: bool = True,
    ):
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("遗忘因子 forgetting 必须位于 (0, 1]")
        if (
            not np.isfinite(gaw_error_power_scale)
            or gaw_error_power_scale <= 0.0
        ):
            raise ValueError("GAW 误差功率尺度必须为有限正数")
        normalized_power_mode = gaw_error_power_mode.lower()
        if normalized_power_mode not in GAW_ERROR_POWER_MODES:
            raise ValueError("gaw_error_power_mode 必须为 'sum' 或 'mean'")
        normalized_granularity = gaw_weight_granularity.lower()
        if normalized_granularity not in GAW_WEIGHT_GRANULARITIES:
            raise ValueError(
                "gaw_weight_granularity 必须为 'global' 或 'per_stream'"
            )
        if weights.ndim != 2:
            raise ValueError("readout 权重必须是二维矩阵")
        if p_matrix.ndim != 2 or p_matrix.shape != (
            weights.shape[0],
            weights.shape[0],
        ):
            raise ValueError("RLS 初始逆相关矩阵维度必须与 readout 特征一致")
        if normalized_granularity == "per_stream" and weights.shape[1] % 2:
            raise ValueError("按复数流分组要求 readout 输出维度为偶数")

        self.weights = weights
        self.forgetting = forgetting
        self.adaptive_weighting = adaptive_weighting
        self.gaw_alpha = gaw_alpha
        self.gaw_beta = gaw_beta
        self.gaw_error_power_scale = gaw_error_power_scale
        self.gaw_error_power_mode = normalized_power_mode
        self.gaw_weight_granularity = normalized_granularity
        self.include_bias = include_bias

        output_dim = weights.shape[1]
        if self.gaw_weight_granularity == "global":
            # 论文公式 (14)--(15)：完整输出误差共用一个标量 omega 和一份 P。
            self.output_groups = (np.arange(output_dim),)
            self.p_matrix = p_matrix
        else:
            # 实值输出排列为 [Re(x_0...x_Nt-1), Im(x_0...x_Nt-1)]。
            # 每个复数流需要独立 P_k，才能让不同 omega_k 在公式 (15)
            # 中形成自洽的独立递推；该分支仅是未公开实现的敏感性实验。
            stream_count = output_dim // 2
            self.output_groups = tuple(
                np.asarray((index, index + stream_count), dtype=int)
                for index in range(stream_count)
            )
            self.p_matrix = np.repeat(
                p_matrix[None, :, :],
                stream_count,
                axis=0,
            )

        self.group_count = len(self.output_groups)
        # Fig. 6 口径审计：公式 (11) 明确定义的是更新前误差，论文没有
        # 说明图中的 training MSE 是否也采用更新后的同样本残差。
        self.last_posterior_error = np.full(output_dim, np.nan, dtype=float)
        self.last_batch_posterior_errors = np.empty(
            (0, output_dim),
            dtype=float,
        )
        if self.gaw_weight_granularity == "global":
            self.last_weight = 1.0
            self.last_error_power = np.nan
            self.last_batch_weights = np.empty(0, dtype=float)
            self.last_batch_error_powers = np.empty(0, dtype=float)
        else:
            self.last_weight = np.ones(self.group_count, dtype=float)
            self.last_error_power = np.full(self.group_count, np.nan)
            self.last_batch_weights = np.empty(
                (0, self.group_count),
                dtype=float,
            )
            self.last_batch_error_powers = np.empty(
                (0, self.group_count),
                dtype=float,
            )

    @classmethod
    def from_batch(
        cls,
        z: np.ndarray,
        labels: np.ndarray,
        ridge: float = 1e-6,
        forgetting: float = 0.9995,
        adaptive_weighting: bool = True,
        gaw_alpha: float = 27.0,
        gaw_beta: float = 15.0,
        gaw_error_power_scale: float = 1.0,
        gaw_error_power_mode: str = "sum",
        gaw_weight_granularity: str = "global",
        include_bias: bool = True,
        covariance_mode: str = "ts_gram",
        initial_covariance_scale: float = 1.0,
    ):
        """用 TS 批量数据初始化 RLS/GAW-RLS 的权重与逆相关矩阵。

        论文公式 (12)--(15) 没有给出初始 Ψ^{-1}。`ts_gram` 使用 TS
        扩展状态相关矩阵的伪逆；`discounted_ts_gram` 按公式 (10)
        使用遗忘因子对较早的 TS 样本降权；`identity` 使用缩放单位阵。
        三者均作为可追溯的复现假设保留，公式 (9) 的初始 readout
        权重始终由未加权 TS 批量最小二乘得到。
        """
        if ridge < 0.0:
            raise ValueError("ridge 不能为负数")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("遗忘因子 forgetting 必须位于 (0, 1]")
        if initial_covariance_scale <= 0.0:
            raise ValueError("RLS 初始逆相关矩阵尺度必须大于 0")

        design = _readout_design(z, include_bias)
        normalized_mode = covariance_mode.lower()
        if normalized_mode == "ts_gram":
            p_matrix = _regularized_gram_pseudoinverse(design, ridge)
        elif normalized_mode == "discounted_ts_gram":
            # 论文公式 (10)：在 TS 末样本 n=M-1 处，
            # Ψ(n)=Σ_m λ^(M-1-m) z(m)z(m)^T。
            exponents = np.arange(
                design.shape[0] - 1,
                -1,
                -1,
                dtype=float,
            )
            discounts = np.power(forgetting, exponents)
            weighted_design = design * np.sqrt(discounts)[:, None]
            p_matrix = _regularized_gram_pseudoinverse(
                weighted_design,
                ridge,
            )
        elif normalized_mode == "identity":
            covariance_dtype = np.result_type(design.dtype, np.float64)
            p_matrix = np.eye(design.shape[1], dtype=covariance_dtype)
        else:
            raise ValueError(
                "covariance_mode 必须为 'ts_gram'、"
                "'discounted_ts_gram' 或 'identity'"
            )
        p_matrix *= initial_covariance_scale
        weights = train_ridge_readout(
            z,
            labels,
            ridge=ridge,
            include_bias=include_bias,
        )
        return cls(
            weights=weights,
            p_matrix=p_matrix,
            forgetting=forgetting,
            adaptive_weighting=adaptive_weighting,
            gaw_alpha=gaw_alpha,
            gaw_beta=gaw_beta,
            gaw_error_power_scale=gaw_error_power_scale,
            gaw_error_power_mode=gaw_error_power_mode,
            gaw_weight_granularity=gaw_weight_granularity,
            include_bias=include_bias,
        )

    def predict(self, z: np.ndarray) -> np.ndarray:
        """对一批 reservoir 特征进行线性预测。"""
        return apply_readout(z, self.weights, include_bias=self.include_bias)

    def update_one(
        self,
        z_t: np.ndarray,
        label_t: np.ndarray,
        weight: float | np.ndarray | None = None,
    ):
        """使用一个样本执行公式 (11)、(13)、(15) 的递归更新。

        未显式传入 `weight` 且启用自适应加权时，先根据旧 readout 的
        预测误差计算论文权重；显式设置 `weight=1` 即执行标准 RLS。
        """
        if self.include_bias:
            x = np.concatenate([z_t, np.ones(1)])
        else:
            x = z_t
        # 论文公式 (11) 中的先验预测误差 e_{n-1}(n)。
        error = label_t - x @ self.weights

        error_powers = np.asarray(
            [
                gaw_error_power(
                    error[output_indices],
                    mode=self.gaw_error_power_mode,
                )
                for output_indices in self.output_groups
            ],
            dtype=float,
        )
        if weight is None:
            if self.adaptive_weighting:
                used_weights = np.asarray(
                    [
                        gaw_weight_from_error(
                            error[output_indices],
                            alpha=self.gaw_alpha,
                            beta=self.gaw_beta,
                            error_power_scale=(
                                self.gaw_error_power_scale
                            ),
                            error_power_mode=self.gaw_error_power_mode,
                        )
                        for output_indices in self.output_groups
                    ],
                    dtype=float,
                )
            else:
                used_weights = np.ones(self.group_count, dtype=float)
        else:
            explicit_weights = np.asarray(weight, dtype=float)
            if explicit_weights.ndim == 0:
                used_weights = np.full(
                    self.group_count,
                    float(explicit_weights),
                )
            elif explicit_weights.shape == (self.group_count,):
                used_weights = explicit_weights
            else:
                raise ValueError(
                    "显式 GAW-RLS 权重必须为标量或与权重组数相同的一维向量"
                )
        if (
            not np.all(np.isfinite(used_weights))
            or np.any(used_weights < 0.0)
            or np.any(used_weights > 1.0)
        ):
            raise ValueError("GAW-RLS 样本权重必须位于 [0, 1]")

        for group_index, (output_indices, group_weight) in enumerate(
            zip(self.output_groups, used_weights)
        ):
            if self.gaw_weight_granularity == "global":
                p_matrix = self.p_matrix
            else:
                p_matrix = self.p_matrix[group_index]
            px = p_matrix @ x
            # 论文公式 (15)：GAW-RLS 增益；omega=1 时退化为公式 (12)。
            denom = self.forgetting + group_weight * float(x.T @ px)
            gain = group_weight * px / denom

            # 论文公式 (11)：利用当前组的先验误差更新对应输出列。
            self.weights[:, output_indices] = (
                self.weights[:, output_indices]
                + np.outer(gain, error[output_indices])
            )

            # 论文公式 (13)：递归更新该权重组的逆加权相关矩阵。
            updated_p = (
                p_matrix - np.outer(gain, x.T @ p_matrix)
            ) / self.forgetting
            if self.gaw_weight_granularity == "global":
                self.p_matrix = updated_p
            else:
                self.p_matrix[group_index] = updated_p

        if self.gaw_weight_granularity == "global":
            self.last_weight = float(used_weights[0])
            self.last_error_power = float(error_powers[0])
        else:
            self.last_weight = used_weights.copy()
            self.last_error_power = error_powers.copy()
        # 该量不是论文公式 (11) 的 e_{n-1}(n)，而是当前样本完成
        # 公式 (11)、(13)、(15) 更新后，由 W_out(n) 得到的后验残差。
        self.last_posterior_error = label_t - x @ self.weights
        return error

    def update_batch(self, z: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None):
        """逐样本更新一批训练数据，并记录每个样本的实际权重。"""
        if z.shape[0] != labels.shape[0]:
            raise ValueError("特征与标签的样本数必须相同")
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim not in (1, 2) or weights.shape[0] != z.shape[0]:
                raise ValueError("显式样本权重的数量必须与训练样本数相同")

        errors = []
        posterior_errors = []
        used_weights = []
        used_error_powers = []
        if weights is None:
            weights = [None] * z.shape[0]
        for z_t, label_t, weight in zip(z, labels, weights):
            errors.append(self.update_one(z_t, label_t, weight=weight))
            posterior_errors.append(self.last_posterior_error.copy())
            used_weights.append(self.last_weight)
            used_error_powers.append(self.last_error_power)
        self.last_batch_posterior_errors = np.asarray(
            posterior_errors,
            dtype=float,
        ).reshape((-1, self.weights.shape[1]))
        if used_weights:
            self.last_batch_weights = np.asarray(used_weights, dtype=float)
            self.last_batch_error_powers = np.asarray(
                used_error_powers,
                dtype=float,
            )
        elif self.gaw_weight_granularity == "global":
            self.last_batch_weights = np.empty(0, dtype=float)
            self.last_batch_error_powers = np.empty(0, dtype=float)
        else:
            empty_shape = (0, self.group_count)
            self.last_batch_weights = np.empty(empty_shape, dtype=float)
            self.last_batch_error_powers = np.empty(
                empty_shape,
                dtype=float,
            )
        return np.asarray(errors)
