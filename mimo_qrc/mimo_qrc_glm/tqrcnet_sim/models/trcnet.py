import numpy as np

from tqrcnet_sim.mimo_ofdm.frame import flatten_time_symbols
from tqrcnet_sim.mimo_ofdm.metrics import bit_error_counts, detect_data_bits
from tqrcnet_sim.mimo_ofdm.pilot_extract import pilot_time_training_tuple
from tqrcnet_sim.models.readout import (
    apply_readout,
    complex_to_real_matrix,
    make_windows,
    real_to_complex_matrix,
    RLSReadout,
    train_ridge_readout,
)


class TRCNetDetector:
    """Algorithm 1 外层流程。

    `reservoir` 可以是单个后端，也可以是按层排列的后端列表。论文实验使用
    V=2：第 1 层输入接收信号 y，第 v>1 层输入前一层输出 x_hat^(v-1)。
    每层都先通过 TS 初始化 readout，再通过数据符号中的 pilot 在线更新。
    """

    def __init__(
        self,
        reservoir,
        cfg,
        window: int = 4,
        ridge: float = 1e-6,
        forgetting: float = 0.9995,
        use_gaw_rls: bool = True,
        gaw_alpha: float = 27.0,
        gaw_beta: float = 15.0,
        gaw_error_power_scale: float = 1.0,
        gaw_error_power_mode: str = "sum",
        gaw_weight_granularity: str = "global",
        readout_bias: bool = False,
        delay_candidates: tuple[int, ...] = (0,),
        state_boundary_mode: str = "frame_continuous",
        ts_data_state_handoff: str = "reset",
        online_update_timing: str = "update_then_detect",
        pilot_state_handoff: str = "isolated_stream",
        pilot_training_stream_mode: str = "reset_each_symbol",
        input_normalization: str = "none",
        collect_online_diagnostics: bool = False,
        rls_covariance_mode: str = "ts_gram",
        rls_initial_covariance_scale: float = 1.0,
    ):
        if isinstance(reservoir, (list, tuple)):
            self.reservoirs = list(reservoir)
        else:
            # 保留原来的单层调用方式，便于开展 V=1 与 V=2 消融实验。
            self.reservoirs = [reservoir]
        if not self.reservoirs:
            raise ValueError("T-RCNet 至少需要一个 reservoir 层")

        self.n_layers = len(self.reservoirs)
        self.reservoir = self.reservoirs[0]
        self.cfg = cfg
        self.window = window
        self.ridge = ridge
        self.forgetting = forgetting
        self.use_gaw_rls = use_gaw_rls
        self.gaw_alpha = gaw_alpha
        self.gaw_beta = gaw_beta
        if (
            not np.isfinite(gaw_error_power_scale)
            or gaw_error_power_scale <= 0.0
        ):
            raise ValueError("GAW 误差功率尺度必须为有限正数")
        self.gaw_error_power_scale = gaw_error_power_scale
        if gaw_error_power_mode.lower() not in ("sum", "mean"):
            raise ValueError(
                "gaw_error_power_mode 必须为 'sum' 或 'mean'"
            )
        self.gaw_error_power_mode = gaw_error_power_mode.lower()
        if gaw_weight_granularity.lower() not in (
            "global",
            "per_stream",
        ):
            raise ValueError(
                "gaw_weight_granularity 必须为 'global' 或 'per_stream'"
            )
        self.gaw_weight_granularity = gaw_weight_granularity.lower()
        self.readout_bias = readout_bias
        if state_boundary_mode.lower() not in (
            "frame_continuous",
            "symbol_reset",
        ):
            raise ValueError(
                "state_boundary_mode 必须为 "
                "'frame_continuous' 或 'symbol_reset'"
            )
        self.state_boundary_mode = state_boundary_mode.lower()
        if ts_data_state_handoff.lower() not in (
            "reset",
            "inherit_ts",
        ):
            raise ValueError(
                "ts_data_state_handoff 必须为 'reset' 或 'inherit_ts'"
            )
        if (
            ts_data_state_handoff.lower() == "inherit_ts"
            and self.state_boundary_mode != "frame_continuous"
        ):
            raise ValueError(
                "inherit_ts 仅适用于 frame_continuous 状态边界"
            )
        self.ts_data_state_handoff = ts_data_state_handoff.lower()
        if online_update_timing.lower() not in (
            "update_then_detect",
            "detect_then_update",
        ):
            raise ValueError(
                "online_update_timing 必须为 "
                "'update_then_detect' 或 'detect_then_update'"
            )
        self.online_update_timing = online_update_timing.lower()
        if pilot_state_handoff.lower() not in (
            "isolated_stream",
            "inherit_pilot",
        ):
            raise ValueError(
                "pilot_state_handoff 必须为 "
                "'isolated_stream' 或 'inherit_pilot'"
            )
        self.pilot_state_handoff = pilot_state_handoff.lower()
        if pilot_training_stream_mode.lower() not in (
            "reset_each_symbol",
            "continuous",
        ):
            raise ValueError(
                "pilot_training_stream_mode 必须为 "
                "'reset_each_symbol' 或 'continuous'"
            )
        self.pilot_training_stream_mode = (
            pilot_training_stream_mode.lower()
        )
        if input_normalization.lower() not in (
            "none",
            "global_rms",
            "per_channel_rms",
        ):
            raise ValueError(
                "input_normalization 必须为 'none'、"
                "'global_rms' 或 'per_channel_rms'"
            )
        self.input_normalization = input_normalization.lower()
        self.collect_online_diagnostics = bool(collect_online_diagnostics)
        if rls_covariance_mode.lower() not in (
            "ts_gram",
            "discounted_ts_gram",
            "identity",
        ):
            raise ValueError(
                "rls_covariance_mode 必须为 'ts_gram'、"
                "'discounted_ts_gram' 或 'identity'"
            )
        if rls_initial_covariance_scale <= 0.0:
            raise ValueError("RLS 初始逆相关矩阵尺度必须大于 0")
        self.rls_covariance_mode = rls_covariance_mode.lower()
        self.rls_initial_covariance_scale = rls_initial_covariance_scale
        normalized_delays = tuple(sorted(set(int(p) for p in delay_candidates)))
        if not normalized_delays:
            raise ValueError("响应延迟候选集合不能为空")
        if any(p < 0 or p > cfg.ncp for p in normalized_delays):
            raise ValueError("响应延迟候选必须位于 [0, Ncp]")
        self.delay_candidates = normalized_delays
        self.weights = None
        self.rls = None
        self.layer_weights = [None] * self.n_layers
        self.layer_rls = [None] * self.n_layers
        self.ts_layer_weights = [None] * self.n_layers
        self.layer_delays = [0] * self.n_layers
        self.last_delay_objectives = [[] for _ in range(self.n_layers)]
        self.last_fit_states = []
        self.last_ts_layer_outputs = []
        self.last_online_layer_errors = []
        self.last_online_layer_outputs = []
        self.last_online_layer_diagnostics = []
        self.online_diagnostics = []
        self.layer_input_scales = [None] * self.n_layers
        self._ts_data_contexts = None
        self._online_contexts = None
        self._pilot_training_contexts = None

    def _fit_input_normalization(
        self,
        layer_index: int,
        input_time: np.ndarray,
    ) -> np.ndarray:
        """只用当前层 TS 输入估计后续固定复用的 RMS 尺度。"""
        seq = flatten_time_symbols(input_time)
        channel_power = np.mean(np.abs(seq) ** 2, axis=0)
        if self.input_normalization == "none":
            scales = np.ones(seq.shape[1], dtype=float)
        elif self.input_normalization == "global_rms":
            global_rms = float(np.sqrt(np.mean(channel_power)))
            scales = np.full(seq.shape[1], global_rms, dtype=float)
        else:
            scales = np.sqrt(channel_power).astype(float, copy=False)

        # 全零 TS 通道保持原值，避免除零，也不凭空放大无信息通道。
        scales = np.where(scales > 1e-12, scales, 1.0)
        self.layer_input_scales[layer_index] = scales
        return scales.copy()

    def _normalize_input_sequence(
        self,
        layer_index: int,
        seq: np.ndarray,
    ) -> np.ndarray:
        """按该层 TS 尺度归一化复数序列，不使用数据段统计量。"""
        if self.input_normalization == "none":
            return seq
        scales = self.layer_input_scales[layer_index]
        if scales is None:
            raise RuntimeError("输入归一化尺度尚未通过 TS 估计")
        if seq.shape[1] != scales.size:
            raise ValueError("输入通道数与 TS 归一化尺度维度不一致")
        return seq / scales[None, :]

    def _realify_input_sequence(
        self,
        layer_index: int,
        seq: np.ndarray,
    ) -> np.ndarray:
        """将归一化后的复数输入按 [Re(all), Im(all)] 展开。"""
        normalized = self._normalize_input_sequence(layer_index, seq)
        return complex_to_real_matrix(normalized)

    def _prepare_inputs(
        self,
        input_time: np.ndarray,
        tail_samples: int = 0,
        layer_index: int = 0,
    ) -> np.ndarray:
        """将复数时域输入转换为滑动窗口序列，并可在末尾补零。

        补零发生在构造窗口之前，因此末尾几个窗口仍会保留真实历史样本。
        这与线性移位寄存器的行为一致。
        """
        if tail_samples < 0:
            raise ValueError("末尾补零样本数不能为负数")
        seq = flatten_time_symbols(input_time)
        real_seq = self._realify_input_sequence(layer_index, seq)
        if tail_samples:
            real_seq = np.concatenate(
                [
                    real_seq,
                    np.zeros(
                        (tail_samples, real_seq.shape[1]),
                        dtype=real_seq.dtype,
                    ),
                ],
                axis=0,
            )
        return make_windows(real_seq, self.window)

    def _prepare_labels(self, x_time: np.ndarray) -> np.ndarray:
        seq = flatten_time_symbols(x_time)
        return complex_to_real_matrix(seq)

    def _fit_reservoir_input_encoder(
        self,
        layer_index: int,
        input_time: np.ndarray,
    ) -> None:
        """若后端需要线性输入编码，只使用当前层 TS 窗口完成拟合。"""
        fit_encoder = getattr(
            self.reservoirs[layer_index],
            "fit_input_encoder",
            None,
        )
        if fit_encoder is None:
            return
        if self.state_boundary_mode == "symbol_reset":
            inputs = np.concatenate(
                [
                    self._prepare_inputs(
                        symbol[None, :, :],
                        layer_index=layer_index,
                    )
                    for symbol in input_time
                ],
                axis=0,
            )
        else:
            inputs = self._prepare_inputs(
                input_time,
                layer_index=layer_index,
            )
        # DTC-QRC 论文在量子编码前使用 PCA。这里严格禁止用数据段
        # 或 pilot 在线样本重新拟合，避免测试信息泄漏。
        fit_encoder(inputs)

    def _forward_reservoir(
        self,
        layer_index: int,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """从当前内部状态运行指定层的 reservoir。"""
        reservoir = self.reservoirs[layer_index]
        expected_dim = getattr(reservoir, "input_dim", inputs.shape[1])
        if expected_dim != inputs.shape[1]:
            raise ValueError(
                f"第 {layer_index + 1} 层 reservoir 输入维度为 {expected_dim}，"
                f"但滑动窗口数据维度为 {inputs.shape[1]}"
            )
        return reservoir.forward_sequence(inputs)

    def _run_reservoir(self, layer_index: int, inputs: np.ndarray) -> np.ndarray:
        """重置并运行指定层的 reservoir，返回扩展状态 Z。"""
        self.reservoirs[layer_index].reset_state()
        return self._forward_reservoir(layer_index, inputs)

    def _snapshot_reservoir_state(self, layer_index: int):
        """复制 reservoir 动态状态，隔离 pilot 训练与真实信号推理轨迹。"""
        state = getattr(self.reservoirs[layer_index], "state", None)
        if state is None:
            return None
        return np.array(state, copy=True)

    def _restore_reservoir_state(self, layer_index: int, state):
        reservoir = self.reservoirs[layer_index]
        if state is None:
            reservoir.reset_state()
        else:
            reservoir.state = np.array(state, copy=True)

    def _stream_window_inputs(
        self,
        real_seq: np.ndarray,
        history: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """利用上一符号的真实历史构造连续滑动窗口。"""
        if history is None:
            history = np.empty((0, real_seq.shape[1]), dtype=real_seq.dtype)
        if history.ndim != 2 or history.shape[1] != real_seq.shape[1]:
            raise ValueError("在线窗口历史的维度与当前输入不一致")

        combined = np.concatenate([history, real_seq], axis=0)
        windows = make_windows(combined, self.window)[history.shape[0] :]
        history_size = max(self.window - 1, 0)
        if history_size:
            next_history = combined[-history_size:].copy()
        else:
            next_history = np.empty(
                (0, real_seq.shape[1]),
                dtype=real_seq.dtype,
            )
        return windows, next_history

    def _reset_online_contexts(self):
        """初始化各层仅供真实接收流使用的连续推理状态。"""
        contexts = []
        for layer_index, reservoir in enumerate(self.reservoirs):
            reservoir.reset_state()
            contexts.append(
                {
                    "state": self._snapshot_reservoir_state(layer_index),
                    "history": None,
                }
            )
        self._online_contexts = contexts

    def _initialize_data_contexts(self):
        """按配置从零状态或 TS 末态初始化完整数据接收流。"""
        if self.ts_data_state_handoff == "reset":
            self._reset_online_contexts()
            return
        if self._ts_data_contexts is None:
            raise RuntimeError("TS 状态尚未通过 fit_ts 建立，无法交接到数据段")

        self._online_contexts = []
        for context in self._ts_data_contexts:
            state = context["state"]
            history = context["history"]
            self._online_contexts.append(
                {
                    "state": (
                        None if state is None else np.array(state, copy=True)
                    ),
                    "history": (
                        None if history is None else np.array(history, copy=True)
                    ),
                }
            )

    def _reset_pilot_training_contexts(self):
        """初始化各层独立的 pilot-only 连续训练状态。"""
        contexts = []
        for layer_index, reservoir in enumerate(self.reservoirs):
            reservoir.reset_state()
            contexts.append(
                {
                    "state": self._snapshot_reservoir_state(layer_index),
                    "history": None,
                }
            )
        self._pilot_training_contexts = contexts

    def _pilot_streaming_states(
        self,
        layer_index: int,
        input_pilot: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        """连续处理一个 `Φ_i`，并返回与公式 (45) 标签对齐的状态。

        相邻 `Φ_i` 共享 pilot-only recurrent state 和滑动窗口历史，但
        与完整接收信号的在线推理上下文相互隔离。响应延迟需要的 p 个
        尾部零输入只用于临时生成 `Z[n+p]`，不会写回下一 `Φ_i` 的初态。
        """
        if input_pilot.ndim != 2:
            raise ValueError("pilot 训练输入必须具有 [Nchannel, Ntd] 维度")
        if input_pilot.shape[1] != self.cfg.ntd:
            raise ValueError("pilot 训练输入的时域长度必须等于 Ntd")
        if delay < 0:
            raise ValueError("响应延迟不能为负数")
        if self._pilot_training_contexts is None:
            self._reset_pilot_training_contexts()

        context = self._pilot_training_contexts[layer_index]
        real_seq = self._realify_input_sequence(
            layer_index,
            input_pilot.T,
        )
        inputs, next_history = self._stream_window_inputs(
            real_seq,
            context["history"],
        )
        self._restore_reservoir_state(layer_index, context["state"])
        actual_states = self._forward_reservoir(layer_index, inputs)
        actual_end_state = self._snapshot_reservoir_state(layer_index)

        if delay:
            tail_real = np.zeros(
                (delay, real_seq.shape[1]),
                dtype=real_seq.dtype,
            )
            tail_inputs, _ = self._stream_window_inputs(
                tail_real,
                next_history,
            )
            tail_states = self._forward_reservoir(
                layer_index,
                tail_inputs,
            )
            states = np.concatenate([actual_states, tail_states], axis=0)
        else:
            states = actual_states

        # 延迟补零只是当前 Φ_i 的对齐工具，下一 Φ_i 从真实 pilot 末态继续。
        self._restore_reservoir_state(layer_index, actual_end_state)
        context["state"] = actual_end_state
        context["history"] = next_history
        return states[delay : delay + self.cfg.ntd]

    def _pilot_training_states(
        self,
        layer_index: int,
        input_pilot: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        """按配置生成当前公式 (43)--(45) 训练元组的扩展状态。"""
        if self.pilot_training_stream_mode == "continuous":
            return self._pilot_streaming_states(
                layer_index,
                input_pilot,
                delay,
            )
        return self._aligned_states(
            layer_index,
            input_pilot[None, :, :],
            delay,
        )

    def _streaming_states(
        self,
        layer_index: int,
        input_symbol: np.ndarray,
        lookahead_time: np.ndarray | None = None,
        preview_samples: int = 0,
        inherit_current_state: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """连续处理一个真实 OFDM 符号，并生成当前与前视扩展状态。

        默认恢复真实接收流在上一符号末尾的状态；消融模式则保留调用
        本函数前的状态，即 pilot 短序列训练末态。两种模式的滑动窗口
        历史都来自真实接收流。为补偿本层延迟并向下一层提供前视输出，
        会从当前真实末态临时处理下一符号开头的有限样本；临时演化
        结束后恢复真实末态，因此前视不会污染下一符号状态。
        """
        if input_symbol.shape[0] != 1:
            raise ValueError("在线连续状态接口每次只处理一个 OFDM 符号")
        if preview_samples < 0:
            raise ValueError("前视输出样本数不能为负数")

        if self.state_boundary_mode == "symbol_reset":
            # 符号重置假设：每个 OFDM 符号都是独立 reservoir 序列。
            # 与 RCNet 延迟训练集一致，在符号尾部补 p 个零输入；下一
            # 符号的真实样本不作为当前符号的延迟前视。
            delay = self.layer_delays[layer_index]
            inputs = self._prepare_inputs(
                input_symbol,
                tail_samples=delay + preview_samples,
                layer_index=layer_index,
            )
            if inherit_current_state:
                states = self._forward_reservoir(layer_index, inputs)
            else:
                states = self._run_reservoir(layer_index, inputs)
            current_states = states[delay : delay + self.cfg.ntd]
            preview_states = states[
                delay + self.cfg.ntd :
                delay + self.cfg.ntd + preview_samples
            ]
            return current_states, preview_states

        inherited_state = None
        if self._online_contexts is None and inherit_current_state:
            inherited_state = self._snapshot_reservoir_state(layer_index)
        if self._online_contexts is None:
            self._reset_online_contexts()
        if inherited_state is not None:
            self._restore_reservoir_state(layer_index, inherited_state)

        context = self._online_contexts[layer_index]
        seq = flatten_time_symbols(input_symbol)
        real_seq = self._realify_input_sequence(layer_index, seq)
        inputs, next_history = self._stream_window_inputs(
            real_seq,
            context["history"],
        )

        if not inherit_current_state:
            self._restore_reservoir_state(layer_index, context["state"])
        actual_states = self._forward_reservoir(layer_index, inputs)
        actual_end_state = self._snapshot_reservoir_state(layer_index)

        delay = self.layer_delays[layer_index]
        tail_count = delay + preview_samples
        if tail_count:
            if lookahead_time is None:
                lookahead_seq = np.empty(
                    (0, real_seq.shape[1]),
                    dtype=real_seq.dtype,
                )
            else:
                if lookahead_time.ndim != 2:
                    raise ValueError("前视输入必须具有 [Nchannel, Nsample] 维度")
                lookahead_seq = self._realify_input_sequence(
                    layer_index,
                    lookahead_time.T,
                )
                if lookahead_seq.shape[1] != real_seq.shape[1]:
                    raise ValueError("前视输入通道数与当前符号不一致")

            tail_real = np.zeros(
                (tail_count, real_seq.shape[1]),
                dtype=real_seq.dtype,
            )
            available = min(tail_count, lookahead_seq.shape[0])
            tail_real[:available] = lookahead_seq[:available]
            tail_inputs, _ = self._stream_window_inputs(
                tail_real,
                next_history,
            )
            tail_states = self._forward_reservoir(layer_index, tail_inputs)
            states = np.concatenate([actual_states, tail_states], axis=0)
        else:
            states = actual_states

        # 下一符号必须从真实输入末态继续，而不是从临时补零末态继续。
        self._restore_reservoir_state(layer_index, actual_end_state)
        context["state"] = actual_end_state
        context["history"] = next_history
        current_states = states[delay : delay + self.cfg.ntd]
        preview_states = states[
            delay + self.cfg.ntd : delay + self.cfg.ntd + preview_samples
        ]
        return current_states, preview_states

    def _aligned_states(
        self,
        layer_index: int,
        input_time: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        """生成与原始标签时间轴对齐的扩展状态。

        前序 RCNet 的 Algorithm 1 指出 reservoir 响应存在 lag-effect，并在
        0...Ncp 内搜索延迟 p。状态 Z[n+p] 对齐标签 L[n]；推理时采用同一
        左移操作。本函数把该继承机制用于 T-RCNet 的每一层。
        """
        if self.state_boundary_mode == "symbol_reset":
            symbol_states = []
            for symbol in input_time:
                inputs = self._prepare_inputs(
                    symbol[None, :, :],
                    tail_samples=delay,
                    layer_index=layer_index,
                )
                states = self._run_reservoir(layer_index, inputs)
                symbol_states.append(
                    states[delay : delay + self.cfg.ntd]
                )
            return np.concatenate(symbol_states, axis=0)

        inputs = self._prepare_inputs(
            input_time,
            tail_samples=delay,
            layer_index=layer_index,
        )
        states = self._run_reservoir(layer_index, inputs)
        sample_count = input_time.shape[0] * self.cfg.ntd
        return states[delay : delay + sample_count]

    @property
    def trainable_parameter_count(self) -> int:
        """返回所有 readout 的可训练参数总数。"""
        output_dim = 2 * self.cfg.nt
        return sum(
            (reservoir.feature_dim + int(self.readout_bias)) * output_dim
            for reservoir in self.reservoirs
        )

    def _real_output_to_time(self, pred_real: np.ndarray, n_symbols: int) -> np.ndarray:
        """将 readout 的 [Nsym*Ntd, 2Nt] 实数输出还原为时域 MIMO 符号。"""
        pred_complex = real_to_complex_matrix(pred_real)
        expected_samples = n_symbols * self.cfg.ntd
        if pred_complex.shape != (expected_samples, self.cfg.nt):
            raise ValueError(
                "readout 输出维度错误：每个时间采样应输出 Nt 个复数发送流"
            )
        return np.transpose(
            pred_complex.reshape(n_symbols, self.cfg.ntd, self.cfg.nt),
            (0, 2, 1),
        )

    def _fit_layer(
        self,
        layer_index: int,
        input_time: np.ndarray,
        label_time: np.ndarray,
    ) -> np.ndarray:
        """搜索响应延迟，并按论文公式 (9) 初始化一层 readout。"""
        if input_time.shape[0] != label_time.shape[0]:
            raise ValueError("一层训练的输入符号数与标签符号数必须相同")

        labels = self._prepare_labels(label_time)
        candidates = []
        for delay in self.delay_candidates:
            if self.state_boundary_mode == "symbol_reset":
                state_chunks = []
                label_chunks = []
                aligned_chunks = []
                for input_symbol, label_symbol in zip(
                    input_time,
                    label_time,
                ):
                    # 符号重置消融中，每个符号分别构造前序 RCNet 的
                    # 延迟训练集 [I_q, 0_p] -> [0_p, L_q]，随后汇总
                    # 各符号样本求解论文公式 (9)。
                    inputs = self._prepare_inputs(
                        input_symbol[None, :, :],
                        tail_samples=delay,
                        layer_index=layer_index,
                    )
                    states = self._run_reservoir(layer_index, inputs)
                    symbol_labels = self._prepare_labels(
                        label_symbol[None, :, :]
                    )
                    padded_symbol_labels = np.concatenate(
                        [
                            np.zeros(
                                (delay, symbol_labels.shape[1]),
                                dtype=symbol_labels.dtype,
                            ),
                            symbol_labels,
                        ],
                        axis=0,
                    )
                    state_chunks.append(states)
                    label_chunks.append(padded_symbol_labels)
                    aligned_chunks.append(
                        states[delay : delay + self.cfg.ntd]
                    )
                training_states = np.concatenate(state_chunks, axis=0)
                padded_labels = np.concatenate(label_chunks, axis=0)
                aligned_states = np.concatenate(aligned_chunks, axis=0)
            else:
                # 整帧连续假设只在帧开始时重置一次 reservoir。前序
                # RCNet 的延迟训练集为 [I, 0_p] -> [0_p, L]。
                inputs = self._prepare_inputs(
                    input_time,
                    tail_samples=delay,
                    layer_index=layer_index,
                )
                training_states = self._run_reservoir(
                    layer_index,
                    inputs,
                )
                padded_labels = np.concatenate(
                    [
                        np.zeros(
                            (delay, labels.shape[1]),
                            dtype=labels.dtype,
                        ),
                        labels,
                    ],
                    axis=0,
                )
                aligned_states = training_states[
                    delay : delay + labels.shape[0]
                ]

            weights = train_ridge_readout(
                training_states,
                padded_labels,
                ridge=self.ridge,
                include_bias=self.readout_bias,
            )
            valid_prediction = apply_readout(
                aligned_states,
                weights,
                include_bias=self.readout_bias,
            )
            # RCNet 公式 (6)：使用时域输出与标签的二范数误差选择 p*。
            objective = float(
                np.linalg.norm(labels - valid_prediction) ** 2
            )
            candidates.append(
                (
                    objective,
                    delay,
                    training_states,
                    padded_labels,
                    aligned_states,
                    weights,
                )
            )

        (
            _objective,
            delay,
            training_states,
            training_labels,
            aligned_states,
            weights,
        ) = min(
            candidates,
            key=lambda item: item[0],
        )
        rls = RLSReadout.from_batch(
            training_states,
            training_labels,
            ridge=self.ridge,
            forgetting=self.forgetting,
            adaptive_weighting=self.use_gaw_rls,
            gaw_alpha=self.gaw_alpha,
            gaw_beta=self.gaw_beta,
            gaw_error_power_scale=self.gaw_error_power_scale,
            gaw_error_power_mode=self.gaw_error_power_mode,
            gaw_weight_granularity=self.gaw_weight_granularity,
            include_bias=self.readout_bias,
            covariance_mode=self.rls_covariance_mode,
            initial_covariance_scale=self.rls_initial_covariance_scale,
        )
        self.layer_weights[layer_index] = weights
        self.layer_rls[layer_index] = rls
        self.layer_delays[layer_index] = delay
        self.last_delay_objectives[layer_index] = [
            (candidate_delay, candidate_objective)
            for candidate_objective, candidate_delay, _, _, _, _ in candidates
        ]
        # 对外仍返回与原始 TS 标签逐样本对齐的 Z[n+p]。
        return aligned_states

    def _predict_layer(self, layer_index: int, input_time: np.ndarray) -> np.ndarray:
        """令指定层对完整时域符号进行推理。"""
        weights = self.layer_weights[layer_index]
        if weights is None:
            raise RuntimeError(f"第 {layer_index + 1} 层尚未通过 fit_ts 训练")
        z = self._aligned_states(
            layer_index,
            input_time,
            self.layer_delays[layer_index],
        )
        pred_real = apply_readout(
            z,
            weights,
            include_bias=self.readout_bias,
        )
        return self._real_output_to_time(pred_real, input_time.shape[0])

    def _predict_ts_layer_and_capture_context(
        self,
        layer_index: int,
        input_time: np.ndarray,
    ) -> np.ndarray:
        """推理一层 TS，并保存不含延迟补零污染的真实流末态。"""
        weights = self.layer_weights[layer_index]
        if weights is None:
            raise RuntimeError(f"第 {layer_index + 1} 层尚未通过 fit_ts 训练")
        if self._ts_data_contexts is None:
            raise RuntimeError("TS→数据状态交接上下文尚未初始化")

        seq = flatten_time_symbols(input_time)
        real_seq = self._realify_input_sequence(layer_index, seq)
        inputs, next_history = self._stream_window_inputs(real_seq, None)
        self.reservoirs[layer_index].reset_state()
        actual_states = self._forward_reservoir(layer_index, inputs)
        actual_end_state = self._snapshot_reservoir_state(layer_index)

        delay = self.layer_delays[layer_index]
        if delay:
            # RCNet 的 Z[n+p] 需要临时尾部样本；这里沿用既有零补齐口径，
            # 但恢复真实 TS 末态，避免把人工补零交接到首个数据符号。
            tail_real = np.zeros(
                (delay, real_seq.shape[1]),
                dtype=real_seq.dtype,
            )
            tail_inputs, _ = self._stream_window_inputs(
                tail_real,
                next_history,
            )
            tail_states = self._forward_reservoir(layer_index, tail_inputs)
            states = np.concatenate([actual_states, tail_states], axis=0)
        else:
            states = actual_states

        self._restore_reservoir_state(layer_index, actual_end_state)
        self._ts_data_contexts[layer_index] = {
            "state": actual_end_state,
            "history": next_history,
        }
        aligned_states = states[delay : delay + seq.shape[0]]
        pred_real = apply_readout(
            aligned_states,
            weights,
            include_bias=self.readout_bias,
        )
        return self._real_output_to_time(pred_real, input_time.shape[0])

    def _predict_layer_streaming(
        self,
        layer_index: int,
        input_symbol: np.ndarray,
        lookahead_time: np.ndarray | None = None,
        preview_samples: int = 0,
        inherit_current_state: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """连续推理当前符号，并为下一层生成有限前视输出。"""
        weights = self.layer_weights[layer_index]
        if weights is None:
            raise RuntimeError(f"第 {layer_index + 1} 层尚未通过 fit_ts 训练")
        z, preview_z = self._streaming_states(
            layer_index,
            input_symbol,
            lookahead_time=lookahead_time,
            preview_samples=preview_samples,
            inherit_current_state=inherit_current_state,
        )
        pred_real = apply_readout(
            z,
            weights,
            include_bias=self.readout_bias,
        )
        output = self._real_output_to_time(pred_real, 1)
        if preview_samples:
            preview_real = apply_readout(
                preview_z,
                weights,
                include_bias=self.readout_bias,
            )
            preview = real_to_complex_matrix(preview_real).T
        else:
            preview = np.empty(
                (self.cfg.nt, 0),
                dtype=np.complex128,
            )
        return output, preview

    def _sync_final_readout(self):
        """维护旧版单层接口：weights 与 rls 始终指向最后一层。"""
        self.weights = self.layer_weights[-1]
        self.rls = self.layer_rls[-1]

    def fit_ts(self, y_ts: np.ndarray, x_ts: np.ndarray):
        """按 Algorithm 1 第 2--6 行逐层完成 TS 初始化训练。

        第 1 层训练元组为公式 (37) 的 {y_TS, x_TS}；第 v>1 层将
        前一层在 TS 上的推理结果作为输入，监督标签始终为 x_TS。
        """
        labels = self._prepare_labels(x_ts)
        layer_input = y_ts
        self.last_fit_states = []
        self.last_ts_layer_outputs = []
        self._ts_data_contexts = (
            [None] * self.n_layers
            if self.ts_data_state_handoff == "inherit_ts"
            else None
        )
        self._online_contexts = None
        self._pilot_training_contexts = None
        self.layer_input_scales = [None] * self.n_layers

        for layer_index in range(self.n_layers):
            self._fit_input_normalization(layer_index, layer_input)
            self._fit_reservoir_input_encoder(layer_index, layer_input)
            z = self._fit_layer(layer_index, layer_input, x_ts)
            self.last_fit_states.append(z)

            # Algorithm 1 第 6 行：当前层推理 TS，并送入下一层。
            if self.ts_data_state_handoff == "inherit_ts":
                layer_output = self._predict_ts_layer_and_capture_context(
                    layer_index,
                    layer_input,
                )
            else:
                layer_output = self._predict_layer(layer_index, layer_input)
            self.last_ts_layer_outputs.append(layer_output)
            layer_input = layer_output

        # 保存 TS 初始化点，供在线阶段量化 readout 的累计漂移。
        self.ts_layer_weights = [
            np.array(weights, copy=True)
            for weights in self.layer_weights
        ]
        self._sync_final_readout()
        return self.last_fit_states[-1], labels

    def update_with_pilots_batch(self, y_data: np.ndarray, x_freq_data: np.ndarray):
        """使用 pilot 提取结果逐层做批量再训练。

        这个函数保留作对照实验。正式在线流程请使用 detect_data_online。
        """
        if y_data.shape[0] == 0:
            return None

        layer_input = y_data
        final_z = None
        final_labels = None
        for layer_index in range(self.n_layers):
            pilot_inputs = []
            pilot_labels = []
            for i in range(y_data.shape[0]):
                input_pilot, label_pilot = pilot_time_training_tuple(
                    layer_input[i],
                    x_freq_data[i],
                    self.cfg,
                )
                pilot_inputs.append(input_pilot)
                pilot_labels.append(label_pilot)

            pilot_inputs = np.asarray(pilot_inputs)
            pilot_labels = np.asarray(pilot_labels)
            final_z = self._fit_layer(
                layer_index,
                pilot_inputs,
                pilot_labels,
            )
            final_labels = self._prepare_labels(pilot_labels)
            layer_input = self._predict_layer(layer_index, layer_input)

        self._sync_final_readout()
        return final_z, final_labels

    def update_with_pilots_online(
        self,
        y_symbol: np.ndarray,
        x_freq_symbol: np.ndarray,
        lookahead_time: np.ndarray | None = None,
    ):
        """对一个数据 OFDM 符号逐层执行 GAW-RLS 更新与推理。

        第 1 层从 y_i 提取训练输入；第 v>1 层从 x_hat_i^(v-1)
        提取训练输入。所有层共享公式 (45) 构造的 pilot 标签。
        """
        if any(rls is None for rls in self.layer_rls):
            raise RuntimeError("需要先调用 fit_ts 初始化 GAW-RLS readout")

        layer_input_symbol = y_symbol
        self.last_online_layer_errors = []
        self.last_online_layer_outputs = []
        self.last_online_layer_diagnostics = []
        layer_lookahead = lookahead_time

        # Algorithm 1 第 8--12 行：对当前符号按 v=1,...,V 顺序处理。
        for layer_index in range(self.n_layers):
            input_pilot, label_pilot = pilot_time_training_tuple(
                layer_input_symbol,
                x_freq_symbol,
                self.cfg,
            )
            labels = self._prepare_labels(label_pilot[None, :, :])
            z = self._pilot_training_states(
                layer_index,
                input_pilot,
                self.layer_delays[layer_index],
            )

            rls = self.layer_rls[layer_index]
            if self.collect_online_diagnostics:
                weights_before = np.array(
                    self.layer_weights[layer_index],
                    copy=True,
                )
                p_matrix_before = getattr(rls, "p_matrix", None)
                if p_matrix_before is not None:
                    p_matrix_before = np.array(p_matrix_before, copy=True)
            if self.online_update_timing == "update_then_detect":
                # Algorithm 1 第 11--12 行：先按公式 (11)、(13)、(15)
                # 更新 W_out，再用新权重检测当前完整 OFDM 符号。
                errors = rls.update_batch(z, labels)
                self.layer_weights[layer_index] = rls.weights

            if self.state_boundary_mode == "frame_continuous":
                preview_samples = sum(
                    self.layer_delays[layer_index + 1 :]
                )
            else:
                # 符号重置模式下，下一层也在符号尾部补零，不跨边界传递前视。
                preview_samples = 0
            layer_output, layer_preview = self._predict_layer_streaming(
                layer_index,
                layer_input_symbol[None, :, :],
                lookahead_time=layer_lookahead,
                preview_samples=preview_samples,
                inherit_current_state=(
                    self.pilot_state_handoff == "inherit_pilot"
                ),
            )
            layer_output = layer_output[0]

            if self.online_update_timing == "detect_then_update":
                # 诊断模式：当前输出仍使用 W_out(n-1)，当前 Φ_i 的
                # GAW-RLS 更新只从下一个 OFDM 符号开始生效。
                errors = rls.update_batch(z, labels)
                self.layer_weights[layer_index] = rls.weights

            if self.collect_online_diagnostics:
                weights_after = np.array(
                    self.layer_weights[layer_index],
                    copy=True,
                )
                weight_delta_fro = float(
                    np.linalg.norm(weights_after - weights_before)
                )
                weight_before_fro = float(np.linalg.norm(weights_before))
                weight_after_fro = float(np.linalg.norm(weights_after))
                reference_weights = self.ts_layer_weights[layer_index]
                if reference_weights is None:
                    reference_weights = weights_before
                distance_from_ts_fro = float(
                    np.linalg.norm(weights_after - reference_weights)
                )
                ts_weight_fro = float(np.linalg.norm(reference_weights))

                p_matrix_after = getattr(rls, "p_matrix", None)
                if p_matrix_before is None or p_matrix_after is None:
                    p_before_fro = np.nan
                    p_after_fro = np.nan
                    p_delta_fro = np.nan
                else:
                    p_matrix_after = np.asarray(p_matrix_after)
                    p_before_fro = float(np.linalg.norm(p_matrix_before))
                    p_after_fro = float(np.linalg.norm(p_matrix_after))
                    p_delta_fro = float(
                        np.linalg.norm(p_matrix_after - p_matrix_before)
                    )

                sample_weights = getattr(rls, "last_batch_weights", None)
                if sample_weights is None:
                    sample_weights = np.full(z.shape[0], np.nan)
                error_power = getattr(
                    rls,
                    "last_batch_error_powers",
                    None,
                )
                if error_power is None:
                    error_power = np.sum(np.square(errors), axis=1)
                # Fig. 6 的完整 2Nt 输出误差平方和独立保存，不随 P1-2 的
                # GAW 分组或 mean 口径改变。
                full_error_power = np.sum(np.square(errors), axis=1)
                posterior_errors = getattr(
                    rls,
                    "last_batch_posterior_errors",
                    None,
                )
                if posterior_errors is None:
                    posterior_full_error_power = np.full(
                        z.shape[0],
                        np.nan,
                    )
                else:
                    posterior_full_error_power = np.sum(
                        np.square(posterior_errors),
                        axis=1,
                    )
                self.last_online_layer_diagnostics.append(
                    {
                        "gaw_weights": np.asarray(
                            sample_weights,
                            dtype=float,
                        ).copy(),
                        "error_power": np.asarray(
                            error_power,
                            dtype=float,
                        ).copy(),
                        "full_error_power": np.asarray(
                            full_error_power,
                            dtype=float,
                        ).copy(),
                        # 明确别名：公式 (11) 的 e_{n-1}(n)，用于 GAW。
                        "prior_full_error_power": np.asarray(
                            full_error_power,
                            dtype=float,
                        ).copy(),
                        # 同一样本使用更新后的 W_out(n) 重新预测所得残差；
                        # 仅供 Fig. 6 定义敏感性审计，不参与 GAW 或 BER。
                        "posterior_full_error_power": np.asarray(
                            posterior_full_error_power,
                            dtype=float,
                        ).copy(),
                        # 公式 (14) 实际取对数的量；默认尺度 1 时与原始
                        # 预测误差功率完全相同。
                        "gaw_scaled_error_power": np.asarray(
                            self.gaw_error_power_scale * error_power,
                            dtype=float,
                        ).copy(),
                        "weight_before_fro": weight_before_fro,
                        "weight_after_fro": weight_after_fro,
                        "weight_delta_fro": weight_delta_fro,
                        "relative_weight_delta": (
                            weight_delta_fro
                            / max(weight_before_fro, np.finfo(float).eps)
                        ),
                        "distance_from_ts_fro": distance_from_ts_fro,
                        "relative_distance_from_ts": (
                            distance_from_ts_fro
                            / max(ts_weight_fro, np.finfo(float).eps)
                        ),
                        "p_before_fro": p_before_fro,
                        "p_after_fro": p_after_fro,
                        "p_delta_fro": p_delta_fro,
                    }
                )
            self.last_online_layer_errors.append(errors)
            self.last_online_layer_outputs.append(layer_output)
            layer_input_symbol = layer_output
            layer_lookahead = layer_preview

        self._sync_final_readout()
        # 保留原单层方法的返回语义；逐层误差可从 last_online_layer_errors 查看。
        return self.last_online_layer_errors[-1]

    def predict_time_layers(self, y_time: np.ndarray) -> list[np.ndarray]:
        """依次返回所有 T-RCNet 层的完整时域输出。"""
        if any(weights is None for weights in self.layer_weights):
            raise RuntimeError("需要先调用 fit_ts 训练 readout")

        if self.ts_data_state_handoff == "inherit_ts":
            return self._predict_time_layers_from_ts_context(y_time)

        layer_input = y_time
        layer_outputs = []
        for layer_index in range(self.n_layers):
            layer_input = self._predict_layer(layer_index, layer_input)
            layer_outputs.append(layer_input)
        return layer_outputs

    def _predict_time_layers_from_ts_context(
        self,
        y_time: np.ndarray,
    ) -> list[np.ndarray]:
        """从各层 TS 末态开始，逐符号执行不更新权重的连续推理。"""
        self._initialize_data_contexts()
        outputs_by_layer = [[] for _ in range(self.n_layers)]
        for symbol_index in range(y_time.shape[0]):
            total_delay = sum(self.layer_delays)
            if symbol_index + 1 < y_time.shape[0] and total_delay:
                layer_lookahead = y_time[
                    symbol_index + 1,
                    :,
                    :total_delay,
                ]
            else:
                layer_lookahead = None
            layer_input_symbol = y_time[symbol_index]

            for layer_index in range(self.n_layers):
                preview_samples = sum(
                    self.layer_delays[layer_index + 1 :]
                )
                layer_output, layer_preview = self._predict_layer_streaming(
                    layer_index,
                    layer_input_symbol[None, :, :],
                    lookahead_time=layer_lookahead,
                    preview_samples=preview_samples,
                )
                layer_input_symbol = layer_output[0]
                layer_lookahead = layer_preview
                outputs_by_layer[layer_index].append(layer_input_symbol)

        return [np.asarray(outputs) for outputs in outputs_by_layer]

    def predict_time(self, y_time: np.ndarray) -> np.ndarray:
        """对接收时域 OFDM 符号逐层检测，返回第 V 层输出。"""
        return self.predict_time_layers(y_time)[-1]

    def detect_data_online(self, y_data: np.ndarray, x_freq_data: np.ndarray) -> np.ndarray:
        """按照 Algorithm 1 的顺序逐个数据 OFDM 符号更新并检测。

        每个符号内部严格执行 v=1,...,V 的更新与推理，最终输出第 V 层结果。
        """
        if self.state_boundary_mode == "frame_continuous":
            # 数据符号构成同一连续接收流；帧开始状态由 P0-1 消融配置决定。
            self._initialize_data_contexts()
        else:
            self._online_contexts = None
        self._pilot_training_contexts = None
        if self.pilot_training_stream_mode == "continuous":
            self._reset_pilot_training_contexts()
        self.online_diagnostics = []
        outputs = []
        for i in range(y_data.shape[0]):
            total_delay = sum(self.layer_delays)
            if (
                self.state_boundary_mode == "frame_continuous"
                and i + 1 < y_data.shape[0]
                and total_delay
            ):
                lookahead_time = y_data[i + 1, :, :total_delay]
            else:
                lookahead_time = None
            self.update_with_pilots_online(
                y_data[i],
                x_freq_data[i],
                lookahead_time=lookahead_time,
            )
            if self.collect_online_diagnostics:
                self.online_diagnostics.append(
                    {
                        "data_symbol_index": i,
                        "layers": self.last_online_layer_diagnostics,
                    }
                )
            outputs.append(self.last_online_layer_outputs[-1])
        return np.asarray(outputs)

    def evaluate_error_counts(
        self,
        y_data: np.ndarray,
        true_bits: dict,
        data_mask: np.ndarray,
    ) -> tuple[int, int]:
        """检测数据符号并返回错误比特数和总比特数。"""
        x_hat = self.predict_time(y_data)
        pred_bits = detect_data_bits(
            x_hat,
            true_bits,
            data_mask,
            self.cfg.ncp,
            symbol_offset=self.cfg.n_ts,
            transform_normalization=(
                self.cfg.transform_normalization
            ),
        )
        return bit_error_counts(pred_bits, true_bits)

    def evaluate_ber(
        self,
        y_data: np.ndarray,
        true_bits: dict,
        data_mask: np.ndarray,
    ) -> float:
        """检测数据符号并计算 BER。"""
        errors, total = self.evaluate_error_counts(y_data, true_bits, data_mask)
        return errors / max(total, 1)

    def evaluate_error_counts_online(
        self,
        y_data: np.ndarray,
        x_freq_data: np.ndarray,
        true_bits: dict,
        data_mask: np.ndarray,
    ) -> tuple[int, int]:
        """使用在线 pilot 更新流程并返回错误比特数和总比特数。"""
        x_hat = self.detect_data_online(y_data, x_freq_data)
        pred_bits = detect_data_bits(
            x_hat,
            true_bits,
            data_mask,
            self.cfg.ncp,
            symbol_offset=self.cfg.n_ts,
            transform_normalization=(
                self.cfg.transform_normalization
            ),
        )
        return bit_error_counts(pred_bits, true_bits)

    def evaluate_ber_online(
        self,
        y_data: np.ndarray,
        x_freq_data: np.ndarray,
        true_bits: dict,
        data_mask: np.ndarray,
    ) -> float:
        """使用在线 pilot 更新流程检测数据符号并计算 BER。"""
        errors, total = self.evaluate_error_counts_online(
            y_data,
            x_freq_data,
            true_bits,
            data_mask,
        )
        return errors / max(total, 1)
