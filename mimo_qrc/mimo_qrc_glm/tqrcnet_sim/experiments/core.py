from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from tqrcnet_sim.mimo_ofdm.channel import (
    ChannelRealization,
    generate_channel_realization,
    observe_channel_realization,
)
from tqrcnet_sim.mimo_ofdm.config import OfdmConfig
from tqrcnet_sim.mimo_ofdm.frame import generate_frame
from tqrcnet_sim.models.trcnet import TRCNetDetector
from tqrcnet_sim.reservoirs.dtc_qrc import (
    DTCQRCReservoir,
    QRC_INPUT_ENCODINGS,
)
from tqrcnet_sim.reservoirs.esn import ESNReservoir


SUPPORTED_BACKENDS = ("esn", "dtc_qrc")


@dataclass(frozen=True)
class DetectorSettings:
    """T-RCNet 与替换后端共用的检测器超参数。"""

    n_layers: int = 2
    window: int = 4
    esn_states: int = 32
    esn_spectral_radius: float = 0.2
    esn_input_scale: float = 1.0
    # 论文只说明均匀分布，未给出 W_in 与 W 的区间及是否以零为中心。
    esn_input_distribution: str = "symmetric"
    esn_recurrent_distribution: str = "symmetric"
    # 原文未公开递归矩阵的稀疏度；1.0 保持当前稠密基线。
    esn_recurrent_connectivity: float = 1.0
    # 原文公式 (7) 未公开状态激活函数 f 的具体形式。
    esn_activation: str = "tanh"
    # 原文将复数通信信号送入实值 ESN，但未说明是否先做幅度归一化。
    # 归一化统计量只能由每层 TS 输入估计，数据段始终复用该尺度。
    input_normalization: str = "none"
    qrc_qubits: int = 6
    # DTC-QRC 主文理想仿真使用 m=12，并把演化划分为四段测量。
    qrc_cycles: int = 12
    qrc_measurement_count: int = 4
    qrc_g: float = 0.84
    # dense angle 在每个 qubit 上编码两个 TS-PCA 分量。
    qrc_input_encoding: str = "pca_dense_angle"
    qrc_input_scale: float = 1.0
    # 混合替换默认保留 T-RCNet 公式 z=[s,i] 的直接窗口输入。
    qrc_include_input: bool = True
    # 论文公式 (9) 使用无正则 Moore-Penrose 伪逆，且 z 中不含常数偏置。
    ridge: float = 0.0
    readout_bias: bool = False
    forgetting: float = 0.9995
    use_gaw_rls: bool = True
    gaw_alpha: float = 27.0
    gaw_beta: float = 15.0
    # 默认 1 严格对应论文公式；非 1 值只用于审计误差功率单位。
    gaw_error_power_scale: float = 1.0
    # 论文原式为完整误差向量平方和；mean 只用于 P1-2 口径消融。
    gaw_error_power_mode: str = "sum"
    # 论文原式为全输出共享标量权重；per_stream 只审计未公开实现。
    gaw_weight_granularity: str = "global"
    # 论文公式 (12)--(15) 未说明 Ψ^{-1} 初值，因此将假设显式参数化。
    rls_covariance_mode: str = "ts_gram"
    rls_initial_covariance_scale: float = 1.0
    # 前序 RCNet 在 0...Ncp 内均匀选择 5 个响应延迟候选；设为 1 即仅使用 p=0。
    delay_candidate_count: int = 5
    # 原文没有说明 OFDM 符号边界是否清空 reservoir 状态，故显式保留两种假设。
    state_boundary_mode: str = "frame_continuous"
    # 原文未说明 TS 推理末态和窗口历史是否交接给后续数据接收流。
    ts_data_state_handoff: str = "reset"
    # Algorithm 1 明确采用先更新、后检测；另一模式仅用于单符号滞后诊断。
    online_update_timing: str = "update_then_detect"
    # 原文未说明 pilot 短序列结束状态是否传给当前完整符号推理。
    pilot_state_handoff: str = "isolated_stream"
    # 原文未说明相邻 Φ_i 是否共享 pilot-only reservoir 状态和窗口历史。
    pilot_training_stream_mode: str = "reset_each_symbol"
    # 常规 Monte Carlo 默认关闭逐样本追踪，诊断入口会显式启用。
    collect_online_diagnostics: bool = False

    def __post_init__(self):
        if self.n_layers <= 0:
            raise ValueError("T-RCNet 层数必须大于 0")
        if self.window <= 0:
            raise ValueError("滑动窗口长度必须大于 0")
        if self.esn_states <= 0:
            raise ValueError("ESN 状态数必须大于 0")
        if self.esn_input_scale <= 0.0:
            raise ValueError("ESN 输入权重尺度必须大于 0")
        if self.esn_input_distribution.lower() not in (
            "symmetric",
            "positive",
        ):
            raise ValueError(
                "esn_input_distribution 必须为 'symmetric' 或 'positive'"
            )
        if self.esn_recurrent_distribution.lower() not in (
            "symmetric",
            "positive",
        ):
            raise ValueError(
                "esn_recurrent_distribution 必须为 "
                "'symmetric' 或 'positive'"
            )
        if (
            not np.isfinite(self.esn_recurrent_connectivity)
            or self.esn_recurrent_connectivity <= 0.0
            or self.esn_recurrent_connectivity > 1.0
        ):
            raise ValueError(
                "esn_recurrent_connectivity 必须位于 (0, 1]"
            )
        if self.esn_activation.lower() not in (
            "tanh",
            "linear",
            "relu",
            "sigmoid",
        ):
            raise ValueError(
                "esn_activation 必须为 'tanh'、'linear'、"
                "'relu' 或 'sigmoid'"
            )
        if self.input_normalization.lower() not in (
            "none",
            "global_rms",
            "per_channel_rms",
        ):
            raise ValueError(
                "input_normalization 必须为 'none'、"
                "'global_rms' 或 'per_channel_rms'"
            )
        if self.ridge < 0.0:
            raise ValueError("ridge 不能为负数")
        if (
            not np.isfinite(self.gaw_error_power_scale)
            or self.gaw_error_power_scale <= 0.0
        ):
            raise ValueError("GAW 误差功率尺度必须为有限正数")
        if self.gaw_error_power_mode.lower() not in ("sum", "mean"):
            raise ValueError(
                "gaw_error_power_mode 必须为 'sum' 或 'mean'"
            )
        if self.gaw_weight_granularity.lower() not in (
            "global",
            "per_stream",
        ):
            raise ValueError(
                "gaw_weight_granularity 必须为 'global' 或 'per_stream'"
            )
        if (
            self.qrc_qubits <= 0
            or self.qrc_cycles <= 0
            or self.qrc_measurement_count <= 0
        ):
            raise ValueError(
                "QRC qubit 数、Floquet 周期数和测量分段数必须大于 0"
            )
        if self.qrc_cycles % self.qrc_measurement_count:
            raise ValueError("QRC Floquet 周期数必须能被测量分段数整除")
        if not np.isfinite(self.qrc_g) or not 0.0 <= self.qrc_g <= 1.0:
            raise ValueError("qrc_g 必须位于 [0, 1]")
        if self.qrc_input_encoding.lower() not in QRC_INPUT_ENCODINGS:
            raise ValueError(
                "qrc_input_encoding 必须为 'pca_angle' 或 "
                "'pca_dense_angle'"
            )
        if (
            not np.isfinite(self.qrc_input_scale)
            or self.qrc_input_scale <= 0.0
        ):
            raise ValueError("qrc_input_scale 必须为有限正数")
        if self.delay_candidate_count <= 0:
            raise ValueError("响应延迟候选数必须大于 0")
        if self.state_boundary_mode.lower() not in (
            "frame_continuous",
            "symbol_reset",
        ):
            raise ValueError(
                "state_boundary_mode 必须为 "
                "'frame_continuous' 或 'symbol_reset'"
            )
        if self.ts_data_state_handoff.lower() not in (
            "reset",
            "inherit_ts",
        ):
            raise ValueError(
                "ts_data_state_handoff 必须为 'reset' 或 'inherit_ts'"
            )
        if (
            self.ts_data_state_handoff.lower() == "inherit_ts"
            and self.state_boundary_mode.lower() != "frame_continuous"
        ):
            raise ValueError(
                "inherit_ts 仅适用于 frame_continuous 状态边界"
            )
        if self.online_update_timing.lower() not in (
            "update_then_detect",
            "detect_then_update",
        ):
            raise ValueError(
                "online_update_timing 必须为 "
                "'update_then_detect' 或 'detect_then_update'"
            )
        if self.pilot_state_handoff.lower() not in (
            "isolated_stream",
            "inherit_pilot",
        ):
            raise ValueError(
                "pilot_state_handoff 必须为 "
                "'isolated_stream' 或 'inherit_pilot'"
            )
        if self.pilot_training_stream_mode.lower() not in (
            "reset_each_symbol",
            "continuous",
        ):
            raise ValueError(
                "pilot_training_stream_mode 必须为 "
                "'reset_each_symbol' 或 'continuous'"
            )
        if self.rls_covariance_mode.lower() not in (
            "ts_gram",
            "discounted_ts_gram",
            "identity",
        ):
            raise ValueError(
                "rls_covariance_mode 必须为 'ts_gram'、"
                "'discounted_ts_gram' 或 'identity'"
            )
        if self.rls_initial_covariance_scale <= 0.0:
            raise ValueError("RLS 初始逆相关矩阵尺度必须大于 0")


@dataclass(frozen=True)
class SharedObservation:
    """一个 Eb/N0 点上供所有检测后端共享的观测。"""

    cfg: OfdmConfig
    frame: dict[str, Any]
    received_time: np.ndarray
    channel_realization: ChannelRealization
    realization_seed: int


@dataclass(frozen=True)
class BackendEvaluation:
    """一个检测后端在同一观测上的两种 Algorithm 1 结果。"""

    kind: str
    ts_only_errors: int
    online_errors: int
    bit_count: int

    @property
    def ts_only_ber(self) -> float:
        return self.ts_only_errors / max(self.bit_count, 1)

    @property
    def online_ber(self) -> float:
        return self.online_errors / max(self.bit_count, 1)


def build_trial_observations(
    base_cfg: OfdmConfig,
    ebn0_values: Iterable[float],
    realization_seed: int,
) -> list[SharedObservation]:
    """为一个 trial 构造多个 Eb/N0 下的配对观测。

    一个 trial 只生成一次发送帧、时变信道和单位噪声。所有 Eb/N0 点
    复用这些对象，仅改变单位噪声的缩放系数。
    """
    if realization_seed < 0:
        raise ValueError("realization_seed 不能为负数")

    values = tuple(float(value) for value in ebn0_values)
    if not values:
        raise ValueError("至少需要一个 Eb/N0 取值")
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Eb/N0 必须是有限数值")

    trial_cfg = replace(base_cfg, seed=realization_seed)
    frame = generate_frame(trial_cfg)
    channel_realization = generate_channel_realization(frame["x_time"], trial_cfg)

    observations = []
    for ebn0_db in values:
        cfg = replace(trial_cfg, ebn0_db=ebn0_db)
        received_time = observe_channel_realization(
            channel_realization,
            frame["x_time"],
            cfg,
            ebn0_db,
        )
        observations.append(
            SharedObservation(
                cfg=cfg,
                frame=frame,
                received_time=received_time,
                channel_realization=channel_realization,
                realization_seed=realization_seed,
            )
        )
    return observations


def make_reservoirs(
    kind: str,
    cfg: OfdmConfig,
    settings: DetectorSettings,
    model_seed: int,
):
    """创建各层 reservoir；模型种子与通信 realization 种子相互独立。"""
    if kind not in SUPPORTED_BACKENDS:
        raise ValueError(f"未知 reservoir 类型: {kind}")
    if model_seed < 0:
        raise ValueError("model_seed 不能为负数")

    # 第 1 层输入 Nr 路接收信号，后续层输入上一层的 Nt 路估计信号。
    input_channels = (cfg.nr,) + (cfg.nt,) * (settings.n_layers - 1)
    reservoirs = []
    for layer_index, channels in enumerate(input_channels):
        input_dim = 2 * channels * settings.window
        seed = model_seed + layer_index
        if kind == "esn":
            # 论文实验设置：每层 recurrent neuron 数 Ns=32，谱半径 0.2。
            reservoir = ESNReservoir(
                input_dim=input_dim,
                n_state=settings.esn_states,
                spectral_radius=settings.esn_spectral_radius,
                input_scale=settings.esn_input_scale,
                input_distribution=settings.esn_input_distribution,
                recurrent_distribution=(
                    settings.esn_recurrent_distribution
                ),
                recurrent_connectivity=(
                    settings.esn_recurrent_connectivity
                ),
                activation=settings.esn_activation,
                seed=seed,
            )
        else:
            reservoir = DTCQRCReservoir(
                input_dim=input_dim,
                n_qubits=settings.qrc_qubits,
                cycles=settings.qrc_cycles,
                measurement_count=settings.qrc_measurement_count,
                g=settings.qrc_g,
                seed=seed,
                input_encoding=settings.qrc_input_encoding,
                input_scale=settings.qrc_input_scale,
                include_input=settings.qrc_include_input,
            )
        reservoirs.append(reservoir)
    return reservoirs


def make_detector(
    kind: str,
    cfg: OfdmConfig,
    settings: DetectorSettings,
    model_seed: int,
) -> TRCNetDetector:
    """创建一个尚未训练的双层 T-RCNet 检测器。"""
    if settings.delay_candidate_count == 1:
        delay_candidates = (0,)
    else:
        # RCNet 实验设置 P=5，即在闭区间 [0, Ncp] 均匀评估 5 个候选。
        delay_candidates = tuple(
            sorted(
                set(
                    np.rint(
                        np.linspace(
                            0,
                            cfg.ncp,
                            settings.delay_candidate_count,
                        )
                    ).astype(int)
                )
            )
        )
    return TRCNetDetector(
        reservoir=make_reservoirs(kind, cfg, settings, model_seed),
        cfg=cfg,
        window=settings.window,
        ridge=settings.ridge,
        forgetting=settings.forgetting,
        use_gaw_rls=settings.use_gaw_rls,
        gaw_alpha=settings.gaw_alpha,
        gaw_beta=settings.gaw_beta,
        gaw_error_power_scale=settings.gaw_error_power_scale,
        gaw_error_power_mode=settings.gaw_error_power_mode,
        gaw_weight_granularity=settings.gaw_weight_granularity,
        readout_bias=settings.readout_bias,
        delay_candidates=delay_candidates,
        state_boundary_mode=settings.state_boundary_mode,
        ts_data_state_handoff=settings.ts_data_state_handoff,
        online_update_timing=settings.online_update_timing,
        pilot_state_handoff=settings.pilot_state_handoff,
        pilot_training_stream_mode=settings.pilot_training_stream_mode,
        input_normalization=settings.input_normalization,
        collect_online_diagnostics=settings.collect_online_diagnostics,
        rls_covariance_mode=settings.rls_covariance_mode,
        rls_initial_covariance_scale=(
            settings.rls_initial_covariance_scale
        ),
    )


def evaluate_backend(
    kind: str,
    observation: SharedObservation,
    settings: DetectorSettings | None = None,
    model_seed: int = 7,
) -> BackendEvaluation:
    """在固定观测上评估 TS-only 与 pilot-GAW-RLS 两种检测流程。"""
    if settings is None:
        settings = DetectorSettings()

    cfg = observation.cfg
    frame = observation.frame
    y_time = observation.received_time
    y_ts = y_time[: cfg.n_ts]
    x_ts = frame["x_time"][: cfg.n_ts]
    y_data = y_time[cfg.n_ts :]
    x_freq_data = frame["x_freq"][cfg.n_ts :]

    detector = make_detector(kind, cfg, settings, model_seed)
    detector.fit_ts(y_ts, x_ts)
    ts_errors, ts_bits = detector.evaluate_error_counts(
        y_data,
        frame["data_bits"],
        frame["data_mask"],
    )

    # 在线流程必须使用全新的、同权重初始化的检测器，避免 TS-only 推理状态
    # 污染 Algorithm 1 的 reservoir 状态和 GAW-RLS 更新过程。
    detector_online = make_detector(kind, cfg, settings, model_seed)
    detector_online.fit_ts(y_ts, x_ts)
    online_errors, online_bits = detector_online.evaluate_error_counts_online(
        y_data,
        x_freq_data,
        frame["data_bits"],
        frame["data_mask"],
    )
    if ts_bits != online_bits:
        raise RuntimeError("TS-only 与在线检测统计的总比特数不一致")

    return BackendEvaluation(
        kind=kind,
        ts_only_errors=ts_errors,
        online_errors=online_errors,
        bit_count=ts_bits,
    )
