from dataclasses import dataclass


# 20 MHz IEEE 802.11a/g OFDM 的 64 点 FFT 子载波映射。
# NumPy FFT 下标 0 对应 DC，负频率 -k 对应下标 Nsc-k。
WIFI64_PILOT_INDICES = (43, 57, 7, 21)  # 物理频率顺序：-21, -7, +7, +21
WIFI64_OCCUPIED_INDICES = tuple(range(1, 27)) + tuple(range(38, 64))
WIFI64_DATA_INDICES = tuple(
    index
    for index in WIFI64_OCCUPIED_INDICES
    if index not in WIFI64_PILOT_INDICES
)


@dataclass
class OfdmConfig:
    """MIMO-OFDM 仿真参数。"""

    profile_name: str = "quick_debug"
    nt: int = 2
    nr: int = 2
    nsc: int = 32
    ncp: int = 8
    n_ts: int = 4
    n_data: int = 12
    pilot_indices: tuple[int, ...] = (5, 13, 21, 29)
    # None 表示除 pilot 外的全部子载波均承载数据，供快速调试使用。
    data_subcarrier_indices: tuple[int, ...] | None = None
    bits_per_qam_symbol: int = 4
    ebn0_db: float = 15.0
    # random_qam：当前可复现的已知 16-QAM TS；
    # wifi_ltf：legacy L-LTF 的 52 子载波 BPSK 序列，加 4 流 HT-LTF
    # 正交映射并重复到 NTS=8。论文未公开其 4×4、8-symbol 具体矩阵。
    training_sequence_mode: str = "random_qam"
    # paper_change：按论文公式 (23) 在相邻 OFDM 符号间递推 pilot；
    # fixed：论文 Fig. 6 的 T-RCNet-Fix-Pilot 对照，pilot 保持不变。
    pilot_pattern_mode: str = "paper_change"
    # 论文公式 (23) 只规定相邻符号的递推关系，没有公开初始 MIMO
    # pilot 矩阵。dft 是当前复现基线；wifi_bpsk 将 IEEE HT-LTF 的
    # 4×4 正交 BPSK 矩阵复用为敏感性假设，不代表论文明确采用该矩阵。
    initial_mimo_pilot_mode: str = "dft"
    # unitary：F^H F=I 的酉变换；matlab_native：MATLAB/NumPy 原生
    # ifft 含 1/N、fft 不缩放。原文只给出矩阵推导，未公开代码口径。
    transform_normalization: str = "unitary"

    # 默认使用论文实验中的 3GPP EPA 时变多径信道。
    channel_model: str = "epa"
    # 快速配置将论文的 Nsc/Ncp=64/16 等比例缩为 32/8，因此采样率也
    # 缩为 10 MHz，以保持 3.2 us 有效符号和 0.8 us CP 的物理时长。
    sample_rate_hz: float = 10_000_000.0
    doppler_hz: float = 20.0
    # symbol_block：论文明确采用的“符号内不变、符号间变化”假设；
    # sample_continuous：P2-1 对照，模拟 lteFadingChannel 的逐样本路径增益。
    channel_time_evolution_mode: str = "symbol_block"
    # None 表示按真实 OFDM 符号时长演化。论文没有说明 MATLAB 信道
    # 在 OFDM 符号轴上的采样间隔，因此允许显式覆盖以做敏感性实验。
    channel_evolution_interval_s: float | None = None
    jakes_sinusoids: int = 32
    # linear 保留历史一阶插值；matlab_fir 使用 R2018b 实测 FIR，补偿已知
    # 的实现延迟。当前校准只覆盖 EPA、20 MHz，不凭空外推到其他采样率。
    fractional_delay_mode: str = "linear"
    # total：所有 Nt 路叠加后的平均接收功率保持不变；
    # per_link：每条 MIMO 链路的 EPA 总路径功率归一为 1，不再除以 Nt；
    # 相当于 lteFadingChannel 的 NormalizeTxAnts="Off"，不是其默认值。
    # total 对应该字段为 "On"；论文没有公开这一字段的实际取值。
    mimo_power_normalization: str = "total"
    # MATLAB lteFadingChannel 将 3GPP "Low" 定义为天线间无相关；
    # 因此独立 Tx-Rx 链路就是论文复现采用的 low 空间相关配置。
    spatial_correlation: str = "low"

    # 仅供旧版 simplified 信道消融实验使用。
    channel_taps: int = 4
    channel_drift: float = 0.01
    seed: int = 7

    def __post_init__(self):
        if self.nt <= 0 or self.nr <= 0:
            raise ValueError("Nt 和 Nr 必须大于 0")
        if self.nsc <= 0 or self.ncp < 0:
            raise ValueError("Nsc 必须大于 0，Ncp 不能为负数")
        if self.n_ts <= 0 or self.n_data <= 0:
            raise ValueError("TS 和数据 OFDM 符号数必须大于 0")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz 必须大于 0")
        if self.doppler_hz < 0.0:
            raise ValueError("doppler_hz 不能为负数")
        if self.fractional_delay_mode.lower() not in ("linear", "matlab_fir"):
            raise ValueError("fractional_delay_mode 必须为 'linear' 或 'matlab_fir'")
        if self.fractional_delay_mode.lower() == "matlab_fir" and (
            self.channel_model.lower() != "epa" or abs(self.sample_rate_hz - 20e6) > 1e-6
        ):
            raise ValueError("matlab_fir 校准仅支持 EPA、20 MHz")
        if self.channel_time_evolution_mode.lower() not in (
            "symbol_block",
            "sample_continuous",
        ):
            raise ValueError(
                "channel_time_evolution_mode 必须为 "
                "'symbol_block' 或 'sample_continuous'"
            )
        if (
            self.channel_evolution_interval_s is not None
            and self.channel_evolution_interval_s <= 0.0
        ):
            raise ValueError("信道演化时间间隔必须大于 0")
        if self.mimo_power_normalization.lower() not in ("total", "per_link"):
            raise ValueError(
                "mimo_power_normalization 必须为 'total' 或 'per_link'"
            )
        if self.bits_per_qam_symbol != 4:
            raise ValueError("当前调制器仅支持每符号 4 bit 的 16-QAM")
        if self.training_sequence_mode.lower() not in (
            "random_qam",
            "wifi_ltf",
        ):
            raise ValueError(
                "training_sequence_mode 必须为 'random_qam' 或 'wifi_ltf'"
            )
        if self.pilot_pattern_mode.lower() not in (
            "paper_change",
            "fixed",
        ):
            raise ValueError(
                "pilot_pattern_mode 必须为 'paper_change' 或 'fixed'"
            )
        if self.initial_mimo_pilot_mode.lower() not in (
            "dft",
            "wifi_bpsk",
        ):
            raise ValueError(
                "initial_mimo_pilot_mode 必须为 'dft' 或 'wifi_bpsk'"
            )
        if self.transform_normalization.lower() not in (
            "unitary",
            "matlab_native",
        ):
            raise ValueError(
                "transform_normalization 必须为 "
                "'unitary' 或 'matlab_native'"
            )

        pilot_set = set(self.pilot_indices)
        if len(pilot_set) != len(self.pilot_indices):
            raise ValueError("pilot_indices 不能包含重复下标")
        if any(index < 0 or index >= self.nsc for index in pilot_set):
            raise ValueError("pilot 子载波下标超出 FFT 范围")
        if (
            self.initial_mimo_pilot_mode.lower() == "wifi_bpsk"
            and (len(pilot_set) != 4 or self.nt > 4)
        ):
            raise ValueError(
                "wifi_bpsk 初始 MIMO pilot 要求 4 个 pilot 且 Nt<=4"
            )

        if self.data_subcarrier_indices is not None:
            data_set = set(self.data_subcarrier_indices)
            if len(data_set) != len(self.data_subcarrier_indices):
                raise ValueError("data_subcarrier_indices 不能包含重复下标")
            if any(index < 0 or index >= self.nsc for index in data_set):
                raise ValueError("数据子载波下标超出 FFT 范围")
            if pilot_set & data_set:
                raise ValueError("pilot 与数据子载波不能重叠")
        else:
            data_set = set(range(self.nsc)) - pilot_set

        if self.training_sequence_mode.lower() == "wifi_ltf":
            if self.nsc != 64 or self.nt > 4:
                raise ValueError(
                    "wifi_ltf 仅支持 64 点 FFT 和不超过 4 条发送流"
                )
            occupied = data_set | pilot_set
            if occupied != set(WIFI64_OCCUPIED_INDICES):
                raise ValueError(
                    "wifi_ltf 要求 legacy Wi-Fi 的 52 个占用子载波"
                )
            n_ltf = 1 if self.nt == 1 else 2 if self.nt == 2 else 4
            if self.n_ts % n_ltf != 0:
                raise ValueError(
                    "NTS 必须是对应 Wi-Fi LTF 正交映射列数的整数倍"
                )

        if self.spatial_correlation.lower() not in ("low", "independent"):
            raise NotImplementedError(
                "当前论文复现阶段仅实现 low/independent 空间相关配置"
            )

    @classmethod
    def quick_debug(cls, **overrides):
        """创建计算量较小、保持论文时间尺度的调试配置。"""
        params = {
            "profile_name": "quick_debug",
            "nt": 2,
            "nr": 2,
            "nsc": 32,
            "ncp": 8,
            "n_ts": 4,
            "n_data": 12,
            "pilot_indices": (5, 13, 21, 29),
            "data_subcarrier_indices": None,
            "sample_rate_hz": 10_000_000.0,
        }
        params.update(overrides)
        return cls(**params)

    @classmethod
    def paper_baseline(cls, **overrides):
        """创建论文 Algorithm 1 的完整线性区基线配置。

        论文明确给出 Nsc=64、Ncp=16、NTS=8、N=100、4 个 pilot、
        48 个数据子载波和 12 个 null 子载波。
        """
        params = {
            "profile_name": "paper_baseline",
            "nt": 4,
            "nr": 4,
            "nsc": 64,
            "ncp": 16,
            "n_ts": 8,
            "n_data": 92,
            "pilot_indices": WIFI64_PILOT_INDICES,
            "data_subcarrier_indices": WIFI64_DATA_INDICES,
            "sample_rate_hz": 20_000_000.0,
            "doppler_hz": 20.0,
            "mimo_power_normalization": "per_link",
            "spatial_correlation": "low",
        }
        params.update(overrides)
        return cls(**params)

    @property
    def ntd(self) -> int:
        """一个时域 OFDM 符号的长度，包含 CP。"""
        return self.nsc + self.ncp

    @property
    def n_total(self) -> int:
        """一个帧中的 OFDM 符号总数。"""
        return self.n_ts + self.n_data

    @property
    def symbol_duration_s(self) -> float:
        """一个含 CP 的 OFDM 符号持续时间。"""
        return self.ntd / self.sample_rate_hz

    @property
    def resolved_channel_evolution_interval_s(self) -> float:
        """返回 Jakes 信道在相邻 OFDM 符号之间采用的时间间隔。"""
        if self.channel_evolution_interval_s is None:
            return self.symbol_duration_s
        return self.channel_evolution_interval_s

    @property
    def resolved_channel_sample_interval_s(self) -> float:
        """返回逐样本信道轨迹的时间步长。

        默认值为 ``1/sample_rate_hz``。当符号演化间隔被显式覆盖时，
        同步按 ``interval/Ntd`` 缩放，保证逐样本轨迹在符号首样本处与
        `symbol_block` 使用完全相同的时间点。
        """
        return self.resolved_channel_evolution_interval_s / self.ntd

    @property
    def cp_duration_s(self) -> float:
        """循环前缀持续时间。"""
        return self.ncp / self.sample_rate_hz

    @property
    def resolved_data_indices(self) -> tuple[int, ...]:
        """返回实际使用的数据子载波下标。"""
        if self.data_subcarrier_indices is None:
            pilot_set = set(self.pilot_indices)
            return tuple(index for index in range(self.nsc) if index not in pilot_set)
        return tuple(sorted(self.data_subcarrier_indices))

    @property
    def null_indices(self) -> tuple[int, ...]:
        """返回既不承载数据也不承载 pilot 的 null 子载波下标。"""
        occupied = set(self.pilot_indices) | set(self.resolved_data_indices)
        return tuple(index for index in range(self.nsc) if index not in occupied)

    @property
    def n_data_subcarriers(self) -> int:
        """每个数据 OFDM 符号中的数据子载波数量。"""
        return len(self.resolved_data_indices)
