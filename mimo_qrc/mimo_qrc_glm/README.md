# 用 GLM 与 OpenEvolve 搜索量子储备池 MIMO 检测器

在复用的 4×4 MIMO-OFDM 仿真中，寻找一个量子储备池检测器，使 **0、5、10、15、20、25、30 dB 七个 Eb/N0 点的 payload BER 均严格低于 LMMSE**。参考为原项目的联合 TS-LMMSE 加条件导频跟踪，使用估计 CSI。QRC 比特数、结构、编码、观测、读出层及其训练方式均可修改。

OpenEvolve 的三个输入为 `initial_program.py`（既有 DTC-QRC 起点）、`evaluator.py`（固定仿真和评分）和 `config.yaml`（任务提示、GLM 接口和搜索设置）。`tqrcnet_sim/` 与 `ofdm.json` 是仿真依赖。包内无需 MATLAB，FIR 系数文件已经包含。

## 1. 准备环境并检查评估

在解压后的目录运行。推荐 Python 3.12。下面以 Linux 为例。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONUTF8=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python evaluator.py initial_program.py --frames 1 --workers 1
```

最后一条命令在同一仿真帧上运行七点的初始 QRC 和 LMMSE，打印两个 BER、误码数和耗时，不连接 GLM。初始 QRC 可以比 LMMSE 差。先确认有完整七点结果并记录耗时，再开始搜索。

## 2. 填写超节点的 GLM 配置

GLM 服务须提供兼容 Chat Completions 的 HTTP 接口。修改 `config.yaml` 的两个值。接口格式可参考 [智谱/Z.AI 开发文档](https://docs.z.ai/guides/develop/openai/python)。

- `llm.api_base` 填接口基址。例如完整接口为 `http://node:8000/v1/chat/completions` 时，填 `http://node:8000/v1`。
- `llm.models[0].name` 填部署实际暴露的模型 ID，替换模板中的占位名。

`llm.provider: openai` 选择 OpenEvolve 的兼容协议适配器，实际请求发往上述 GLM 地址。鉴权服务从环境变量读取密钥。

```bash
read -rsp 'GLM API key: ' GLM_API_KEY; echo
export GLM_API_KEY
```

若部署方确认服务不鉴权，设置 `export GLM_API_KEY=local-no-key` 即可满足客户端的非空要求。真实密钥请在运行端注入。

## 3. 搜索与独立复核

首次接入可以用两次改写检查实际返回的是完整 Python 程序、生成的候选能运行、评分能回传。

```bash
openevolve-run initial_program.py evaluator.py --config config.yaml \
  --iterations 2 --output runs/glm-smoke
```

首轮探索默认 **20 次改写**，每份候选使用 **10 个独立开发帧**评估全部七点。确认耗时和部署配额后启动。

```bash
openevolve-run initial_program.py evaluator.py --config config.yaml --output runs/glm-20
```

20 次是起步预算。若候选经常超时、返回不完整或持续重复，先查看失败日志和评估反馈。若设计在变化且接近目标，可把总预算分阶段增至 50–100 次。续跑可用 `--checkpoint` 指向输出目录的检查点，`--iterations` 表示续跑追加的迭代数。沿用同一个配置和开发帧集合，以保持评分可比较。

搜索结束后，封存一个拟选候选并检查源码，确认量子储备池产生的特征实际参与了检测。开发评分满足 `target_met = 1` 且源码符合 QRC 要求后，在独立的 **30 个留出帧**上复核。留出结果不要送回同一轮搜索。

```bash
python evaluator.py runs/glm-20/best/best_program.py --split heldout \
  --frames 30 --workers 4 --output runs/glm-20/heldout.json
```

若需要测量该候选的开发集性能，可以用同一命令改为 `--split development --frames 10`。如设计使用单个加速设备，可把 `config.yaml` 中的 `evaluator.evaluator_args.workers` 和复核命令的 `--workers` 都设为 1。`frame_timeout` 控制一个帧的七点预测可用时间，`evaluator.timeout` 应覆盖整次候选评估和 GLM 请求。例如用 1 worker、10 帧及每帧 240 秒的上限时，可把 `evaluator.timeout` 从 1800 调至 3300 秒。OpenEvolve 工作进程在运行本地仿真的主机上执行，GLM 部署在超节点只决定模型推理的位置。

## 评估口径

- 固定 FFT64/CP16、8 个训练符号和 92 个数据符号、16-QAM、20 MHz、EPA 20 Hz 与 MATLAB FIR 配置。每帧每个点统计 70,656 个 payload bit。QRC 和 LMMSE 使用相同接收波形、已知训练符号、导频和噪声方差信息。七个点复用同一帧和噪声样本，仅调整噪声强度。
- 每个点按总误码数比较。分母一致，因此候选误码更少才算该点严格胜出。七点都胜出时 `target_met=1`，并按七点等权的平均 QRC BER 排名。其余候选仍收到平均 BER 和超过 LMMSE 的差距作为改写反馈。如果某一点 LMMSE 样本误码数恰为零，则这一样本上无法取得严格胜出，平局也不会通过。
- 开发集和留出集的通信种子不重叠。留出报告保留各帧计数与候选哈希。七点通过是对该测试样本的判断，如要做研究结论，可基于逐帧配对结果分析不确定性。评估器自动检查接口和 BER，QRC 的贡献要通过源代码审查确认。对重要结果可额外用关闭量子特征的消融检查其效果。
- 每个帧和 SNR 点分别启动候选进程，接收机接口只传入该点观测和合法训练信息。该进程隔离不构成安全沙箱。运行模型生成代码的账户应与超节点的服务、凭据和共享研究数据分开。

来源为之前项目中已核验的 `tqrcnet_sim` 仿真快照和原十帧基线物理配置。LMMSE 调用 `tqrcnet_sim/models/lmmse.py` 的 `detect_estimated_csi`，启用导频跟踪。起始 QRC 复用此前 4 比特、4 周期的双层检测器，其参数可以随候选一起替换。

交付前的本地核验中，LMMSE 在历史帧 61000 上的七项误码计数与原记录一致。起始 QRC 在一条新开发帧上完成全七点评估，两个检测器的平均 BER 分别为 0.238289 和 0.045856。该检查用于确认接口和基线，实际 GLM 搜索由部署端执行。
