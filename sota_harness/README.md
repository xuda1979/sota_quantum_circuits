# sota_harness — Classical Quantum Algorithms, Optimized and Audited

Goal: take **classical quantum algorithms** (reversible arithmetic — the
in-place adder `|a>|b> -> |a+b mod 2^n>|b>`, n=3 through n=6, the kernel
inside Shor's algorithm), run **our agent** (evolutionary loop with an LLM proposer,
template proposer, and mutation pipeline) on them, and see whether it matches
or beats known constructions. Document the outcome honestly.

All numbers below are harness-verified (deterministic evaluator over the full
truth table; `exit 0`). Raw logs: `results_adder_m3/`, `results_adder_m4/`,
`results_adder_m5/`, `baselines_adder/results_adder_baselines.json`, and the
matched ablation logs in `results_ablation/`.

**Update 2026-11-24 (§3)**: we extended the agent's *candidate-generation
machinery* with a general structural mutation operator
(`RedundantPairElimRewriter`) plus canonical-source dedup, and ran a matched
ablation (8 arms at n=3/4, extended to n=5/6 — 14 arms total). The
previously out-of-reach 2n−2 circuits (TQ 28/54/88/130 at n=3/4/5/6) are
now **discovered by the loop in one mutation step** from the CDKM seed —
at every size, in both no-LLM and LLM settings — while matched
base-mutator controls stay at parity (42/72/110/156). The LLM alone never
finds the target in any arm.

**Update 2026-11-25 (§3b)**: operator-transfer test on a **held-out task
family** — the in-place subtractor, the exact gate-mirror of the adder.
The unchanged operator hits a clean, measurable boundary on the mirror
(returns None offline; matched arms structural ≡ base). A minimal
order-agnostic generalization (no task semantics, adder regression
byte-clean) restores full transfer: the mirrored 2n−2 subtractor is
loop-discovered in one step (TQ 28/54), all controls stay at parity.
10-arm ablation: `results_ablation/subtractor_ablation_summary.json`.

## 主要成果（中文总结）

**任务**：经典量子算法（可逆算术）——Shor 算法内核的就地加法器
`|a>|b> → |a+b mod 2^n>|b>`（n=3、n=4）。用我们的智能体（LLM 演化搜索）
优化，并与已知构造对比、如实记录。

**优化结果（好结果，全部经确定性求值器验证，exit 0，2026-11-24 复验）**：
跳过 CDKM 顶部 MAJ/UMA 级（其算出的顶部进位本就被丢弃），最高和位直接
从进位链读出；单个进位 ancilla 保持干净（回归 |0>）。

| 任务 | 构造 | T（Toffoli） | Q（量子比特） | TQ=T×Q | 相对 CDKM |
|---|---|---|---|---|---|
| n=3 | CDKM 基线 | 6 | 7 | 42 | — |
| n=3 | **本工作** `t4_adder_n3.py` | **4** | 7 | **28** | T/TQ −33% |
| n=4 | CDKM 基线 | 8 | 9 | 72 | — |
| n=4 | **本工作** `t6_adder_n4.py` | **6** | 9 | **54** | T/TQ −25% |

**最优性证书（n=3）**：在 span 模型下对全部 651 个首乘积向量穷举证明
2 个 Toffoli 乘积**不可能**实现 n=3 加法器，并给出 3 乘积见证
`{a0·b0, a1·b1, a0·b0·(a1⊕b1)}` → **乘法复杂度恰为 3**。本工作 T=4 距
span 最优仅差 1。秩下界：n=3/4/5 时 T≥2/3/4。

**智能体诚实审计（由原始 JSONL 重新统计）**：智能体（LLM+模板）可复现
CDKM 并达到持平（最好 TQ=42/72；11/12、29/30 有效），但**从未超越**；
无模板纯 LLM 两次运行 0/26 有效（全部为脏 ancilla 失败）。本次优化来自
证书引导的构造，而非智能体本身。

**2026-11-24 更新（结构突变体使优化可被智能体发现）**：为智能体的
候选生成机制新增通用结构突变算子 `RedundantPairElimRewriter`（纹波进位
链顶部 MAJ/UMA 对消去：其算出的顶部进位本就被丢弃，最高和位直接从进位
链读出，省 2 个 Toffoli），并加入规范源码去重。匹配消融（30 迭代，
n=3–6，LLM/无LLM，各臂同预算同种子模板，共 14 臂）结果：

| 设置 | n | 基座突变体 best TQ | 结构突变体 best TQ | 结构臂首次命中 |
|---|---|---|---|---|
| 无 LLM | 3 | 42 | **28** | 第 1 步变异（p0002） |
| 无 LLM | 4 | 72 | **54** | 第 1 步变异（p0002） |
| 无 LLM | 5 | 110 | **88** | 第 1 步变异（p0002） |
| 无 LLM | 6 | 156 | **130** | 第 1 步变异（p0002） |
| LLM 开启 | 3 | 42 | **28** | 第 3 轮（cdkm 家族重写产物 p0004） |
| LLM 开启 | 4 | 72 | **54** | 第 2 轮（cdkm 家族重写产物 p0003） |
| LLM 开启 | 5 | 110 | **88** | 第 2 轮（cdkm 家族重写产物 p0003） |

结论：先前不可达的 2n−2 构造（TQ 28/54/88/130）如今由演化循环从 CDKM
种子**一步变异即可发现**，且该算子完全与规模无关（n=3 到 n=6 同一正则
模式一次命中）；匹配的基座对照组全部停在持平（42/72/110/156）。所有臂中
LLM 自身从未独立发现目标——LLM 家族首次命中 28/54/88 分别在第 6/9/5 轮，
晚于重写器发现进入档案之后（档案反馈回路放大了发现）。详见英文 §3a。

**2026-11-25 更新（算子迁移：门镜像减法器，留出任务族）**：§3a 的重写
算子是在加法器 MAJ-优先门序上推导的。为检验它是否只过拟合了这一门序，
我们构建了加法器的**门镜像**——就地减法器 `|a>|b> → |a−b mod 2^n>|b>`
（CDKM 加法器逐门取逆；所有门自逆）。同机制、镜像种子、镜像 2n−2 真值
（n=3/4 TQ 28/54，求值验证；镜像基线 42/72）。

- **阶段 A（边界）**：未改动算子（dcee826）对镜像模板离线返回 None
  （寄存器提取要求前向辅助函数体内含双下标 `ctx.cx(b[i], a[i])`，而
  镜像的前向级是 UMA，其 cx 只触及进位线）。匹配 30 迭代无 LLM 臂
  （n=3/4）证实：结构臂 ≡ 基座臂（1 次求值、29 次去重跳过、持平
  42/72，四臂完全一致）。
- **阶段 B（最小泛化）**：加入顺序无关回退（21 行，不含任何任务语
  义）——前向体缺少双下标 cx 时，寄存器对改从后向辅助函数体提取。
  加法器回归字节级无损：n=3–6 离线重写输出逐字节一致，实机加法器
  n=3 结构臂仍一步命中 TQ 28。
- **迁移结果（匹配臂，30 迭代，同种子同预算）**：泛化后无 LLM 结构臂
  n=3/4 于**第 1 步变异**命中镜像 2n−2 减法器（TQ 28/54，p0002，与
  离线拼接逐字节一致）；LLM 结构臂第 2 轮命中（p0003），LLM 随后于
  第 23 轮在自身家族复现 TQ 28（档案反馈放大，同加法器模式）；所有
  对照（基座突变体、纯 LLM 28+10 次求值 0 有效、LLM 基座臂 16 次求
  值从未命中）全部停在持平。聚合：
  `results_ablation/subtractor_ablation_summary.json`（10 臂）。

**如实声明**：算子设计时仍已知 2n−2 构造，减法器又是该构造的门镜像，
故**这不是发明，也不声称新门数**（两族皆为已知量级）。更窄而更硬的
结论是：**在一个任务族门序上推导的结构算子，在留出的镜像任务族上有
清晰可测的泛化边界；一个不触及任务语义的最小顺序无关修复即恢复完全
迁移——已知更优构造在匹配预算下一步即可被循环发现，而全部对照均
不可达。** 复现阶段 A 需检出 dcee826（泛化前重写器）。详见英文 §3b。

**如实声明（证明 vs 声明）**：2n−2 Toffoli 就地加法器为已知构造
（Takahashi–Kunihiro 量级，本变体保留 1 个进位 ancilla），**不声称门数
新颖，也不声称智能体超越了基线**。本研究的贡献是：经核验的构造与
精确证书 + 对 LLM 智能体真实能力的匹配对照审计。复现命令见文末。

## 1. Baseline: CDKM ripple-carry (Cuccaro et al. 2004)

| n | T (Toffoli) | Q (qubits) | TQ = T*Q |
|---|---|---|---|
| 3 | 6 | 7 | 42 |
| 4 | 8 | 9 | 72 |

## 2. Agent results (honest audit)

The agent (LLM = deepseek_v4 proxy; matched controls) **reaches CDKM parity
but never beats it**:

| Run | Iters | Valid | Best TQ | Note |
|---|---|---|---|---|
| m4 supervised n=3 (llm+cdkm) | 12 | 11 | 42 (parity) | llm family 5/6 valid |
| m4 supervised n=4 (llm+cdkm) | 12 | 11 | 72 (parity) | llm family 5/6 valid |
| m5 mixed 30-iter n=3 | 30 | 29 | 42 (parity) | llm 14/15, cdkm 15/15 |
| m4 mutation-only control n=3 | 12 | 12 | 42 (parity) | no LLM |
| m3 pure-LLM (no template) n=3 | 6 | 0 | — | all invalid[DIRTY_ANCILLA] |
| m5 pure-LLM (no template) n=3 | 20 | 0 | — | 19 DIRTY_ANCILLA, 1 CRASH |

- With CDKM templates in the loop, the LLM **reproduces** the seeded
  construction in its own formulation (e.g. structurally distinct sources
  p0007/p0009/p0011 in `m4_supervised_n3.jsonl`) — repair-from-feedback
  works — but no run ever found a circuit below parity, and no archive cell
  beyond the CDKM-parity bin was ever populated.
- Unaided (no template), the LLM **never produced a valid adder** in 2 runs
  (0/6, 0/20); failure mode is consistently dirty ancillas (carries left
  uncomputed / computed then not restored).
- Conclusion: on this task family the agent is a reliable *reproducer*,
  not a *discoverer*.

## 3. The actual optimization: certificate-guided 2n−2 construction

### Span-model optimality certificate (n=3)

- Exhaustive 2-product span search over **all 651 first-op product vectors**
  fails (5.7 s) → no 2-Toffoli-product circuit computes the n=3 adder
  (≤2 ancillas, span model).
- A 3-product witness exists: `{a0·b0, a1·b1, a0·b0·(a1⊕b1)}` spans all
  degree≥2 parts of the sum bits (found in 0.1 s).
- Therefore the **multiplicative complexity of the n=3 in-place adder is
  exactly 3** in the span model.
- Rank certificates (dim of span of degree≥2 parts): T ≥ 2, 3, 4 for
  n = 3, 4, 5.

### Improved circuits (verified)

Skip the top MAJ/UMA pair of CDKM (the top carry it computes is discarded
anyway); read the top sum bit straight off the carry chain. Single carry
ancilla stays clean (returns to |0>).

| n | Construction | T | Q | TQ | vs CDKM |
|---|---|---|---|---|---|
| 3 | `baselines_adder/t4_adder_n3.py` | **4** | 7 | **28** | T −33%, TQ −33% |
| 4 | `baselines_adder/t6_adder_n4.py` | **6** | 9 | **54** | T −25%, TQ −25% |

`t4_adder_n3.py` uses T=4, within 1 of the span-model optimum (MC=3); the
T=3 span decomposition exists algebraically but a clean 3-Toffoli circuit in
this harness's circuit model was not found (the direct draft
`t3_adder_n3.py` fails `invalid[DIRTY_ANCILLA]` — kept as evidence of the
span-model vs circuit-model gap).

### Honesty / novelty note

**2n−2 Toffoli in-place adders are known** (Takahashi–Kunihiro use 2n−2
Toffolis with zero ancillas; our variant keeps 1 carry ancilla). This is
**not claimed as a novel gate-count record** and on its own is not
publishable. What this study contributes and verifies:

1. Harness-verified improved circuits (T/TQ −25…−33% vs the CDKM baseline
   the agent was seeded with), plus an **exact span-model optimality
   certificate for n=3** (MC = 3: 2 impossible by exhaustive search,
   3 witnessed).
2. A matched, multi-run audit of **where the LLM agent actually helps**
   (reaching parity from templates; repairing invalid proposals from
   feedback) and where it does not (beating a known construction; producing
   any valid adder unaided). In the original runs the optimization itself
   came from certificate-guided hand construction, not from the agent —
   see §3a for the follow-up that makes it agent-discoverable.

## 3a. Agent-machinery update: structural mutation makes the win agent-discoverable

Motivated by §2's honest result (base agent = reproducer, never a
discoverer), we extended the **candidate-generation machinery** itself and
re-ran a matched ablation:

**New machinery** (all in the loop, not in the evaluator):
- `RedundantPairElimRewriter` (`sota_harness/mutators.py`): a *general
  structural mutation operator*. It detects a matched forward/backward
  helper-call loop pair over a single reusable work register (the
  ripple-carry MAJ/UMA shape) and eliminates the top iteration of both,
  re-routing the top output bit to be read directly off the carry chain.
  2 Toffolis saved; the discarded top carry is provably dead (mod-2^n
  adder drops the carry-out). Task-agnostic pattern rewrite: it never
  references adder semantics, only the loop/call/register structure.
- Canonical-source dedup + variant rotation (`sota_harness/evolution.py`):
  whitespace-normalized sources are only evaluated once (JSONL `dedup`
  records); parent-variant and repair-parent pointers rotate instead of
  always taking the first element. Cuts wasted evaluations drastically
  (30-iter template runs: 1–2 real evals vs 30 before) with identical
  outcomes (verified: `--no-dedup` reproduces old behavior; the f24 loop
  found 0 valids before and after).

**Matched ablation** — 14 arms, 30 iterations each, same CDKM seeds, same
budgets; arms differ only in the mutator pipeline (`base` =
Mbu+AncillaPack; `structural` = Mbu+AncillaPack+PairElim) and LLM on/off,
across n=3–6 (no-LLM arms at every n; LLM arms at n=3–5). Logs:
`results_ablation/*.jsonl`, aggregate:
`results_ablation/ablation_summary.json`.

| Arm | n | Evals (valid) | Dedup skips | Best TQ | Found 2n−2 |
|---|---|---|---|---|---|
| no-LLM base | 3 | 1 (1) | 29 | 42 | no |
| no-LLM **structural** | 3 | 2 (2) | 28 | **28** | **yes** |
| no-LLM base | 4 | 1 (1) | 29 | 72 | no |
| no-LLM **structural** | 4 | 2 (2) | 28 | **54** | **yes** |
| no-LLM base | 5 | 1 (1) | 29 | 110 | no |
| no-LLM **structural** | 5 | 2 (2) | 28 | **88** | **yes** |
| no-LLM base | 6 | 1 (1) | 29 | 156 | no |
| no-LLM **structural** | 6 | 2 (2) | 28 | **130** | **yes** |
| LLM base | 3 | 4 (3) | 26 | 42 | no |
| LLM **structural** | 3 | 11 (9) | 19 | **28** | **yes** |
| LLM base | 4 | 3 (2) | 27 | 72 | no |
| LLM **structural** | 4 | 9 (6) | 21 | **54** | **yes** |
| LLM base | 5 | 3 (2) | 27 | 110 | no |
| LLM **structural** | 5 | 8 (7) | 22 | **88** | **yes** |

**Provenance (from raw JSONL)**: in every structural arm the first
28/54/88/130 hit is the `cdkm`-family rewriter splice — byte-identical to
the offline `RedundantPairElimRewriter` output (loop bounds `range(n-1)`/
`range(n-2, -1, -1)` + the two insert `ctx.cx` lines), `diff`-verified at
every n=3–6. The loop therefore reaches the improved construction in
**one mutation step from the CDKM seed**, at every size, in both no-LLM
and LLM settings. In the LLM structural arms the LLM later *reproduces*
the target TQ in its own family (first at iter 6 for n=3, iter 9 for n=4,
iter 5 for n=5) — only **after** the rewriter's discovery entered the
archive context, i.e. archive feedback amplifies the discovery. In all
seven base arms — including all three LLM arms — **the target is never
found** (best LLM-proposed valid: TQ 42/72/110, parity).

**Honest caveat**: the rewriter was designed with knowledge of the 2n−2
construction (we knew which pattern to make reachable). The claim is
*not* "the agent invented a better adder from scratch"; it is:
**once a general structural mutation operator is added to the
candidate-generation machinery, the known-better construction becomes
loop-discoverable in one step from the standard seed, under matched
budgets — and without it, the LLM never gets there.** That is the
mechanism-level contribution: operator design as the lever, with a
matched ablation isolating its effect.

## 3b. Operator transfer: the gate-mirror subtractor (held-out task family)

§3a's rewriter was derived on the adder's MAJ-first gate ordering. To test
whether it generalizes beyond that exact ordering — or silently overfits
one task family — we built the adder's **gate-mirror**: the in-place
subtractor `|a>|b> -> |a-b mod 2^n>|b>`, obtained by reversing every gate
of the CDKM adder (all gates self-inverse). Same machinery
(`run-subtractor`/`eval-subtractor`), mirrored CDKM seeds, mirrored 2n−2
ground truths (TQ 28/54 at n=3/4, eval-verified; mirrored baselines 42/72).

**Phase A — the boundary (operator unchanged, commit dcee826).** Offline,
the rewriter returns `None` on the mirrored template: its register
extraction requires a both-indexed `ctx.cx(b[i], a[i])` inside the
*forward* helper body, and the mirror's forward stage is UMA, whose cx's
only touch the carry wire. Matched 30-iter noLlm arms (n=3/4) confirm the
boundary in-loop: structural ≡ base *exactly* — 1 eval, 29 dedup skips,
parity 42/72 in all four arms. Logs: `results_ablation/subA_noLlm_*.jsonl`.

**Phase B — minimal generalization.** One order-agnostic fallback (21
lines, no task semantics): when the forward body lacks a both-indexed cx,
take the register pair from the *backward* helper body (maj_rev ends with
`ctx.cx(b[i], a[i])`). Adder regression is byte-clean: offline rewriter
outputs at n=3–6 are byte-identical pre/post edit, and the live adder n=3
structural arm still finds TQ 28 in one step
(`results_ablation/regression/`).

**Transfer result (matched arms, 30 iters, same seeds/budgets):**

| Phase | Setting | n | Base best TQ | Structural best TQ | First hit (structural) |
|---|---|---|---|---|---|
| A unchanged op | no-LLM | 3 | 42 | 42 (= base) | never |
| A unchanged op | no-LLM | 4 | 72 | 72 (= base) | never |
| B generalized op | no-LLM | 3 | 42 | **28** | iter 1 (p0002, splice) |
| B generalized op | no-LLM | 4 | 72 | **54** | iter 1 (p0002, splice) |
| B generalized op | LLM | 3 | 42 | **28** | iter 2 (p0003, splice) |

Every discovery record is byte-identical to the offline generalized-op
splice (diff-verified) — the loop reaches the mirrored 2n−2 construction
in **one mutation step** from the mirrored CDKM seed. LLM arms: base —
16 evals, target never found (the LLM's only valid circuit is a parity
reproduction at iter 8); structural — the splice finds TQ 28 at iter 2,
and the LLM later *reproduces* it in its own family at iter 23 (archive
feedback, the same amplification pattern as the adder). Bonus negative
controls (pure-LLM, no template): 28+10 evals, 0 valid. Aggregate:
`results_ablation/subtractor_ablation_summary.json` (10 arms).

**Honest framing.** The operator was still designed with knowledge of the
2n−2 construction, and the subtractor is that construction's gate-mirror —
so this is not invention and not a new gate-count claim (both families are
known-order, Takahashi–Kunihiro style). The claim is narrower and sharper:
**a structural operator derived on one task family's gate ordering hits a
clean, measurable generalization boundary on a held-out mirror family; a
minimal order-agnostic fix that touches no task semantics restores full
transfer — the known-better construction becomes loop-discoverable in one
step under matched budgets, while every control (base mutators, LLM-only)
never reaches it.** Reproducing phase A requires checking out dcee826
(the pre-generalization rewriter).

## 4. Proven vs claimed

**Proven** (deterministic evaluator / exhaustive search, artifacts on disk):
- All T/Q/TQ numbers in the tables above (`eval-adder`, exit 0,
  re-verified 2026-11-24).
- MC(n=3) = 3 in the span model (651-first-op exhaustive 2-product failure
  + 3-product witness).
- Agent-run statistics, counted from the raw JSONL logs (see §2 table).
- §3a ablation: 14/14 arms summarized in
  `results_ablation/ablation_summary.json` (evals, dedup skips, valid
  counts, TQ trajectories, invalid categories), each number re-derived
  from the raw JSONL logs in the same directory.
- §3b transfer study: 10/10 subtractor arms summarized in
  `results_ablation/subtractor_ablation_summary.json`; every structural
  discovery record byte-compared (`==`) against the offline rewriter
  splice (`results_ablation/regression/post_mirror_n{3,4}.py`), all
  True; adder regression pre/post generalization diff-clean at n=3–6;
  subtractor baselines/ground truths eval-verified via `eval-subtractor`
  (42/72, 28/54).

**Explicitly not claimed**:
- No novelty for the 2n−2 Toffoli count (known order).
- No claim that the agent beat the baseline in the §2-era runs — it did
  not; §3a's ablation shows the structural-mutator loop reaching
  28/54/88/130 in one step, but the operator was designed knowing the
  target.
- No claim that this is publishable as a new gate-count record.

## 5. Reproduce

```bash
PYTHONPATH=. python3 -m sota_harness.cli eval-adder --n 3 baselines_adder/t4_adder_n3.py
PYTHONPATH=. python3 -m sota_harness.cli eval-adder --n 4 baselines_adder/t6_adder_n4.py
PYTHONPATH=. python3 -m sota_harness.cli run-adder --n 3 --iters 12 \
    --log results_adder_m4/m4_supervised_n3.jsonl --snapshot ..._archive.json
# controls: --no-llm (mutation/template only), --no-template (pure LLM)

# §3a matched ablation (no-LLM arms below; for the LLM arms drop --no-llm
# and name the log llm_n${n}_${arm}.jsonl; n=3-5 for LLM arms, n=3-6 no-LLM):
for arm in base structural; do
  for n in 3 4 5 6; do
    PYTHONPATH=. python3 -m sota_harness.cli run-adder --n $n --iters 30 \
        --mutators $arm --no-llm --quiet \
        --log results_ablation/noLlm_n${n}_${arm}.jsonl
  done
done
python3 results_ablation/summarize_ablation.py   # -> ablation_summary.json

# §3b subtractor transfer study (gate-mirror of the adder):
PYTHONPATH=. python3 -m sota_harness.cli eval-subtractor --n 3 baselines_subtractor/t4_sub_n3.py   # TQ 28
PYTHONPATH=. python3 -m sota_harness.cli eval-subtractor --n 4 baselines_subtractor/t6_sub_n4.py   # TQ 54
# phase B arms (generalized op; phase A = same commands at commit dcee826):
for arm in base structural; do
  for n in 3 4; do
    PYTHONPATH=. python3 -m sota_harness.cli run-subtractor --n $n --iters 30 \
        --mutators $arm --no-llm --quiet \
        --log results_ablation/subB_noLlm_n${n}_${arm}.jsonl
  done
done
# LLM pair (n=3, template on — matched to the adder LLM arms):
PYTHONPATH=. python3 -m sota_harness.cli run-subtractor --n 3 --iters 30 \
    --mutators base --quiet --log results_ablation/subB_llm_n3_base.jsonl
PYTHONPATH=. python3 -m sota_harness.cli run-subtractor --n 3 --iters 30 \
    --mutators structural --quiet --log results_ablation/subB_llm_n3_structural.jsonl
python3 results_ablation/summarize_ablation.py   # -> subtractor_ablation_summary.json
```
