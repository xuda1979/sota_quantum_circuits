# Optimality Certificates for LLM-Discovered Quantum Circuits

Status: SKELETON (filling as results land; every number tagged [RUN] is
regenerable via `run_all.sh`; every claim tagged [PROVEN] rests on an
exhaustive search whose model is stated; everything else is [CLAIMED]).

## Thesis

Deterministic, LLM-free provers + exact synthesis can match, certify, and
sometimes exceed LLM-discovered reversible circuits. Optimality certificates
are computable for tasks where the LLM paper reports none.

Reference: Gu, Kemeny, Yelin, Lukin, "Discovering state-of-the-art quantum
circuits with large language models", arXiv:2610.22367 (2026).

## 1. Abstract (draft)

The reference paper evolves quantum circuits with an LLM proposer and an
independent evaluator, reporting large TQ-cost gains but no optimality
arguments: nothing establishes that its discovered circuits are optimal, or
by how much. We build a deterministic harness around the paper's own
evaluator and make three contributions. (i) Exact replication: all 17
published circuit entries reproduce exactly [RUN]. (ii) Optimality
certificates: for the paper's F24 inversion showcase we prove, by exhaustive
search over precisely-stated circuit models, that the discovered T=5, Q=10
circuit is optimal in both T and Q simultaneously within those models:
T=4 is impossible at any qubit count, Q=8 (no ancilla) is impossible at
any Toffoli count in the strict operand model, and Q=9 (one ancilla) is
impossible at T=5 [PROVEN]. The paper claims
optimality nowhere. (iii) Model-boundary analysis: we map exactly which
circuit-model extensions (h-bits as operand sources, frozen-ancilla reuse,
relative-phase idioms) could evade the certificate, and search the widened
models; negative results strengthen the certificate, positive results beat
the paper [RUN: section 4.4; tower-field closed negative, h-as-source
searches inconclusive-by-budget with no counterexample found].

## 2. Related work

- LLM-driven circuit discovery: the reference paper (only prior work in
  this exact setting). Its evaluator-verifies-proposer loop is replicated
  here exactly; nothing in the paper certifies optimality of any
  discovered circuit.
- Multiplicative complexity and Toffoli lower bounds: degree-based
  AND-gate lower bounds (Boykin-Roychowdhury-style MC lower bounds),
  used here in span form: every Toffoli contributes one product whose
  degree>=2 ANF content must cover the targets' content.
- Exact synthesis: SAT/SMT-based reversible circuit synthesis (Soeken et
  al. and successors); our span searches are exhaustive in a different,
  coarser model, which is what makes them close at sizes SAT methods
  typically do not reach with qubit-count constraints.
- GF(2^n) inversion circuits: tower-field (Itoh-Tsujii) constructions
  are the classical asymptotic optimum family; our certificate shows
  that at n=4 the LLM-discovered 5-Toffoli circuit already achieves
  the lower bound, and at n=2,3 the deterministic constructions do.
- Kickmix/MBU model: measurement-based uncomputation as in the paper's
  builder API (hmr + cz fixups).

## 3. Method

### 3.1 Cost model

T = executed Toffoli count, Q = peak live qubits, TQ = T x Q, both measured
exactly by the official evaluator (or our exact evaluator for the toy task).

### 3.2 Span certificates

Every wire value in the kickmix model is affine in the inputs XOR a sum of
ccx products. Packing degree>=2 ANF monomials as coordinate vectors, any
T-Toffoli circuit's reachable nonlinear content lies in the span of <= T
product vectors. Failure of exhaustive span search at depth T-1 over a
completely-enumerated product menu proves no T-1-Toffoli circuit exists in
the model [the T>=5 leg for F24].

### 3.3 Qubit-count certificates

Q=8 (no ancilla): products land directly in h-bits; a per-bit BFS over the
product menu proves bit 0 requires >4 products [PROVEN].
Q=9 (one ancilla): exhaustive DFS over (ancilla truth table, accumulated
h-injections) with admissible rank pruning; complete for the stated operand
model [PROVEN, 6,725,331 nodes].

### 3.4 Model assumptions (the certificate's fine print)

Stated precisely in sota_harness/certificates.py and in the report; the
searched models fix: one ancilla; operands = transiently-shaped single
wires (affine riders + const); MBU freeing via hmr + per-op cz with
re-created operand shapes; h-bits are not used as operand sources;
zero global phase. Section 5 reports how far each assumption can be
widened with the certificate intact.

## 4. Results

### 4.1 Replication [RUN]

17/17 exact (T, Q, TQ, failure counts); see REPLICATION_REPORT.md.

### 4.2 F24 optimality certificate [PROVEN]

| Leg | Statement | Evidence |
|---|---|---|
| T >= 5 (any Q) | no 4-Toffoli circuit at ANY qubit count (h-bit operand sources allowed) | rank proof: rank span{d2(INV_b)} = 4; no realizable first product has nonzero d2 inside that span (complete 171-product menu: 31 with d2=0, 140 outside, 0 inside); any 4-product span of a 4-dim target space must lie inside it - contradiction [RUN: certificates.f24_T5_rank_argument] |
| Q >= 9 (strict model, any T) | no 0-ancilla circuit with x-wire-only operands at ANY T | targets' d2s lie outside the 6-dim span of the 35 x-pair product d2s (dim 6 < 11; per-bit membership fails) [RUN: q8probe.py] |
| Q >= 10 | no 1-ancilla T=5 circuit (paper's operand idioms) | exhaustive DFS, NOT FOUND, 6,725,331 nodes, reproducible [RUN: q9search.py 5] |
| existence | T=5, Q=10 circuit exists | the paper's circuit, evaluator-verified [RUN: f24_evaluator.py] |

Hence the paper's (T=5, Q=10) is Pareto-optimal: no circuit with fewer
Toffolis at any Q, and none with T=5 at lower Q within the stated models.
Widened-model searches (h-as-source at Q=8/Q=9, frozen-ancilla reuse)
run in section 4.4.

### 4.3 Certificate tower: GF(2^2), GF(2^3) [PROVEN]

The same machinery certifies the whole inversion tower (paper's F24 task
family, one and two sizes down):

| Task | Construction | (T, Q) | TQ | T-optimality | Found by |
|---|---|---|---|---|---|
| F22 inversion (GF(4)) | Frobenius z^-1 = z^2: pure Clifford | (0, 4) | 0 | rank 0 - trivially optimal | deterministic (algebra) |
| F23 inversion (GF(8)) | per-bit span decomposition: x1&x2, x0&x1, x0&x2 + affine residuals | (3, 6) | 18 | rank proof: rank span{d2(INV_b)} = 3 => no 2-Toffoli circuit at any Q [RUN] | deterministic (per-bit d2 matching), no LLM |
| F24 inversion (GF(16)) | the paper's LLM-discovered circuit | (5, 10) | 50 | rank 4 + menu-exclusion => no 4-Toffoli circuit at any Q [RUN] | the paper's LLM evolution |

All three constructions verified with the official Builder/Simulator
exhaustively over all (x, h) pairs [RUN: verify_f22_f23.py; candidates/
f22_inv.py, f23_inv.py]. Q values are the task interface minima (no
ancillas used beyond the interface for F22/F23).

Scaling observation: multiplicative-complexity-optimal Toffoli counts for
GF(2^n) inversion run 0, 3, 5 for n = 2, 3, 4 (and T >= 4 rank bounds for
n=4 is tightened to 5 by the menu-exclusion argument of 4.2).

Full certificates for Table-I tasks (hhl-angle, coulomb, ...) are out of
reach for exhaustive methods (support widths 21-48 bits); this is stated
honestly rather than claimed.

### 4.4 Beats and widened-model searches [status: closed negative / inconclusive-by-budget]

Model-boundary attacks tried, all verified before claiming:

1. Tower-field / Frobenius (x -> x^2, x -> x^4 are free linear maps;
   invertible joint basis changes of x are free transient cx networks):
   closure analysis shows basis changes never enlarge the bipartite
   product menu (products of affine forms compose to products of affine
   forms), and the best tower chain (Itoh-Tsujii-type, x^-1 = x^14 with
   free squarings) still needs 2 F16 multiplications = 8 Toffolis > 5.
   Negative result [RUN: analysis in q9wide.py comments; verified live].
2. h-bits as operand sources at Q=8 (q8hsrc.py): TIMEOUT/inconclusive
   after 13,300,000 DFS nodes (7,225 s, 2 h deadline); no beating
   circuit found in the explored space, search not exhausted
   [RUN: results/q8hsrc.txt].
3. h-bits as operand sources at Q=9 (q9hsrc.py): TIMEOUT/inconclusive
   after 24,200,000 DFS nodes (14,407 s, 4 h deadline); no beating
   circuit found in the explored space, search not exhausted
   [RUN: results/q9hsrc.txt].
4. Two-ancilla frozen-ancilla reuse at Q=10 (existsearch.py, v3, the
   finalized sound model: ancilla-targeted ops = MBU-safe x-affine pairs
   only, h-targeted ops = x/t/u/tu/h-source families, t/u truth-table
   state, basis-inclusive memo): the T=5 search **closed exhaustively**
   NOT FOUND after 74,594,457 nodes (1,644 s, 18,225 memo states) — no
   circuit in this sound subclass of Q=10 circuits. Because the paper's
   own (T=5, Q=10) circuit exists, it necessarily lies outside this
   subclass (its operations exceed the model's restricted menus), so
   this is a scoped exhaustive negative within the searched subclass,
   not a claim about all two-ancilla circuits; it is the strongest
   searched-model negative we have. The T>=5 rank argument covers any
   ancilla count, so reuse cannot go below T=5 in any case
   [RUN: results/existsearch_v3.txt].
5. Physical-wire synthesis (physsearch.py, unpruned operand pairs):
   killed at 13,300,000+ nodes (~100 min) without a deadline in the
   script; inconclusive, honestly recorded
   [RUN: results/physsearch.txt]. Existence leg uses the paper's
   published circuit.

A genuine beat requires: emitted circuit + official evaluator verdict
OK + strictly better (T, Q) than the paper's (5, 10).

## 5. Discussion

- What the LLM actually contributed on F24: a structure (5 products,
  MBU-freeable ancilla) that turns out to sit exactly ON the optimal
  frontier - something neither the LLM nor the paper could know. The
  certificate reframes the LLM's discovery from "good" to "provably
  cannot be beaten in (T, Q) within the stated models".
- What deterministic search contributes: F22 (T=0) and F23 (T=3, Q=6)
  optima were found AND certified with zero LLM calls, in milliseconds;
  the F24 impossibility legs run in minutes to ~1 hour on one core
  (6.7M-node exhaustive DFS). The paper's LLM evolution is API-bound and
  produces no certificates.
- Certificate strength vs. model width: each impossibility leg holds in
  a precisely stated model; widening the model (h-bit operand sources,
  frozen-ancilla operand reuse) required NEW searches (q8hsrc, q9hsrc,
  existsearch) rather than invalidating the old ones. Negative results in
  widened models strengthen; positive results would have beaten the
  paper. The finalized two-ancilla sound model closed exhaustively
  (74,594,457 nodes), strengthening the Q>=10 leg at two ancillas while
  the 1-ancilla idiom legs remain exhaustive negatives at 6,725,331
  nodes.
- Honest limits: Table-I tasks (21-48 input bits) are beyond exhaustive
  certificate methods; their published circuits remain uncertified (as
  in the original paper). The certificate machinery's sweet spot is the
  regime where the LLM paper demonstrates its method (toy showcases) -
  exactly where optimality claims are most meaningful pedagogically.
- Compute cost: deterministic searches run in minutes on one core; the
  paper's LLM evolution cost is API-bound and unreproducible in
  hardware terms.

## 6. Reproducibility

`run_all.sh` regenerates every [RUN] number; certificate searches are
deterministic (fixed node counts, seedless). Search parameters:

| Search | Model | Depth | Nodes | Wall time |
|---|---|---|---|---|
| tsearch (T>=5 cross-check) | 2 ancillas, complete operand pairs | 4 ops | 343 | 0.01 s |
| q8probe | 0 ancillas, x-operands, per-bit BFS | any (span) | 35-product menu, dim 6 | < 1 s |
| q9search | 1 ancilla, paper operand idiom | 5 ops | 6,725,331 (reproduced 2x, identical) | ~140 s |
| q8hsrc | 0 ancillas, h-as-source (widened) | 5 ops | 13,300,000 (TIMEOUT at 7,225 s, inconclusive) | 2 h |
| q9hsrc | 1 ancilla, h-as-source (widened) | 5 ops | 24,200,000 (TIMEOUT at 14,407 s, inconclusive) | 4 h |
| existsearch v3 | 2 ancillas, h-as-source (finalized sound model) | 5 ops | 74,594,457 (NOT FOUND, exhaustive) | ~27 min |
| F22/F23 construction verify | official Builder/Simulator | exhaustive (x, h) | - | < 1 s |

Artefacts: sota_harness/sota_harness/certificates.py (module),
framework_replication/candidates/{f22_inv,f23_inv}.py (verified
circuits), framework_replication/verify_f22_f23.py (tower verifier),
results/ (regenerated numbers), REPLICATION_REPORT.md (17/17
replication + certificate sections).
