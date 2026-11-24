#!/usr/bin/env python3
"""Aggregate ablation JSONL logs into per-task summary JSONs.

Outputs:
  results_ablation/ablation_summary.json           -- adder arms (cycles 1-2)
  results_ablation/subtractor_ablation_summary.json -- subtractor arms (cycle 3)

Each JSONL record: {iteration, program_id, family, verdict, valid, toffoli,
qubits, tq, notes} or a dedup-skip record {verdict: "dedup", dedup: true, ...}.
Groups by (task, phase, llm_on, n, arm) and reports evals, dedup skips, best
TQ, and the TQ trajectory of valid discoveries.

Subtractor phases:
  unchanged_op  -- phase A: the UNCHANGED dcee826 operator (boundary mapping;
                   expected structural == base == mirrored 2n parity)
  generalized_op -- phase B: the order-agnostic operator (register extraction
                   falls back to the backward stage body; transfer test)
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

ARMS = [
    # (jsonl stem, llm_on, n, mutator arm)
    ("noLlm_n3_base", False, 3, "base"),
    ("noLlm_n3_structural", False, 3, "structural"),
    ("noLlm_n4_base", False, 4, "base"),
    ("noLlm_n4_structural", False, 4, "structural"),
    ("noLlm_n5_base", False, 5, "base"),
    ("noLlm_n5_structural", False, 5, "structural"),
    ("noLlm_n6_base", False, 6, "base"),
    ("noLlm_n6_structural", False, 6, "structural"),
    ("llm_n3_base", True, 3, "base"),
    ("llm_n3_structural", True, 3, "structural"),
    ("llm_n4_base", True, 4, "base"),
    ("llm_n4_structural", True, 4, "structural"),
    ("llm_n5_base", True, 5, "base"),
    ("llm_n5_structural", True, 5, "structural"),
]

CDKM = {3: 42.0, 4: 72.0, 5: 110.0, 6: 156.0}   # baseline TQ (2n Toffolis)
TARGET = {3: 28.0, 4: 54.0, 5: 88.0, 6: 130.0}  # 2n-2 construction TQ

# Subtractor (gate-mirror task family, cycle 3): same constructions mirrored.
SUBTRACTOR_ARMS = [
    # (jsonl stem, llm_on, n, mutator arm, phase)
    ("subA_noLlm_n3_base", False, 3, "base", "unchanged_op"),
    ("subA_noLlm_n3_structural", False, 3, "structural", "unchanged_op"),
    ("subA_noLlm_n4_base", False, 4, "base", "unchanged_op"),
    ("subA_noLlm_n4_structural", False, 4, "structural", "unchanged_op"),
    ("subB_noLlm_n3_base", False, 3, "base", "generalized_op"),
    ("subB_noLlm_n3_structural", False, 3, "structural", "generalized_op"),
    ("subB_noLlm_n4_base", False, 4, "base", "generalized_op"),
    ("subB_noLlm_n4_structural", False, 4, "structural", "generalized_op"),
    ("subB_llm_n3_base", True, 3, "base", "generalized_op"),
    ("subB_llm_n3_structural", True, 3, "structural", "generalized_op"),
]

SUB_BASELINE = {3: 42.0, 4: 72.0}   # mirrored 2n CDKM subtractor TQ
SUB_TARGET = {3: 28.0, 4: 54.0}     # mirrored 2n-2 subtractor TQ


def summarize(path: pathlib.Path, n: int, baseline_tq: float | None = None,
              target_tq: float | None = None) -> dict:
    baseline = CDKM[n] if baseline_tq is None else baseline_tq
    target = TARGET[n] if target_tq is None else target_tq
    evals = skips = 0
    valid_traj = []          # [(program_id, tq)] in evaluation order
    invalid_cats: dict[str, int] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("dedup"):
            skips += 1
            continue
        evals += 1
        if r.get("valid"):
            valid_traj.append((r.get("program_id") or r.get("id"),
                               float(r["tq"])))
        else:
            cat = (r.get("category") or
                   (r.get("verdict", "").split("[", 1)[-1].split("]", 1)[0]
                    or "unknown"))
            invalid_cats[cat] = invalid_cats.get(cat, 0) + 1
    best_tq = min((t for _, t in valid_traj), default=None)
    return {
        "n": n,
        "cdkm_baseline_tq": baseline,
        "target_2n_minus_2_tq": target,
        "evaluations": evals,
        "dedup_skips": skips,
        "valid": len(valid_traj),
        "best_tq": best_tq,
        "best_program": (min(valid_traj, key=lambda t: t[1])[0]
                         if valid_traj else None),
        "found_target_2n_minus_2": best_tq is not None and best_tq <= target,
        "improvement_over_cdkm": (round(baseline - best_tq, 1)
                                  if best_tq is not None else None),
        "valid_tq_trajectory": valid_traj,
        "invalid_categories": invalid_cats,
    }


results = {}
for stem, llm_on, n, arm in ARMS:
    path = HERE / f"{stem}.jsonl"
    if not path.exists():
        print(f"missing: {stem}.jsonl", file=sys.stderr)
        continue
    key = ("llm" if llm_on else "nollm") + f"_n{n}_{arm}"
    results[key] = summarize(path, n)

out = HERE / "ablation_summary.json"
out.write_text(json.dumps(results, indent=2) + "\n")
print(f"wrote {out} with {len(results)} arms")
for k, v in results.items():
    print(f"  {k}: best_TQ={v['best_tq']} evals={v['evaluations']} "
          f"target_found={v['found_target_2n_minus_2']}")

sub_results = {}
for stem, llm_on, n, arm, phase in SUBTRACTOR_ARMS:
    path = HERE / f"{stem}.jsonl"
    if not path.exists():
        print(f"missing: {stem}.jsonl", file=sys.stderr)
        continue
    key = f"{phase}_" + ("llm" if llm_on else "nollm") + f"_n{n}_{arm}"
    sub_results[key] = summarize(path, n, SUB_BASELINE[n], SUB_TARGET[n])

sub_out = HERE / "subtractor_ablation_summary.json"
sub_out.write_text(json.dumps(sub_results, indent=2) + "\n")
print(f"wrote {sub_out} with {len(sub_results)} arms")
for k, v in sub_results.items():
    print(f"  {k}: best_TQ={v['best_tq']} evals={v['evaluations']} "
          f"target_found={v['found_target_2n_minus_2']}")
