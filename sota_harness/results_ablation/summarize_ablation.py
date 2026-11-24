#!/usr/bin/env python3
"""Aggregate ablation JSONL logs into results_ablation/ablation_summary.json.

Each JSONL record: {iteration, program_id, family, verdict, valid, toffoli,
qubits, tq, notes} or a dedup-skip record {verdict: "dedup", dedup: true, ...}.
Groups by (llm_on, n, arm) and reports evals, dedup skips, best TQ, and the
TQ trajectory of valid discoveries.
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
    ("llm_n3_base", True, 3, "base"),
    ("llm_n3_structural", True, 3, "structural"),
    ("llm_n4_base", True, 4, "base"),
    ("llm_n4_structural", True, 4, "structural"),
]

CDKM = {3: 42.0, 4: 72.0}   # baseline TQ (2n Toffolis)
TARGET = {3: 28.0, 4: 54.0}  # 2n-2 construction TQ


def summarize(path: pathlib.Path, n: int) -> dict:
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
        "cdkm_baseline_tq": CDKM[n],
        "target_2n_minus_2_tq": TARGET[n],
        "evaluations": evals,
        "dedup_skips": skips,
        "valid": len(valid_traj),
        "best_tq": best_tq,
        "best_program": (min(valid_traj, key=lambda t: t[1])[0]
                         if valid_traj else None),
        "found_target_2n_minus_2": best_tq is not None and best_tq <= TARGET[n],
        "improvement_over_cdkm": (round(CDKM[n] - best_tq, 1)
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
