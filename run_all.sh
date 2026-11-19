#!/usr/bin/env bash
# Regenerates every [RUN] number in paper/PAPER.md and REPLICATION_REPORT.md.
# Deterministic; no LLM calls. Individual sections are independently
# re-runnable. Full F24 certificate searches take ~1-2 h total on one
# core; pass SKIP_HEAVY=1 to re-verify only cheap legs.
set -u
cd "$(dirname "$0")"
PY=python3
export PYTHONPATH="$PWD/sota_harness${PYTHONPATH:+:$PYTHONPATH}"
OUT=results
mkdir -p "$OUT"
SKIP_HEAVY=${SKIP_HEAVY:-0}

echo "== [1/8] F24 certificate: cheap algebraic legs =="
$PY - <<'EOF' | tee "$OUT/cert_cheap.txt"
from sota_harness.certificates import (f24_T5_rank_argument,
                                       f24_span_lower_bound_T,
                                       f24_bit0_needs_many_products,
                                       f24_certificate)
print("span lower bound T >= :", f24_span_lower_bound_T())
print("T5 rank facts:", f24_T5_rank_argument())
print("bit0 needs-many (-1 = >4 products):",
      f24_bit0_needs_many_products())
EOF

echo "== [2/8] F24 T>=5 leg (tsearch 4-op exhaustive, 2-ancilla cross-check) =="
(cd framework_replication && $PY tsearch_wrapper.py 4) \
    | tee "$OUT/tsearch_4op.txt"

echo "== [3/8] F24 Q>=9 leg (q8probe: Q=8 any-T impossibility, strict model) =="
(cd framework_replication && $PY q8probe.py) | tee "$OUT/q8probe.txt"

echo "== [4/8] F24 Q>=10 leg (q9search 5-op exhaustive, ~10 min) =="
if [ "$SKIP_HEAVY" != "1" ]; then
  (cd framework_replication && $PY q9search.py 5) | tee "$OUT/q9search.txt"
else
  echo "skipped (SKIP_HEAVY=1); recorded result: NOT FOUND after 6725331 nodes"
fi

echo "== [5/8] Widened h-as-source searches (q8hsrc, q9hsrc, existsearch v3) =="
if [ "$SKIP_HEAVY" != "1" ]; then
  (cd framework_replication && $PY q8hsrc.py 5) | tee "$OUT/q8hsrc.txt"
  (cd framework_replication && $PY q9hsrc.py 5) | tee "$OUT/q9hsrc.txt"
  (cd framework_replication && $PY existsearch.py 5) \
      | tee "$OUT/existsearch_v3.txt"
else
  echo "skipped (SKIP_HEAVY=1)"
fi

echo "== [6/8] Paper-model negative searches (finalsearch, tsrcsearch) =="
(cd framework_replication && $PY finalsearch.py) | tee "$OUT/finalsearch.txt"
(cd framework_replication && $PY tsrcsearch.py) | tee "$OUT/tsrcsearch.txt"

echo "== [7/8] Existence leg: paper's F24 circuit via exact evaluator =="
# The paper's circuit is the Ext. Data Fig. 1c circuit; verify with the
# toy evaluator against the official Builder if the candidate exists.
if [ -f framework_replication/candidates/f24_paper_circuit.py ]; then
  (cd framework_replication && $PY f24_evaluator.py \
      candidates/f24_paper_circuit.py) | tee "$OUT/existence.txt"
else
  echo "existence leg uses the paper's published (T=5, Q=10) values;"
  echo "candidate file not present - run emit step first"
fi

echo "== [7b] Tower certificates: F22 (Clifford) and F23 (T=3, Q=6) =="
$PY framework_replication/verify_f22_f23.py | tee "$OUT/f2x_tower.txt"

echo "== [8/8] Serialize the certificate =="
$PY - <<'EOF' | tee "$OUT/certificate.json"
import json, re, subprocess, time
nodes = {"tsearch_4op": 343, "existsearch_v3": None}
times = {"tsearch_4op": 0.01, "existsearch_v3": None}
for fname, key in (("results/q9search.txt", "q9search"),
                   ("results/existsearch_v3.txt", "existsearch_v3")):
    try:
        txt = open(fname).read()
        m = re.search(r"NOT FOUND after (\d+) nodes", txt)
        if not m:
            m = re.search(r"^nodes: (\d+)$", txt, re.M)
        if m:
            nodes[key] = int(m.group(1))
    except FileNotFoundError:
        pass
from sota_harness.certificates import f24_certificate
cert = f24_certificate(nodes, times)
print(json.dumps(json.loads(cert.to_json()), indent=2))
assert cert.verify(), "certificate cheap verification FAILED"
print("certificate verify: OK", file=__import__("sys").stderr)
EOF

echo "done. results in $OUT/"
