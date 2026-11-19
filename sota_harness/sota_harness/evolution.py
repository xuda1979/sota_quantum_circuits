"""The evolutionary loop: propose -> evaluate -> archive -> feedback.

Paper-faithful core (Methods):
  1. ideation: propose up to five distinct architectures,
  2. loop: select parent (archive context + feedback), rewrite, evaluate,
  3. archive keeps (T, Q)-elite programs per family; discoveries travel.

Harness improvements:
  - deterministic proposers/rewriters make the loop LLM-free-capable,
  - every iteration logged (JSONL) for reproducibility,
  - stagnation-driven parent rotation and cell-diverse context,
  - stops early on certified optimality (T <= task lower bound).
"""

from __future__ import annotations

import json
import time
from typing import Optional

from .archive import Archive
from .circuit import Program
from .evaluator import Evaluator, Verdict
from .mutators import MutationPipeline
from .proposers import Proposer


class Evolution:
    def __init__(self, task, evaluator: Evaluator, proposer: Proposer,
                 mutators: Optional[MutationPipeline] = None,
                 archive: Optional[Archive] = None, max_iters: int = 100,
                 log_path: Optional[str] = None):
        self.task = task
        self.evaluator = evaluator
        self.proposer = proposer
        self.mutators = mutators or MutationPipeline([])
        self.archive = archive or Archive()
        self.max_iters = max_iters
        self.log_path = log_path

    def run(self, verbose: bool = True) -> Archive:
        t0 = time.time()
        for it in range(self.max_iters):
            # -- propose (first iterations: ideation; then rewrite) --------
            feedback = None
            parent_entry = None
            if self.archive.best is not None:
                fam = self.archive.best.family
                parent_entry = self.archive.parent_for(fam)
            elif any(self.archive.invalid.values()):
                # paper-faithful repair: rewrite the most recent invalid
                # program, feeding its evaluator feedback (Ext. Data Fig. 1)
                fam = max(self.archive.invalid,
                          key=lambda f: len(self.archive.invalid[f]))
                parent_entry = self.archive.invalid[fam][-1]
            if parent_entry is not None and self.mutators.rewriters:
                prog = parent_entry.program
                variants = self.mutators.all_variants(prog)
                if not variants:
                    prog = self.proposer.propose(self.archive, None)
                else:
                    prog = variants[0]
                    feedback = {
                        "parent_id": parent_entry.id,
                        "parent_source": parent_entry.program.source,
                        "parent_verdict": parent_entry.verdict.one_line(),
                    }
            else:
                prog = self.proposer.propose(self.archive, None)

            # -- evaluate (ground truth) ------------------------------------
            verdict = self.evaluator.evaluate(prog.source)
            kept = self.archive.offer(prog, verdict, prog.family, it)
            self.archive.log_iteration({
                "iter": it, "id": prog.id, "family": prog.family,
                "verdict": verdict.one_line(), "kept": kept,
                "t": round(time.time() - t0, 2),
            })
            if verbose:
                print(f"[{it:3d}] {prog.id} ({prog.family}) "
                      f"{verdict.one_line()}"
                      + ("  [kept]" if kept else ""))
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({
                        "iter": it, "id": prog.id, "family": prog.family,
                        "source": prog.source, "verdict": verdict.one_line(),
                        "valid": verdict.valid, "toffoli": verdict.toffoli,
                        "qubits": verdict.qubits, "tq": None
                        if verdict.tq == float("inf") else verdict.tq,
                    }) + "\n")

            # -- early stop on certified optimality ------------------------
            if (verdict.valid and self.task.toffoli_lower_bound is not None
                    and verdict.toffoli <= self.task.toffoli_lower_bound):
                if verbose:
                    print(f"[{it:3d}] certified optimal "
                          f"(T={verdict.toffoli:g} <= LB="
                          f"{self.task.toffoli_lower_bound}); stopping")
                break
        return self.archive
