"""Candidate archive: MAP-Elites over the (T, Q) plane + extras.

Paper: MAP-Elites cells on (Toffoli, qubits), lowest TQ per cell, plus
families (one per ideation architecture), DISCOVERY notes, limited archive
context passed to each iteration.

Improvements here:
  - explicit Pareto frontier tracking (any cell may be dominated; the
    frontier is what matters for reporting),
  - lineage graph (parent links) with per-family best history,
  - stagnation counter per family driving cross-family migration,
  - JSONL snapshot/restore for reproducible runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from .circuit import Program
from .evaluator import Verdict


@dataclass
class Entry:
    program: Program
    verdict: Verdict
    family: str
    generation: int

    @property
    def id(self) -> str:
        return self.program.id

    @property
    def tq(self) -> float:
        return self.verdict.tq

    def cell(self, t_res: float = 1.25, q_res: int = 4) -> tuple[int, int]:
        """MAP-Elites cell: log-spaced T axis, linear Q axis."""
        t_cell = int(math.log(max(self.verdict.toffoli, 1.0)) / math.log(t_res))
        q_cell = self.verdict.qubits // q_res
        return (t_cell, q_cell)


class Archive:
    def __init__(self, t_res: float = 1.25, q_res: int = 4):
        self.t_res, self.q_res = t_res, q_res
        self.cells: dict[tuple[int, int], Entry] = {}
        self.frontier: list[Entry] = []      # Pareto (min T, min Q)
        self.families: dict[str, list[Entry]] = {}
        self.invalid: dict[str, list[Entry]] = {}  # repair candidates
        self.lineage: dict[str, Optional[str]] = {}  # id -> parent id
        self.discoveries: list[str] = []     # travelling notes
        self.stagnation: dict[str, int] = {}
        self.best: Optional[Entry] = None
        self.history: list[dict] = []        # per-iteration records

    # -- insertion -------------------------------------------------------
    def offer(self, program: Program, verdict: Verdict, family: str,
              generation: int) -> bool:
        """Insert if valid and (cell-empty | better TQ in cell). Returns kept.

        Invalid programs are not archived as elites, but their lineage is
        recorded and they are tracked per-family as repair candidates: the
        paper's loop feeds evaluator feedback on an INVALID program back to
        the rewriter (Extended Data Fig. 1: dirty five-products -> +MBU).
        """
        entry = Entry(program, verdict, family, generation)
        self.lineage[program.id] = program.parent
        if not verdict.valid:
            self.invalid.setdefault(family, []).append(entry)
            return False
        fam = self.families.setdefault(family, [])
        fam.append(entry)
        kept = False
        key = entry.cell(self.t_res, self.q_res)
        cur = self.cells.get(key)
        if cur is None or entry.tq < cur.tq:
            self.cells[key] = entry
            kept = True
        self._update_frontier(entry)
        if self.best is None or entry.tq < self.best.tq:
            self.best = entry
            self.stagnation[family] = 0
        else:
            self.stagnation[family] = self.stagnation.get(family, 0) + 1
        for note in program.notes:
            if note not in self.discoveries:
                self.discoveries.append(note)
        return kept

    def _update_frontier(self, entry: Entry) -> None:
        T, Q = entry.verdict.toffoli, entry.verdict.qubits
        dominated = [e for e in self.frontier
                     if e.verdict.toffoli <= T and e.verdict.qubits <= Q]
        if dominated and any(e.verdict.toffoli < T or e.verdict.qubits < Q
                             for e in dominated):
            pass  # strictly dominated: not on frontier
        else:
            self.frontier = [e for e in self.frontier
                             if not (T <= e.verdict.toffoli and Q <= e.verdict.qubits)]
            self.frontier.append(entry)

    # -- context for the next iteration (the paper's limited view) --------
    def context_for(self, family: str, max_programs: int = 3) -> dict:
        fam_best = sorted(self.families.get(family, []), key=lambda e: e.tq)
        elite = fam_best[:max_programs]
        cell_diverse: list[Entry] = []
        for e in sorted(self.families.get(family, []),
                        key=lambda e: e.tq):
            if e not in elite and all(
                    e.cell(self.t_res, self.q_res) != o.cell(self.t_res, self.q_res)
                    for o in cell_diverse):
                cell_diverse.append(e)
            if len(elite) + len(cell_diverse) >= 2 * max_programs:
                break
        return {
            "elite": [(e.id, e.verdict.one_line()) for e in elite],
            "cell_diverse": [(e.id, e.verdict.one_line()) for e in cell_diverse],
            "discoveries": list(self.discoveries[-8:]),
            "stagnation": self.stagnation.get(family, 0),
            "best_overall": (self.best.id, self.best.verdict.one_line())
                            if self.best else None,
        }

    def parent_for(self, family: str) -> Optional[Entry]:
        """Select the parent program to rewrite: best in family, but rotate
        to a diverse alternative when stagnant (improvement over fixed
        elitist selection: escapes local optima)."""
        fam = sorted(self.families.get(family, []), key=lambda e: e.tq)
        if not fam:
            return None
        stag = self.stagnation.get(family, 0)
        if stag and stag % 3 == 0 and len(fam) > 1:
            return fam[stag % len(fam)]
        return fam[0]

    # -- persistence -------------------------------------------------------
    def snapshot(self, path: str) -> None:
        rec = {
            "cells": {f"{k[0]},{k[1]}": asdict_entry(v) for k, v in self.cells.items()},
            "discoveries": self.discoveries,
            "lineage": self.lineage,
            "stagnation": self.stagnation,
            "history": self.history,
        }
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)

    def log_iteration(self, record: dict) -> None:
        self.history.append(record)


def asdict_entry(e: Entry) -> dict:
    return {
        "id": e.id, "family": e.family, "generation": e.generation,
        "source": e.program.source, "notes": e.program.notes,
        "parent": e.program.parent,
        "valid": e.verdict.valid, "toffoli": e.verdict.toffoli,
        "qubits": e.verdict.qubits, "tq": e.verdict.tq,
        "category": e.verdict.category,
    }
