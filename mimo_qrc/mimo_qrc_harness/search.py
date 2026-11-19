"""Deterministic local search over the method space.

Scoring uses the exact evaluator口径 (seven points, strict wins, 70,656 bits
per frame per point) on the development split. The search explores:
  - the paper's method family (DTC-QRC / ESN) as the baseline,
  - sliced joint-ML search depths (top in 3..8),
  - QRC-gated sliced joint-ML algorithm variants.

Outputs a best.csv / best.json with per-SNR counts and the strict-win flag.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

HARNESS = Path(__file__).resolve().parent
GLM = HARNESS.parent / "mimo_qrc_glm"
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(GLM))

from evaluator import (  # noqa: E402
    ofdm_config, scored_bits, GRID, FIRST_SEEDS, MODEL_SEED_START,
)
from tqrcnet_sim.experiments.core import build_trial_observations  # noqa: E402
from tqrcnet_sim.models.lmmse import detect_estimated_csi  # noqa: E402
from tqrcnet_sim.mimo_ofdm.channel import noise_variance_from_ebn0  # noqa: E402

import methods_paper  # noqa: E402
import methods_innovative as mi  # noqa: E402


def _lmmse_reference(obs_list, cfg):
    refs = []
    for obs in obs_list:
        v = noise_variance_from_ebn0(obs.frame["x_time"], obs.cfg)
        base = detect_estimated_csi(
            obs.received_time, obs.frame["x_freq"][:cfg.n_ts],
            obs.frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices],
            obs.cfg, v, True)
        refs.append(scored_bits(base, obs))
    return refs


def evaluate_method(method_name, frames=1, split="development", workers=4):
    cfg = ofdm_config()
    det = methods_paper.METHODS.get(method_name, mi.METHODS.get(method_name))
    if det is None:
        raise ValueError(method_name)
    l_all = []
    c_all = []
    for f in range(frames):
        obs_list = build_trial_observations(cfg, GRID, FIRST_SEEDS[split] + f)
        model_seed = MODEL_SEED_START[split] + 100 * f
        cand = []
        for obs in obs_list:
            v = noise_variance_from_ebn0(obs.frame["x_time"], obs.cfg)
            p = np.asarray(det(obs.received_time,
                               obs.frame["x_freq"][:cfg.n_ts],
                               obs.frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices],
                               obs.cfg, v, model_seed))
            cand.append(scored_bits(p, obs))
        c_all.append(cand)
        l_all.append(_lmmse_reference(obs_list, cfg))
    l_refs = np.asarray(l_all).sum(axis=0).tolist()
    cand = np.asarray(c_all).sum(axis=0).tolist()
    wins = [c < l for c, l in zip(cand, l_refs)]
    return {
        "method": method_name,
        "frames": frames,
        "split": split,
        "candidate_errors": cand,
        "lmmse_errors": l_refs,
        "points_beaten": sum(wins),
        "target_met": all(wins),
        "mean_candidate_ber": sum(cand) / (7 * 70656 * frames),
        "mean_lmmse_ber": sum(l_refs) / (7 * 70656 * frames),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--split", choices=FIRST_SEEDS, default="development")
    p.add_argument("--out", type=Path, default=Path("search_report.json"))
    p.add_argument("--methods", nargs="+", default=None)
    args = p.parse_args()

    candidates = args.methods
    if candidates is None:
        candidates = (["paper_dtc_qrc", "paper_dtc_qrc_12", "paper_esn"] +
                      list(mi.METHODS))

    rows = []
    for m in candidates:
        t0 = time.monotonic()
        try:
            r = evaluate_method(m, frames=args.frames, split=args.split)
            r["wall_seconds"] = round(time.monotonic() - t0, 1)
            rows.append(r)
            print(f"[{m}] points_beaten={r['points_beaten']} "
                  f"target_met={r['target_met']} "
                  f"mean_ber={r['mean_candidate_ber']:.5f} "
                  f"wall={r['wall_seconds']}s")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[{m}] FAILED: {exc}")

    import json
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
