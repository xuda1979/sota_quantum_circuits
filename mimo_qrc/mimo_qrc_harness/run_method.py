"""Evaluate one detector method against the LMMSE reference at all seven
Eb/N0 points on the development or heldout split.

Usage:
  python run_method.py <method> [--split development|heldout] [--frames N]
                        [--workers W] [--out results.json]

The harness's methods live in methods_paper.py (paper's original) and
methods_innovative.py (harness innovations). This reproduces exactly the
evaluator's scoring (70,656 payload bits per frame per point, strict wins).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
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
    DEFAULT_FRAMES, BITS_PER_FRAME,
)
from tqrcnet_sim.experiments.core import build_trial_observations  # noqa: E402
from tqrcnet_sim.models.lmmse import detect_estimated_csi  # noqa: E402
from tqrcnet_sim.mimo_ofdm.channel import noise_variance_from_ebn0  # noqa: E402

import methods_paper  # noqa: E402
import methods_innovative  # noqa: E402


def get_method(name: str):
    if name in methods_paper.METHODS:
        return methods_paper.METHODS[name], "paper"
    if name in methods_innovative.METHODS:
        return methods_innovative.METHODS[name], "innovative"
    raise SystemExit(f"unknown method {name!r}; available: "
                     f"{sorted(methods_paper.METHODS)} + "
                     f"{sorted(methods_innovative.METHODS)}")


def evaluate_one_frame(received_time, known_ts, known_pilots, cfg, variance,
                       model_seed, detect_fn):
    pred = np.asarray(detect_fn(received_time, known_ts, known_pilots, cfg,
                                variance, model_seed))
    return pred


def evaluate(method_name, split="development", frames=None, workers=4,
             detect_fn=None):
    cfg = ofdm_config()
    if frames is None:
        frames = DEFAULT_FRAMES[split]
    det, source = get_method(method_name)

    def run_frame(f):
        obs_list = build_trial_observations(cfg, GRID, FIRST_SEEDS[split] + f)
        model_seed = MODEL_SEED_START[split] + 100 * f
        cand, ref = [], []
        for obs in obs_list:
            variance = noise_variance_from_ebn0(obs.frame["x_time"], obs.cfg)
            p = np.asarray(det(obs.received_time,
                               obs.frame["x_freq"][:cfg.n_ts],
                               obs.frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices],
                               obs.cfg, variance, model_seed))
            cand.append(scored_bits(p, obs))
            base = detect_estimated_csi(
                obs.received_time, obs.frame["x_freq"][:cfg.n_ts],
                obs.frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices],
                obs.cfg, variance, True)
            ref.append(scored_bits(base, obs))
        return cand, ref

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers, frames)) as pool:
        results = list(pool.map(run_frame, range(frames)))
    wall = round(time.monotonic() - t0, 2)

    bits = BITS_PER_FRAME * frames
    rows = []
    for i, snr in enumerate(GRID):
        c = sum(r[0][i] for r in results)
        l = sum(r[1][i] for r in results)
        rows.append({
            "ebn0_db": snr, "payload_bits": bits,
            "candidate_errors": c, "lmmse_errors": l,
            "candidate_ber": c / bits, "lmmse_ber": l / bits,
            "strict_win": c < l,
        })
    wins = sum(r["strict_win"] for r in rows)
    report = {
        "method": method_name, "source": source, "split": split,
        "frames": frames, "wall_seconds": wall,
        "mean_candidate_ber": sum(r["candidate_ber"] for r in rows) / 7.0,
        "mean_lmmse_ber": sum(r["lmmse_ber"] for r in rows) / 7.0,
        "points_beaten": wins, "target_met": wins == 7,
        "per_snr": rows,
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("method")
    p.add_argument("--split", choices=FIRST_SEEDS, default="development")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = evaluate(args.method, split=args.split, frames=args.frames,
                      workers=args.workers)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
