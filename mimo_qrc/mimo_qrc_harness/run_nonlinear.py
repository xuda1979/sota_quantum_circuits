"""Evaluate detectors on the NONLINEAR scenario (Saleh PA + Wiener phase
noise), against the same LMMSE reference pipeline as the linear study.

The impaired observations are built by nonlinear_scenario.build_impaired_
observations: the PA distorts the transmitted waveform and phase noise
rotates the noiseless channel output, while the receiver sees exactly the
same inputs as in the linear study (received_time, known TS, known pilots,
cfg, noise variance from the clean waveform's Eb/N0). All detectors are
therefore run with unchanged code; only the channel the signal actually
traversed is impaired.

Usage:
  python run_nonlinear.py nl_qrc_top4 nl_esn_top4 [--split development]
      [--frames N] [--ibo 6.0] [--sigma-phi 3e-3] [--workers W]
      [--out report.json] [--methods-file] (list from methods_nonlinear)

LMMSE reference rows are computed once per (split, frames, impairment)
setting and shared by all methods in the same invocation.
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
from tqrcnet_sim.models.lmmse import detect_estimated_csi  # noqa: E402
from tqrcnet_sim.mimo_ofdm.channel import noise_variance_from_ebn0  # noqa: E402
from nonlinear_scenario import build_impaired_observations  # noqa: E402

import methods_nonlinear  # noqa: E402


def lmmse_errors_for_obs(obs):
    cfg = obs.cfg
    variance = noise_variance_from_ebn0(obs.frame["x_time"], cfg)
    base = detect_estimated_csi(
        obs.received_time, obs.frame["x_freq"][:cfg.n_ts],
        obs.frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices],
        cfg, variance, True)
    return scored_bits(base, obs), variance


def evaluate_method(name, split, frames, ibo_db, sigma_phi, workers,
                    shared_frames):
    """shared_frames: {frame_index: [(errors, variance, obs), ...]} computed
    once per impairment setting; reused across methods."""
    det = methods_nonlinear.METHODS[name]

    def run_frame(f):
        entries = shared_frames[f]
        model_seed = MODEL_SEED_START[split] + 100 * f
        cand = []
        for (_, variance, obs) in entries:
            cfg = obs.cfg
            p = np.asarray(det(obs.received_time,
                               obs.frame["x_freq"][:cfg.n_ts],
                               obs.frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices],
                               cfg, variance, model_seed))
            cand.append(scored_bits(p, obs))
        return cand

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers, frames)) as pool:
        results = list(pool.map(run_frame, range(frames)))
    wall = round(time.monotonic() - t0, 2)

    bits = BITS_PER_FRAME * frames
    rows = []
    for i, snr in enumerate(GRID):
        c = sum(r[i] for r in results)
        l = sum(shared_frames[f][i][0] for f in range(frames))
        rows.append({
            "ebn0_db": snr, "payload_bits": bits,
            "candidate_errors": c, "lmmse_errors": l,
            "candidate_ber": c / bits, "lmmse_ber": l / bits,
            "strict_win": c < l,
        })
    wins = sum(r["strict_win"] for r in rows)
    return {
        "method": name, "split": split, "frames": frames,
        "ibo_db": ibo_db, "sigma_phi": sigma_phi,
        "wall_seconds": wall,
        "mean_candidate_ber": sum(r["candidate_ber"] for r in rows) / 7.0,
        "mean_lmmse_ber": sum(r["lmmse_ber"] for r in rows) / 7.0,
        "points_beaten": wins, "target_met": wins == 7,
        "per_snr": rows,
    }


def build_shared_frames(cfg, split, frames, ibo_db, sigma_phi, workers):
    def one(f):
        obs_list = build_impaired_observations(
            cfg, GRID, FIRST_SEEDS[split] + f,
            ibo_db=ibo_db, sigma_phi=sigma_phi)
        return [(lmmse_errors_for_obs(o)[0],
                 noise_variance_from_ebn0(o.frame["x_time"], o.cfg), o)
                for o in obs_list]
    with ThreadPoolExecutor(max_workers=min(workers, frames)) as pool:
        return {f: entries for f, entries in
                enumerate(pool.map(one, range(frames)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("methods", nargs="*",
                   default=["nl_sliced_top4", "nl_cpe_top4", "nl_esn_top4",
                            "nl_qrc_top4"])
    p.add_argument("--split", choices=list(FIRST_SEEDS), default="development")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--ibo", type=float, default=6.0)
    p.add_argument("--sigma-phi", type=float, default=3e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not args.methods:
        args.methods = ["nl_sliced_top4", "nl_cpe_top4", "nl_esn_top4",
                        "nl_qrc_top4"]
    unknown = [m for m in args.methods
               if m not in methods_nonlinear.METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; available: "
                         f"{sorted(methods_nonlinear.METHODS)}")

    cfg = ofdm_config()
    frames = DEFAULT_FRAMES[args.split] if args.frames is None else args.frames
    t0 = time.monotonic()
    shared = build_shared_frames(cfg, args.split, frames, args.ibo,
                                 args.sigma_phi, args.workers)
    print(f"shared impaired frames built in {time.monotonic()-t0:.1f}s")

    reports = []
    for m in args.methods:
        r = evaluate_method(m, args.split, frames, args.ibo, args.sigma_phi,
                            args.workers, shared)
        reports.append(r)
        print(f"[{m}] points_beaten={r['points_beaten']}/7 "
              f"mean_ber={r['mean_candidate_ber']:.5f} "
              f"(lmmse {r['mean_lmmse_ber']:.5f}) wall={r['wall_seconds']}s")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
