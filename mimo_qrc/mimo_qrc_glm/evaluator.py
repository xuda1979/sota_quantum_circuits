"""Fixed MIMO benchmark. The OpenEvolve entry point is evaluate(program_path)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import lru_cache
import importlib.util
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqrcnet_sim.experiments.core import build_trial_observations
from tqrcnet_sim.mimo_ofdm.channel import noise_variance_from_ebn0
from tqrcnet_sim.mimo_ofdm.config import OfdmConfig
from tqrcnet_sim.mimo_ofdm.qam import qam16_to_bits
from tqrcnet_sim.models.lmmse import detect_estimated_csi

GRID = (0, 5, 10, 15, 20, 25, 30)
FIRST_SEEDS = {"development": 410000, "heldout": 420000}
DEFAULT_FRAMES = {"development": 10, "heldout": 30}
MODEL_SEED_START = {"development": 930000, "heldout": 960000}
BITS_PER_FRAME = 70656
REFERENCE_NAME = "joint TS-LMMSE + conditional pilot tracking (estimated CSI)"


@lru_cache(maxsize=1)
def ofdm_config():
    values = json.loads((ROOT / "ofdm.json").read_text(encoding="utf-8"))
    for key in ("pilot_indices", "data_subcarrier_indices"):
        values[key] = tuple(values[key])
    return OfdmConfig(**values)


def scored_bits(prediction, obs):
    cfg = obs.cfg
    expected_shape = (cfg.n_data, cfg.nt, cfg.nsc)
    if prediction.shape != expected_shape or prediction.dtype.kind not in "ciuf":
        raise ValueError(f"detect() must return a numeric array of shape {expected_shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("detect() returned NaN or Inf")
    truth = obs.frame["data_bits"]
    mask = obs.frame["data_mask"]
    total, errors = 0, 0
    for symbol in range(cfg.n_data):
        for stream in range(cfg.nt):
            known = truth[(symbol + cfg.n_ts, stream)]
            estimate = qam16_to_bits(prediction[symbol, stream, mask])
            if estimate.shape != known.shape:
                raise ValueError("Invalid payload coverage")
            errors += int(np.count_nonzero(estimate != known))
            total += known.size
    if total != BITS_PER_FRAME:
        raise ValueError(f"Expected {BITS_PER_FRAME} payload bits, scored {total}")
    return errors


def lmmse_reference(obs):
    # The original baseline uses the simulator noise variance, known TS and pilots.
    cfg, frame = obs.cfg, obs.frame
    variance = noise_variance_from_ebn0(frame["x_time"], cfg)
    prediction = detect_estimated_csi(
        obs.received_time, frame["x_freq"][:cfg.n_ts],
        frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices], cfg, variance, True,
    )
    return scored_bits(prediction, obs)


def candidate_worker(source, inputs, output):
    # This process receives receiver inputs only. Ground-truth BER stays in the parent.
    spec = importlib.util.spec_from_file_location("receiver_candidate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Expected a Python candidate file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    detect = getattr(module, "detect", None)
    if not callable(detect):
        raise ValueError("Candidate must define detect(received_time, known_ts_frequency, known_pilots, cfg, noise_variance, model_seed)")
    with np.load(inputs, allow_pickle=False) as data:
        cfg = replace(ofdm_config(), seed=0, ebn0_db=float(data["ebn0_db"].item()))
        prediction = np.asarray(detect(data["received"], data["known_ts"],
                                       data["known_pilots"], cfg, float(data["variance"].item()),
                                       int(data["model_seed"].item())))
        if prediction.shape != (cfg.n_data, cfg.nt, cfg.nsc):
            raise ValueError("detect() returned the wrong shape")
        if prediction.dtype.kind not in "ciuf" or not np.isfinite(prediction).all():
            raise ValueError("detect() must return finite numeric symbol estimates")
    np.save(output, prediction.astype(np.complex128), allow_pickle=False)


def evaluate_point(program, obs, model_seed, deadline):
    cfg, frame = obs.cfg, obs.frame
    with tempfile.TemporaryDirectory(prefix="mimo-qrc-") as temp:
        directory = Path(temp)
        inputs, output = directory / "receiver_inputs.npz", directory / "output.npy"
        candidate = directory / "candidate.py"
        candidate.write_bytes(program)
        np.savez(inputs, received=obs.received_time,
                 known_ts=frame["x_freq"][:cfg.n_ts],
                 known_pilots=frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices],
                 variance=noise_variance_from_ebn0(frame["x_time"], cfg),
                 ebn0_db=cfg.ebn0_db, model_seed=model_seed)
        # Preserve runtime/device settings. Keep model service credentials in the parent.
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME",
                   "USERPROFILE", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
                   "HIP_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
                   "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
        env = {name: value for name, value in os.environ.items() if name.upper() in allowed}
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("seven-point frame", 0)
        with (directory / "worker.log").open("wb") as log:
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(ROOT / "evaluator.py"), "--worker",
                 str(candidate), str(inputs), str(output)], cwd=directory, env=env,
                stdout=log, stderr=log, timeout=remaining,
            )
        if result.returncode or not output.is_file():
            with (directory / "worker.log").open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 1500))
                detail = log.read().decode("utf-8", errors="replace")
            raise ValueError(f"Candidate subprocess failed (exit {result.returncode}). {detail}")
        return scored_bits(np.load(output, allow_pickle=False), obs)


def evaluate_frame(program, communication_seed, model_seed, frame_timeout):
    observations = build_trial_observations(ofdm_config(), GRID, communication_seed)
    deadline = time.monotonic() + frame_timeout
    candidate_errors = [evaluate_point(program, obs, model_seed, deadline) for obs in observations]
    reference_errors = [lmmse_reference(obs) for obs in observations]
    return candidate_errors, reference_errors


def summarize(frames):
    bits = BITS_PER_FRAME * len(frames)
    rows = []
    for i, snr in enumerate(GRID):
        qrc = sum(a[i] for a, _ in frames)
        lmmse = sum(b[i] for _, b in frames)
        rows.append({"ebn0_db": snr, "payload_bits": bits, "qrc_errors": qrc,
                     "lmmse_errors": lmmse, "qrc_ber": qrc / bits,
                     "lmmse_ber": lmmse / bits, "qrc_minus_lmmse": (qrc - lmmse) / bits,
                     "strict_win": qrc < lmmse})
    mean_ber = sum(row["qrc_ber"] for row in rows) / len(rows)
    mean_excess = sum(max(0.0, row["qrc_minus_lmmse"]) for row in rows) / len(rows)
    wins = sum(row["strict_win"] for row in rows)
    target_met = wins == len(GRID)
    # Every seven-point winner outranks every non-winner. Before that, smaller
    # BER and smaller shortfalls give useful feedback even when no point wins.
    score = (2.0 - mean_ber if target_met else 1.0 - (mean_ber + mean_excess) / 2)
    metrics = {"combined_score": score, "valid": 1.0, "target_met": float(target_met),
               "points_beaten": float(wins), "mean_qrc_ber": mean_ber,
               "mean_lmmse_ber": sum(row["lmmse_ber"] for row in rows) / len(rows),
               "mean_positive_gap": mean_excess}
    return metrics, rows


def evaluate_benchmark(program_path, *, split="development", frames=None, workers=4, frame_timeout=240):
    if split not in FIRST_SEEDS:
        raise ValueError(f"Unknown split: {split}")
    frames = DEFAULT_FRAMES[split] if frames is None else frames
    if not 1 <= frames <= 10000 or workers < 1 or frame_timeout <= 0:
        raise ValueError("Invalid frame count, worker count, or timeout")
    started = time.monotonic()
    program = Path(program_path).read_bytes()
    def run_frame(i):
        return evaluate_frame(program, FIRST_SEEDS[split] + i,
                              MODEL_SEED_START[split] + 100 * i, frame_timeout)

    # At 1 worker (also useful for a GPU-backed candidate), run one frame at a time.
    with ThreadPoolExecutor(max_workers=min(workers, frames)) as pool:
        results = list(pool.map(run_frame, range(frames)))
    metrics, rows = summarize(results)
    return {"split": split, "frames_per_point": frames, "reference": REFERENCE_NAME,
            "candidate_sha256": sha256(program).hexdigest(), "metrics": metrics, "per_snr": rows,
            "frame_counts": [{"qrc_errors": a, "lmmse_errors": b} for a, b in results],
            "wall_seconds": round(time.monotonic() - started, 3)}


def evaluate(program_path, *, frames=10, workers=4, frame_timeout=240):
    from openevolve.evaluation_result import EvaluationResult

    try:
        report = evaluate_benchmark(program_path, frames=frames, workers=workers, frame_timeout=frame_timeout)
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        message = (f"One full seven-point frame exceeded the {frame_timeout}s subprocess limit."
                   if isinstance(exc, subprocess.TimeoutExpired) else str(exc))
        return EvaluationResult(
            metrics={"combined_score": -1.0, "valid": 0.0, "target_met": 0.0,
                     "points_beaten": 0.0, "mean_qrc_ber": 1.0},
            artifacts={"failure": message[-1600:]},
        )
    feedback = {key: report[key] for key in ("frames_per_point", "reference", "per_snr", "wall_seconds")}
    return EvaluationResult(metrics=report["metrics"], artifacts={"benchmark": json.dumps(feedback, indent=2)})


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        candidate_worker(*map(Path, sys.argv[2:]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--split", choices=FIRST_SEEDS, default="development")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame-timeout", type=int, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_benchmark(args.program, split=args.split, frames=args.frames,
                                workers=args.workers, frame_timeout=args.frame_timeout)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frame_counts"}, indent=2))


if __name__ == "__main__":
    main()
