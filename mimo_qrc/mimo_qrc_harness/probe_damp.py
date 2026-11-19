"""Probe: pick damp for the fixed QRC residual-corrected sliced ML.

Runs on dev frame 0 at all 7 SNR points for several damp values and reports
per-method per-point errors plus strict wins vs LMMSE.
"""
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(HARNESS.parent / "mimo_qrc_glm"))

from evaluator import ofdm_config, scored_bits, GRID, FIRST_SEEDS, MODEL_SEED_START
from tqrcnet_sim.experiments.core import build_trial_observations
from tqrcnet_sim.models.lmmse import detect_estimated_csi
from tqrcnet_sim.mimo_ofdm.channel import noise_variance_from_ebn0
from methods_innovative import _qrc_ts_residual_bias, _lmmse_channels, _sliced_ml

cfg = ofdm_config()
obs_list = build_trial_observations(cfg, GRID, FIRST_SEEDS["development"] + 0)
model_seed = MODEL_SEED_START["development"] + 0

DARTS = {"resid_damp0025": 0.025, "resid_damp010": 0.10, "resid_damp025": 0.25,
         "resid_damp050": 0.50, "resid_damp100": 1.00}
errs = {k: [] for k in DARTS}
lmm = []
for obs in obs_list:
    v = noise_variance_from_ebn0(obs.frame["x_time"], cfg)
    known_ts = obs.frame["x_freq"][:cfg.n_ts]
    known_pilots = obs.frame["x_freq"][cfg.n_ts:][:, :, cfg.pilot_indices]
    base = detect_estimated_csi(obs.received_time, known_ts, known_pilots,
                                cfg, v, True)
    lmm.append(int(scored_bits(base, obs)))
    ch = _lmmse_channels(obs.received_time, known_ts, known_pilots, cfg, v)
    for k, damp in DARTS.items():
        bias = _qrc_ts_residual_bias(obs.received_time, known_ts, known_pilots,
                                     cfg, v, model_seed, damp=damp)
        pred, _ = _sliced_ml(obs.received_time, known_ts, known_pilots, cfg, v,
                             top=4, channels=ch, soft_ref_correction=bias)
        errs[k].append(int(scored_bits(pred, obs)))

print("SNR           :", list(GRID))
print("LMMSE frame0  :", lmm)
for k, e in errs.items():
    print(f"{k:14s}:", e, "wins=", sum(a < b for a, b in zip(e, lmm)))
