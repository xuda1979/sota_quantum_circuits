"""Impaired-observation builder for the nonlinear (PA + phase noise) study.

Mirrors tqrcnet_sim.experiments.core.build_trial_observations, inserting:
  1. Saleh PA distortion on the transmitted time waveform (per Tx antenna),
  2. the PA-distorted waveform through the unchanged EPA channel model,
  3. common-LO Wiener phase noise on the noiseless channel output,
  4. AWGN scaled by Eb/N0 computed from the CLEAN (pre-PA) waveform.

The frame dict (truth bits, intended x_freq/x_time) stays clean: the receiver
knows what was intended, while the actual transmitted waveform was distorted —
exactly the receiver-side mismatch that degrades linear detectors. All
downstream scoring (evaluator.scored_bits) and detectors are unchanged.

Eb/N0 convention: N0 is derived from the clean waveform's payload energy.
Because saleh_pa renormalizes output power to input power, this equals the
actual radiated power; only the distortion content differs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

HARNESS = Path(__file__).resolve().parent
GLM = HARNESS.parent / "mimo_qrc_glm"
for _p in (str(HARNESS), str(GLM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tqrcnet_sim.mimo_ofdm.channel import (  # noqa: E402
    ChannelRealization,
    generate_channel_realization,
    noise_variance_from_ebn0,
)
from tqrcnet_sim.mimo_ofdm.frame import generate_frame  # noqa: E402
from tqrcnet_sim.mimo_ofdm.impairments import apply_phase_noise, saleh_pa  # noqa: E402

PHASE_NOISE_SEED_OFFSET = 5000


@dataclass(frozen=True)
class ImpairedObservation:
    """One Eb/N0 point of the impaired scenario, shared by all detectors."""

    cfg: Any
    frame: dict
    received_time: np.ndarray
    channel_realization: ChannelRealization
    realization_seed: int
    x_time_pa: np.ndarray
    phase_noise: np.ndarray


def build_impaired_observations(base_cfg, ebn0_values, realization_seed,
                                ibo_db: float = 6.0,
                                sigma_phi: float = 3e-3):
    """Build paired impaired observations at the given Eb/N0 points.

    With ibo_db large (e.g. 100) and sigma_phi = 0 the builder reduces
    exactly to the clean build_trial_observations (same seeds, same noise).
    """
    if realization_seed < 0:
        raise ValueError("realization_seed must be non-negative")
    values = tuple(float(v) for v in ebn0_values)
    if not values:
        raise ValueError("need at least one Eb/N0 value")

    trial_cfg = replace(base_cfg, seed=realization_seed)
    frame = generate_frame(trial_cfg)
    x_clean = frame["x_time"]
    x_pa = saleh_pa(x_clean, ibo_db=ibo_db)
    realization = generate_channel_realization(x_pa, trial_cfg)
    pn_rng = np.random.default_rng(realization_seed + PHASE_NOISE_SEED_OFFSET)
    pn_received, phi = apply_phase_noise(
        realization.noiseless_received, sigma_phi, pn_rng)

    observations = []
    for ebn0_db in values:
        cfg = replace(trial_cfg, ebn0_db=ebn0_db)
        noise_variance = noise_variance_from_ebn0(x_clean, cfg, ebn0_db)
        received_time = (pn_received
                         + np.sqrt(noise_variance) * realization.unit_noise)
        observations.append(ImpairedObservation(
            cfg=cfg, frame=frame, received_time=received_time,
            channel_realization=realization,
            realization_seed=realization_seed,
            x_time_pa=x_pa, phase_noise=phi))
    return observations
