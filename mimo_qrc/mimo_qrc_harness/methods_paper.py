"""Paper's original method: DTC-QRC based T-RCNet detector (verbatim from
the source package's initial_program.py), plus the ESN backend variant and
the parameter grid the paper defines.

Every method here returns the detector's frequency-domain symbol estimates
of shape (92, 4, 64) via the package's own remove_cp_fft convention.
"""
from __future__ import annotations

import numpy as np

from tqrcnet_sim.experiments.core import DetectorSettings, make_detector
from tqrcnet_sim.mimo_ofdm.frame import ifft_with_cp, remove_cp_fft


def _run_dtc_qrc(received_time, known_ts_frequency, known_pilots, cfg,
                 noise_variance, model_seed, settings: DetectorSettings,
                 backend: str = "dtc_qrc"):
    detector = make_detector(backend, cfg, settings, model_seed)
    ts_time = ifft_with_cp(known_ts_frequency, cfg.ncp, cfg.transform_normalization)
    detector.fit_ts(received_time[:cfg.n_ts], ts_time)
    pilots_frequency = np.zeros((cfg.n_data, cfg.nt, cfg.nsc), dtype=complex)
    pilots_frequency[:, :, cfg.pilot_indices] = known_pilots
    prediction = detector.detect_data_online(received_time[cfg.n_ts:], pilots_frequency)
    return remove_cp_fft(prediction, cfg.ncp, cfg.transform_normalization)


# The paper's exact starting detector (initial_program.py).
def detect_paper_dtc_qrc(received_time, known_ts_frequency, known_pilots, cfg,
                         noise_variance, model_seed):
    """Paper's original DTC-QRC detector (4 qubit, 4 cycles, 4 measurements,
    1 layer, window = 4, no bias, GAW-RLS online pilot updating)."""
    settings = DetectorSettings(qrc_qubits=4, qrc_cycles=4,
                                qrc_measurement_count=4,
                                esn_input_scale=0.001, ridge=0.0)
    return _run_dtc_qrc(received_time, known_ts_frequency, known_pilots, cfg,
                        noise_variance, model_seed, settings)


# Paper-defined knobs: the QRC ideal-sim params in the package.
def detect_paper_dtc_qrc_12(received_time, known_ts_frequency, known_pilots,
                            cfg, noise_variance, model_seed):
    """DTC-QRC with the paper's ideal-simulation QRC hyperparameters
    (m = 12 Floquet cycles, 4 measurement segments, g = 0.84)."""
    settings = DetectorSettings(qrc_qubits=6, qrc_cycles=12,
                                qrc_measurement_count=4, qrc_g=0.84,
                                ridge=0.0)
    return _run_dtc_qrc(received_time, known_ts_frequency, known_pilots, cfg,
                        noise_variance, model_seed, settings)


# The paper's classical RC backend (T-RCNet uses ESN reservoir layers).
def detect_paper_esn(received_time, known_ts_frequency, known_pilots, cfg,
                     noise_variance, model_seed):
    """T-RCNet with the classical ESN reservoir (32 units, spectral radius
    0.2), the same Algorithm 1 flow."""
    settings = DetectorSettings(esn_states=32, esn_spectral_radius=0.2,
                                esn_input_scale=1.0, ridge=0.0)
    return _run_dtc_qrc(received_time, known_ts_frequency, known_pilots, cfg,
                        noise_variance, model_seed, settings, backend="esn")


METHODS = {
    "paper_dtc_qrc": detect_paper_dtc_qrc,
    "paper_dtc_qrc_12": detect_paper_dtc_qrc_12,
    "paper_esn": detect_paper_esn,
}
