"""Starting detector. OpenEvolve can replace the complete QRC and its readout."""
# EVOLVE-BLOCK-START
import numpy as np
from tqrcnet_sim.experiments.core import DetectorSettings, make_detector
from tqrcnet_sim.mimo_ofdm.frame import ifft_with_cp, remove_cp_fft


def detect(received_time, known_ts_frequency, known_pilots, cfg,
           noise_variance, model_seed):
    """Return four streams of payload-period frequency-domain symbol estimates."""
    settings = DetectorSettings(qrc_qubits=4, qrc_cycles=4,
                                qrc_measurement_count=4,
                                esn_input_scale=0.001, ridge=0.0)
    detector = make_detector("dtc_qrc", cfg, settings, model_seed)
    ts_time = ifft_with_cp(known_ts_frequency, cfg.ncp, cfg.transform_normalization)
    detector.fit_ts(received_time[:cfg.n_ts], ts_time)
    pilots_frequency = np.zeros((cfg.n_data, cfg.nt, cfg.nsc), dtype=complex)
    pilots_frequency[:, :, cfg.pilot_indices] = known_pilots
    prediction = detector.detect_data_online(received_time[cfg.n_ts:], pilots_frequency)
    return remove_cp_fft(prediction, cfg.ncp, cfg.transform_normalization)
# EVOLVE-BLOCK-END
