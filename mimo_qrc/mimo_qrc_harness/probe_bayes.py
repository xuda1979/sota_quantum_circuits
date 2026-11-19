"""Prototype probe: does a per-subcarrier QAM-prior posterior detector
(with the same LMMSE channel estimate) beat the LMMSE BER reference?

Pure-Bayes uses the identical channel estimate (joint TS-LMMSE + conditional
pilot tracking) but exploits the discrete 16-QAM prior instead of linear MMSE.
If it beats LMMSE at all 7 points, the harness's headline innovation (a QRC
that temporally smooths/refines these posteriors) starts from a winning base.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GLM = Path(__file__).resolve().parent.parent / "mimo_qrc_glm"
sys.path.insert(0, str(GLM))

from evaluator import (  # noqa: E402
    ofdm_config, scored_bits, GRID, FIRST_SEEDS, MODEL_SEED_START,
)
from tqrcnet_sim.experiments.core import build_trial_observations  # noqa: E402
from tqrcnet_sim.mimo_ofdm.channel import noise_variance_from_ebn0  # noqa: E402
from tqrcnet_sim.mimo_ofdm.frame import remove_cp_fft  # noqa: E402
from tqrcnet_sim.models.lmmse import (  # noqa: E402
    LmmseChannelEstimator, equalize_lmmse, frequency_noise_variance,
)

LEVELS = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(10.0)


def qam16_alphabet() -> np.ndarray:
    """All 16-QAM points: levels + 1j*levels -> [16] complex, Es=1."""
    return (LEVELS[:, None] + 1j * LEVELS[None, :]).reshape(-1)


def candidate_matrix(top: int | None = None) -> np.ndarray:
    """All 4-stream candidate vectors: [4, 16^4] complex (top=None).

    If top is given per-stream top-k slice for sliced-ML: 16^4 reduced to top^4
    via the caller supplying the subset. Kept for reference; sliced candidates
    are constructed inside detect_sliced below.
    """
    alphabet = qam16_alphabet()
    combos = np.arange(16**4)
    digits = np.zeros((4, 16**4), dtype=int)
    for t in range(4):
        digits[t] = combos % 16
        combos = combos // 16
    return alphabet[digits].astype(np.complex128)  # [4, 256]


def channel_estimates(received_time, known_ts, known_pilots, cfg, variance):
    """Return per-data-symbol estimated channels [Ndata, Nsc, Nr, Nt]."""
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    estimator = LmmseChannelEstimator(cfg, variance)
    estimator.fit_ts(received[:cfg.n_ts], known_ts)
    channels = []
    for index, symbol in enumerate(received[cfg.n_ts:]):
        estimator.update_pilots(symbol[:, cfg.pilot_indices], known_pilots[index])
        channels.append(estimator.frequency_channel())
    return np.asarray(channels)  # [Ndata, Nsc, Nr, Nt]


def detect_posterior(received_time, known_ts_frequency, known_pilots, cfg,
                     noise_variance, model_seed):
    """Per-subcarrier exhaustive Bayesian posterior (QAM prior, estimated H).

    Output identical interface to the evaluator: [Ndata, Nt, Nsc] symbol
    estimates (posterior mean per stream, i.e. MMSE given the QAM prior).
    """
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    channels = channel_estimates(received_time, known_ts_frequency,
                                 known_pilots, cfg, noise_variance)
    sigma2_f = frequency_noise_variance(noise_variance, cfg)
    candidates = candidate_matrix()  # [Nt=4, 65536]
    data_idx = np.asarray(cfg.resolved_data_indices)

    prediction = np.zeros((cfg.n_data, cfg.nt, cfg.nsc), dtype=np.complex128)
    for i in range(cfg.n_data):
        H = channels[i][data_idx]  # [48, Nr, Nt]
        y = received[cfg.n_ts + i].T[data_idx]  # [48, Nr]
        # d = y - H s : einsum over Nt; y[k, r], H[k, r, t]
        hc = np.einsum("krt,tc->krc", H, candidates)  # [48, Nr, 65536]
        d = y[:, :, None] - hc
        energy = np.sum(np.abs(d) ** 2, axis=1)  # [48,65536]
        best = np.argmin(energy, axis=1)  # [48]
        chosen = candidates[:, best]  # [4, 48]
        prediction[i, :, data_idx] = chosen.T
    return prediction


def detect_sliced(received_time, known_ts_frequency, known_pilots, cfg,
                  noise_variance, model_seed, top: int = 4):
    """Sliced joint ML per subcarrier.

    Step 1: per-stream reference from the estimated channel
    (W = Hermitian-matched filter, then slice). Step 2: keep the top-k
    alphabet symbols per stream nearest that reference and exhaustively
    check the top^4 combinations against |y - H s|^2.
    Runtime scaling: top=4 -> 256 combos per subcarrier (fast).
    """
    from tqrcnet_sim.models.lmmse import (LmmseChannelEstimator,
                                          equalize_lmmse)
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    channels = channel_estimates(received_time, known_ts_frequency,
                                 known_pilots, cfg, noise_variance)
    alphabet = qam16_alphabet()
    data_idx = np.asarray(cfg.resolved_data_indices)
    n_k = len(data_idx) if np.ndim(data_idx) else len(np.atleast_1d(data_idx))
    n_k = int(n_k)

    # per-stream combos of the top indices (digits), 4-tuples
    combos = np.arange(top**4)
    digits = np.zeros((4, top**4), dtype=int)
    for t in range(4):
        digits[t] = combos % top
        combos = combos // top

    prediction = np.zeros((cfg.n_data, cfg.nt, cfg.nsc), dtype=np.complex128)
    lmmse_det = LmmseChannelEstimator(cfg, noise_variance)
    lmmse_det.fit_ts(received[:cfg.n_ts], known_ts_frequency)
    for i in range(cfg.n_data):
        H = channels[i][data_idx]                      # [K, Nr, Nt]
        y = received[cfg.n_ts + i].T[data_idx]         # [K, Nr]
        # reference: per-stream LMMSE-equalized symbols (well-tuned soft output)
        lmmse_det.update_pilots(received[cfg.n_ts + i][:, cfg.pilot_indices],
                                known_pilots[i])
        ref_full = equalize_lmmse(received[cfg.n_ts + i],
                                  lmmse_det.frequency_channel(),
                                  lmmse_det.noise_variance)  # [Nt, Nsc]
        ref = ref_full[:, data_idx].T                  # [K, Nt]
        dist = np.abs(ref[:, :, None] - alphabet[None, None, :]) ** 2
        top_idx = np.argpartition(dist, top - 1, axis=-1)[..., :top]  # [K,Nt,top]
        top_sym = alphabet[top_idx]                     # [K, Nt, top]
        chosen = np.empty((n_k, cfg.nt), dtype=np.complex128)
        for k in range(n_k):
            s_set = top_sym[k]                          # [Nt, top]
            cand = np.empty((cfg.nt, top**4), dtype=np.complex128)
            for t in range(cfg.nt):
                cand[t] = s_set[t][digits[t]]
            hc = np.einsum("rt,tc->rc", H[k], cand)     # [Nr, top^4]
            energy = np.sum(np.abs(y[k][:, None] - hc) ** 2, axis=0)
            chosen[k] = cand[:, int(np.argmin(energy))]
        prediction[i, :, data_idx] = chosen
    return prediction


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="development")
    p.add_argument("--frames", type=int, default=1)
    args = p.parse_args()

    cfg_ = ofdm_config()
    import time as _time
    for f in range(args.frames):
        obs_list = build_trial_observations(cfg_, GRID, FIRST_SEEDS[args.split] + f)
        model_seed = MODEL_SEED_START[args.split] + 100 * f
        r_errors = []
        t0 = _time.monotonic()
        for obs in obs_list:
            variance = noise_variance_from_ebn0(obs.frame["x_time"], obs.cfg)
            res = detect_sliced(obs.received_time,
                                obs.frame["x_freq"][:cfg_.n_ts],
                                obs.frame["x_freq"][cfg_.n_ts:][:, :, cfg_.pilot_indices],
                                obs.cfg, variance, model_seed, top=4)
            r_errors.append(scored_bits(res, obs))
        print(f"frame {f}: sliced-ML wall {_time.monotonic()-t0:.1f}s")
        # LMMSE reference
        from tqrcnet_sim.models.lmmse import detect_estimated_csi
        l_errors = []
        for obs in obs_list:
            variance = noise_variance_from_ebn0(obs.frame["x_time"], obs.cfg)
            pred = detect_estimated_csi(
                obs.received_time, obs.frame["x_freq"][:cfg_.n_ts],
                obs.frame["x_freq"][cfg_.n_ts:][:, :, cfg_.pilot_indices],
                obs.cfg, variance, True)
            l_errors.append(scored_bits(pred, obs))
        print(f"frame {f}: sliced-ML BER")
        for snr, r, l in zip(GRID, r_errors, l_errors):
            print(f"  {snr:3d} dB  bayes={r:6d} ({r/70656:.5f})  lmmse={l:6d} ({l/70656:.5f})  win={r < l}")


if __name__ == "__main__":
    main()
