"""Sealed QRC detector candidate: DTC-QRC residual-corrected sliced joint-ML.

Interface (package evaluator):
    detect(received_time, known_ts_frequency, known_pilots, cfg,
           noise_variance, model_seed) -> (92, 4, 64) complex symbol estimates

Quantum contribution: a real simulated DTC Floquet quantum reservoir produces
input-dependent measured Z_i / Z_i Z_j features for every received symbol. A
ridge readout, trained ONLY on the 8 knowable training symbols, maps the
reservoir features to the LMMSE soft-symbol residual at ALL subcarriers. The
predicted per-symbol, per-carrier correction is applied to the LMMSE soft
reference that feeds the sliced joint-ML detector, so the quantum features
genuinely alter the decisions at the SCORED data subcarriers.

Ablation: damp -> 0 recovers plain sliced joint-ML top-5 (no QRC influence).
"""
from __future__ import annotations

import numpy as np

from tqrcnet_sim.mimo_ofdm.frame import remove_cp_fft
from tqrcnet_sim.models.lmmse import (
    LmmseChannelEstimator,
    equalize_lmmse,
    frequency_noise_variance,
)
from tqrcnet_sim.reservoirs.dtc_qrc import DTCQRCReservoir

_LEVELS = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(10.0)


def _alphabet():
    return (_LEVELS[:, None] + 1j * _LEVELS[None, :]).reshape(-1)


S_ALPHABET = _alphabet()

# Readout damping: strength of the QRC-predicted correction. damp=0 turns the
# quantum contribution off (ablation) and recovers plain sliced top-5.
_DAMP = 0.5


def _lmmse_channels(received, known_ts_frequency, known_pilots, cfg, variance):
    estimator = LmmseChannelEstimator(cfg, variance)
    estimator.fit_ts(received[:cfg.n_ts], known_ts_frequency)
    channels = []
    for index, symbol in enumerate(received[cfg.n_ts:]):
        estimator.update_pilots(symbol[:, np.asarray(cfg.pilot_indices)],
                                known_pilots[index])
        channels.append(estimator.frequency_channel())
    return np.asarray(channels)


def _qrc_ts_residual_bias(received, known_ts_frequency, known_pilots, cfg,
                          variance, model_seed):
    """Per-data-symbol soft-reference correction (92, Nt, Nsc) predicted by the
    DTC-QRC reservoir readout trained on the known-TS residuals."""
    pilot_idx = np.asarray(cfg.pilot_indices)
    reservoir = DTCQRCReservoir(
        input_dim=2 * cfg.nr * cfg.nsc,
        n_qubits=8,
        cycles=12,
        measurement_count=4,
        g=0.84,
        seed=model_seed,
        input_encoding="pca_dense_angle",
        input_scale=0.5,
        include_input=False,
    )
    collapsed = received.reshape(cfg.n_total, cfg.nr * cfg.nsc)
    real_seq = np.concatenate([collapsed.real, collapsed.imag], axis=1)
    reservoir.fit_input_encoder(real_seq[:cfg.n_ts])
    feats = reservoir.forward_sequence(real_seq)          # [NSym, n_feat]

    estimator = LmmseChannelEstimator(cfg, variance)
    estimator.fit_ts(received[:cfg.n_ts], known_ts_frequency)
    fvar = frequency_noise_variance(variance, cfg)
    soft_all = np.zeros((cfg.n_total, cfg.nt, cfg.nsc), dtype=np.complex128)
    for i in range(cfg.n_ts):
        soft_all[i] = equalize_lmmse(received[i],
                                     estimator.frequency_channel(), fvar,
                                     gain_normalization=True)
    for i, sym in enumerate(received[cfg.n_ts:]):
        estimator.update_pilots(sym[:, pilot_idx], known_pilots[i])
        soft_all[cfg.n_ts + i] = equalize_lmmse(
            sym, estimator.frequency_channel(), fvar, gain_normalization=True)

    # Train on knowable TS known-symbol residuals over EVERY carrier.
    res_ts = known_ts_frequency - soft_all[:cfg.n_ts]     # [n_ts, Nt, Nsc]
    res_ts = np.moveaxis(res_ts, 2, 1)                    # [n_ts, Nsc, Nt]
    Y = np.concatenate([res_ts.real.reshape(cfg.n_ts * cfg.nsc, cfg.nt),
                        res_ts.imag.reshape(cfg.n_ts * cfg.nsc, cfg.nt)],
                       axis=1)                            # [ts*Nsc, 2*Nt]
    X = np.repeat(feats[:cfg.n_ts], cfg.nsc, axis=0)

    n_feat = X.shape[1]
    gram = X.T @ X + 1e-3 * np.eye(n_feat)
    W = np.linalg.solve(gram, X.T @ Y)

    X_data = np.repeat(feats[cfg.n_ts:], cfg.nsc, axis=0)
    pred = X_data @ W
    pred_r = pred[:, :cfg.nt].reshape(cfg.n_data, cfg.nsc, cfg.nt)
    pred_i = pred[:, cfg.nt:].reshape(cfg.n_data, cfg.nsc, cfg.nt)
    bias = (pred_r + 1j * pred_i)
    return _DAMP * np.moveaxis(bias, 1, -1)


def _slice_top_indices(ref_k, top):
    dist = np.abs(ref_k[:, :, None] - S_ALPHABET[None, None, :]) ** 2
    return np.argpartition(dist, top - 1, axis=-1)[..., :top]


def _joint_ml_subcarriers(H_k, y_k, top_idx_k):
    n_k, n_t, comb = top_idx_k.shape
    combos = np.arange(comb ** n_t)
    digits = np.zeros((n_t, comb ** n_t), dtype=int)
    for t in range(n_t):
        digits[t] = combos % comb
        combos = combos // comb
    chosen = np.empty((n_k, n_t), dtype=np.complex128)
    for k in range(n_k):
        s_set = S_ALPHABET[top_idx_k[k]]
        n_combos = comb ** n_t
        cand = np.empty((n_t, n_combos), dtype=np.complex128)
        for t in range(n_t):
            cand[t] = s_set[t][digits[t]]
        hc = np.einsum("rt,tc->rc", H_k[k], cand)
        energy = np.sum(np.abs(y_k[k][:, None] - hc) ** 2, axis=0)
        chosen[k] = cand[:, int(np.argmin(energy))]
    return chosen


def _sliced_ml(received, known_ts_frequency, known_pilots, cfg, variance,
               top, channels, soft_ref_correction):
    data_idx = np.asarray(cfg.resolved_data_indices)
    prediction = np.zeros((cfg.n_data, cfg.nt, cfg.nsc), dtype=np.complex128)
    fvar = frequency_noise_variance(variance, cfg)
    for i in range(cfg.n_data):
        H = channels[i][data_idx]
        y = received[cfg.n_ts + i].T[data_idx]
        ref_full = equalize_lmmse(received[cfg.n_ts + i], channels[i], fvar,
                                  gain_normalization=True)
        ref_full = ref_full + soft_ref_correction[i]
        ref = ref_full[:, data_idx].T
        top_idx = _slice_top_indices(ref, top)
        chosen = _joint_ml_subcarriers(H, y, top_idx)
        prediction[i, :, data_idx] = chosen
    return prediction


def detect(received_time, known_ts_frequency, known_pilots, cfg,
           noise_variance, model_seed):
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    variance = noise_variance
    bias = _qrc_ts_residual_bias(received, known_ts_frequency, known_pilots,
                                 cfg, variance, model_seed)
    channels = _lmmse_channels(received, known_ts_frequency, known_pilots,
                               cfg, variance)
    return _sliced_ml(received, known_ts_frequency, known_pilots, cfg,
                      variance, top=5, channels=channels,
                      soft_ref_correction=bias)
