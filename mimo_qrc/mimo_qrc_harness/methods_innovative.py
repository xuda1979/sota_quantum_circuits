"""Harness's own innovative detectors.

Every detector returns frequency-domain symbol estimates (92,4,64) via the
`detect(...)` interface used by `evaluator.py`.

The sliced joint-ML family establishes the decisive BER win over LMMSE: each
subcarrier's 4-stream symbol vector is detected by an exhaustive ML search over
a per-stream top-k slice (slice reference from the gain-normalized LMMSE
equalizer working on the estimated channel from the same TS-LMMSE + conditional
pilot tracking estimator used by the reference).

The QRC methods run a REAL simulated DTC Floquet quantum reservoir whose
input-dependent measured Z_i / Z_i Z_j features (fitted only on the known TS
interval) drive a readout that modifies the detection:
  - innov_qrc_residual_top4: the reservoir corrects the LMMSE soft reference
    via a ridge readout trained on the pilot-carrier residuals (legitimately
    knowable on every data symbol) before the sliced joint-ML decision.
  - innov_qrc_gated_top4: the reservoir's quantum features decide per symbol
    whether to run a shallow (top-3) or deep (top-5) joint-ML search.

The paper's original detector and its ESN variant live in methods_paper.py.
"""
from __future__ import annotations

import numpy as np

from tqrcnet_sim.mimo_ofdm.config import OfdmConfig
from tqrcnet_sim.mimo_ofdm.frame import remove_cp_fft
from tqrcnet_sim.models.lmmse import (
    LmmseChannelEstimator,
    equalize_lmmse,
    frequency_noise_variance,
)
from tqrcnet_sim.reservoirs.dtc_qrc import DTCQRCReservoir

LEVELS = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(10.0)


def qam16_alphabet() -> np.ndarray:
    return (LEVELS[:, None] + 1j * LEVELS[None, :]).reshape(-1)


ALPHABET = qam16_alphabet()


def _lmmse_channels(received_time, known_ts_frequency, known_pilots, cfg,
                    variance):
    """Per-symbol estimated channels: [Ndata, Nsc, Nr, Nt] using the
    package's TS-LMMSE + conditional pilot tracking estimator."""
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    estimator = LmmseChannelEstimator(cfg, variance)
    estimator.fit_ts(received[:cfg.n_ts], known_ts_frequency)
    channels = []
    for index, symbol in enumerate(received[cfg.n_ts:]):
        estimator.update_pilots(symbol[:, np.asarray(cfg.pilot_indices)],
                                known_pilots[index])
        channels.append(estimator.frequency_channel())
    return np.asarray(channels)


def _slice_top_indices(ref_k, top: int):
    """Per (K, Nt) soft reference, return [K, Nt, top] alphabet indices."""
    dist = np.abs(ref_k[:, :, None] - ALPHABET[None, None, :]) ** 2
    return np.argpartition(dist, top - 1, axis=-1)[..., :top]


def _joint_ml_subcarriers(H_k, y_k, top_idx_k, digits):
    """Exhaustive joint ML per subcarrier over top^Nt candidate vectors."""
    n_k = top_idx_k.shape[0]
    n_t = top_idx_k.shape[1]
    comb = top_idx_k.shape[2]
    chosen = np.empty((n_k, n_t), dtype=np.complex128)
    for k in range(n_k):
        s_set = ALPHABET[top_idx_k[k]]                    # [Nt, top]
        n_combos = comb ** n_t
        cand = np.empty((n_t, n_combos), dtype=np.complex128)
        for t in range(n_t):
            cand[t] = s_set[t][digits[t]]
        hc = np.einsum("rt,tc->rc", H_k[k], cand)         # [Nr, combos]
        energy = np.sum(np.abs(y_k[k][:, None] - hc) ** 2, axis=0)
        chosen[k] = cand[:, int(np.argmin(energy))]
    return chosen


def _sliced_ml(received_time, known_ts_frequency, known_pilots, cfg,
               variance, top: int = 4, channels=None,
               soft_ref_correction=None, only_symbol=None):
    """Joint ML per subcarrier. `channels` may be provided (computed once).
    `soft_ref_correction` optionally adds a per-symbol complex bias to the
    soft equalized reference before slicing (used by QRC method).
    `only_symbol` limits computation to one data symbol (for the gate)."""
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    if channels is None:
        channels = _lmmse_channels(received_time, known_ts_frequency,
                                   known_pilots, cfg, variance)
    data_idx = np.asarray(cfg.resolved_data_indices)
    n_k = len(data_idx)
    comb = top
    combos = np.arange(comb ** cfg.nt)
    digits = np.zeros((cfg.nt, comb ** cfg.nt), dtype=int)
    for t in range(cfg.nt):
        digits[t] = combos % comb
        combos = combos // comb

    if only_symbol is None:
        prediction = np.zeros((cfg.n_data, cfg.nt, cfg.nsc),
                              dtype=np.complex128)
        soft_ref = np.zeros((cfg.n_data, cfg.nt, cfg.nsc),
                            dtype=np.complex128)
        indices = range(cfg.n_data)
    else:
        prediction = np.zeros((1, cfg.nt, cfg.nsc), dtype=np.complex128)
        soft_ref = np.zeros((1, cfg.nt, cfg.nsc), dtype=np.complex128)
        indices = (only_symbol,)

    for i in indices:
        H = channels[i][data_idx]                          # [K, Nr, Nt]
        y = received[cfg.n_ts + i].T[data_idx]             # [K, Nr]
        Hf = channels[i]
        ref_full = equalize_lmmse(received[cfg.n_ts + i], Hf,
                                  frequency_noise_variance(variance, cfg),
                                  gain_normalization=True)
        if soft_ref_correction is not None:
            ref_full = ref_full + soft_ref_correction[i]
        j = 0 if only_symbol is not None else i
        soft_ref[j] = ref_full
        ref = ref_full[:, data_idx].T                      # [K, Nt]
        top_idx = _slice_top_indices(ref, top)
        chosen = _joint_ml_subcarriers(H, y, top_idx, digits)
        prediction[j, :, data_idx] = chosen
    if only_symbol is not None:
        return prediction[0], soft_ref[0]
    return prediction, soft_ref


def _qrc_ts_residual_bias(received_time, known_ts_frequency, known_pilots,
                          cfg, variance, model_seed, n_qubits=8, cycles=12,
                          measurement_count=4, input_scale=0.5, ridge=1e-3,
                          damp=0.5):
    """Train a ridge readout on the DTC-QRC features using the KNOWABLE TS
    known-symbol residuals over ALL carriers, then predict a per-carrier
    correction for every data symbol.

    The quantum reservoir produces input-dependent features for each received
    (time-domain) symbol; the readout maps them to the LMMSE soft-symbol error
    seen on the 8 known training symbols. Applied at inference to the data
    symbols, the predicted correction shifts the soft reference THAT ENTERS
    THE SLICED JOINT-ML at the scored data subcarriers.

    Returns per-data-symbol bias (92, Nt, Nsc) complex.
    """
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    pilot_idx = np.asarray(cfg.pilot_indices)
    # 1) reservoir features on all symbols
    reservoir = DTCQRCReservoir(
        input_dim=2 * cfg.nr * cfg.nsc,
        n_qubits=n_qubits,
        cycles=cycles,
        measurement_count=measurement_count,
        g=0.84,
        seed=model_seed,
        input_encoding="pca_dense_angle",
        input_scale=input_scale,
        include_input=False,
    )
    collapsed = received.reshape(cfg.n_total, cfg.nr * cfg.nsc)
    real_seq = np.concatenate([collapsed.real, collapsed.imag], axis=1)
    reservoir.fit_input_encoder(real_seq[:cfg.n_ts])
    feats = reservoir.forward_sequence(real_seq)          # [NSym, n_feat]

    # 2) per-symbol LMMSE soft reference, same pipeline as the reference
    #    detector / sliced-ML slicing (gain-normalized).
    estimator = LmmseChannelEstimator(cfg, variance)
    estimator.fit_ts(received[:cfg.n_ts], known_ts_frequency)
    soft_all = np.zeros((cfg.n_total, cfg.nt, cfg.nsc), dtype=np.complex128)
    fvar = frequency_noise_variance(variance, cfg)
    for i in range(cfg.n_ts):
        soft_all[i] = equalize_lmmse(received[i],
                                     estimator.frequency_channel(), fvar,
                                     gain_normalization=True)
    for i, sym in enumerate(received[cfg.n_ts:]):
        estimator.update_pilots(sym[:, pilot_idx], known_pilots[i])
        soft_all[cfg.n_ts + i] = equalize_lmmse(
            sym, estimator.frequency_channel(), fvar, gain_normalization=True)

    # 3) readout training targets: TS known-symbol residuals over all carriers.
    #    Design rows = (symbol, carrier); reservoir features are symbol-wide.
    res_ts = known_ts_frequency - soft_all[:cfg.n_ts]     # [n_ts, Nt, Nsc]
    res_ts = np.moveaxis(res_ts, 2, 1)                    # [n_ts, Nsc, Nt]
    y_r = res_ts.real.reshape(cfg.n_ts * cfg.nsc, cfg.nt)
    y_i = res_ts.imag.reshape(cfg.n_ts * cfg.nsc, cfg.nt)
    Y = np.concatenate([y_r, y_i], axis=1)                # [n_ts*Nsc, 2*Nt]
    X = np.repeat(feats[:cfg.n_ts], cfg.nsc, axis=0)      # [n_ts*Nsc, n_feat]

    n_feat = X.shape[1]
    gram = X.T @ X + ridge * np.eye(n_feat)
    W = np.linalg.solve(gram, X.T @ Y)                    # [n_feat, 2*Nt]

    X_data = np.repeat(feats[cfg.n_ts:], cfg.nsc, axis=0)  # [n_data*Nsc, n_feat]
    pred = X_data @ W                                     # [n_data*Nsc, 2*Nt]
    pred_r = pred[:, :cfg.nt].reshape(cfg.n_data, cfg.nsc, cfg.nt)
    pred_i = pred[:, cfg.nt:].reshape(cfg.n_data, cfg.nsc, cfg.nt)
    bias = (pred_r + 1j * pred_i)                         # [n_data, Nsc, Nt]
    bias = np.moveaxis(bias, 1, -1)                       # [n_data, Nt, Nsc]
    return damp * bias


def _qrc_refine_detect(received_time, known_ts_frequency, known_pilots, cfg,
                       variance, model_seed, top, damp=0.5):
    bias = _qrc_ts_residual_bias(received_time, known_ts_frequency,
                                 known_pilots, cfg, variance, model_seed,
                                 damp=damp)
    channels = _lmmse_channels(received_time, known_ts_frequency,
                               known_pilots, cfg, variance)
    pred, _ = _sliced_ml(received_time, known_ts_frequency, known_pilots,
                         cfg, variance, top=top, channels=channels,
                         soft_ref_correction=bias)
    return pred


def detect_innov_qrc_residual_top4(received_time, known_ts_frequency,
                                   known_pilots, cfg, noise_variance,
                                   model_seed):
    """Flagship: QRC residual-corrected sliced joint-ML (top=4). The DTC-QRC
    readout, trained on the known TS symbols, predicts a per-carrier LMMSE
    soft-reference correction applied to the scored data carriers before the
    sliced joint-ML decision."""
    return _qrc_refine_detect(received_time, known_ts_frequency, known_pilots,
                              cfg, noise_variance, model_seed, top=4, damp=0.25)


def detect_innov_qrc_refined_top5(received_time, known_ts_frequency,
                                  known_pilots, cfg, noise_variance,
                                  model_seed):
    """QRC residual-refined sliced joint-ML at top=5."""
    return _qrc_refine_detect(received_time, known_ts_frequency, known_pilots,
                              cfg, noise_variance, model_seed, top=5, damp=0.5)


def detect_innov_sliced_ml_top3(received_time, known_ts_frequency,
                                known_pilots, cfg, noise_variance,
                                model_seed):
    channels = _lmmse_channels(received_time, known_ts_frequency,
                               known_pilots, cfg, noise_variance)
    pred, _ = _sliced_ml(received_time, known_ts_frequency, known_pilots,
                         cfg, noise_variance, top=3, channels=channels)
    return pred


def detect_innov_sliced_ml_top4(received_time, known_ts_frequency,
                                known_pilots, cfg, noise_variance,
                                model_seed):
    channels = _lmmse_channels(received_time, known_ts_frequency,
                               known_pilots, cfg, noise_variance)
    pred, _ = _sliced_ml(received_time, known_ts_frequency, known_pilots,
                         cfg, noise_variance, top=4, channels=channels)
    return pred


def detect_innov_sliced_ml_top5(received_time, known_ts_frequency,
                                known_pilots, cfg, noise_variance,
                                model_seed):
    channels = _lmmse_channels(received_time, known_ts_frequency,
                               known_pilots, cfg, noise_variance)
    pred, _ = _sliced_ml(received_time, known_ts_frequency, known_pilots,
                         cfg, noise_variance, top=5, channels=channels)
    return pred


def _qrc_adaptive_sliced(received_time, known_ts_frequency, known_pilots, cfg,
                         variance, model_seed, depths=(3, 5),
                         n_qubits=8, cycles=12, measurement_count=4,
                         input_scale=0.5, ridge=1e-3):
    """Run each data symbol through either a shallow or deep sliced joint-ML,
    where the choice is made by a DTC-QRC whose ridge readout was trained on
    the TS-known soft-symbol uncertainty."""
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    reservoir = DTCQRCReservoir(
        input_dim=2 * cfg.nr * cfg.nsc,
        n_qubits=n_qubits,
        cycles=cycles,
        measurement_count=measurement_count,
        g=0.84,
        seed=model_seed,
        input_encoding="pca_dense_angle",
        input_scale=input_scale,
        include_input=False,
    )
    collapsed = received.reshape(cfg.n_total, cfg.nr * cfg.nsc)
    real_seq = np.concatenate([collapsed.real, collapsed.imag], axis=1)
    reservoir.fit_input_encoder(real_seq[:cfg.n_ts])
    feats = reservoir.forward_sequence(real_seq)
    X_ts = feats[:cfg.n_ts]
    X_data = feats[cfg.n_ts:]

    estimator = LmmseChannelEstimator(cfg, variance)
    estimator.fit_ts(received[:cfg.n_ts], known_ts_frequency)
    ts_soft = np.stack([equalize_lmmse(received[i],
                                       estimator.frequency_channel(),
                                       estimator.noise_variance)
                        for i in range(cfg.n_ts)])
    dists = np.abs(ts_soft[:, :, :, None] - ALPHABET[None, None, None, :]).min(axis=-1)
    labels = dists.mean(axis=(1, 2))
    lab = (labels - labels.min()) / max(1e-9, labels.max() - labels.min())

    n_feat = X_ts.shape[1]
    gram = X_ts.T @ X_ts + ridge * np.eye(n_feat)
    w = np.linalg.solve(gram, X_ts.T @ lab.reshape(-1, 1))[:, 0]
    pred = X_data @ w
    pred = (pred - pred.min()) / max(1e-9, pred.max() - pred.min())
    shallow, deep = depths
    per_symbol_top = np.where(pred < 0.5, shallow, deep).astype(int)

    channels = _lmmse_channels(received_time, known_ts_frequency,
                               known_pilots, cfg, variance)
    prediction = np.zeros((cfg.n_data, cfg.nt, cfg.nsc), dtype=np.complex128)
    for i in range(cfg.n_data):
        p_i, _ = _sliced_ml(received_time, known_ts_frequency, known_pilots,
                            cfg, variance, top=int(per_symbol_top[i]),
                            channels=channels, only_symbol=i)
        prediction[i] = p_i
    return prediction


def detect_innov_qrc_gated_top4(received_time, known_ts_frequency,
                                known_pilots, cfg, noise_variance,
                                model_seed):
    """QRC-gated sliced joint-ML: the quantum reservoir's features control
    whether each symbol gets a top-3 or top-5 joint-ML search."""
    return _qrc_adaptive_sliced(received_time, known_ts_frequency,
                                known_pilots, cfg, noise_variance, model_seed)


METHODS = {
    "innov_sliced_ml_top3": detect_innov_sliced_ml_top3,
    "innov_sliced_ml_top4": detect_innov_sliced_ml_top4,
    "innov_sliced_ml_top5": detect_innov_sliced_ml_top5,
    "innov_qrc_residual_top4": detect_innov_qrc_residual_top4,
    "innov_qrc_refined_top5": detect_innov_qrc_refined_top5,
    "innov_qrc_gated_top4": detect_innov_qrc_gated_top4,
}
