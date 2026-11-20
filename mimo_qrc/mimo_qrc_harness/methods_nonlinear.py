"""Matched-pair detectors for the nonlinear (PA + phase-noise) scenario.

Architecture (v2): a bank of per-subcarrier temporal reservoir equalizers.
For subcarrier k, a reservoir processes the frame's 100-symbol temporal
sequence with input u_t = [Re/Im of the raw received vector y_{t,k} (Nr),
Re/Im of the gain-normalized LMMSE soft reference ref_{t,k} (Nt)], optionally
delay-embedded with `lags` past samples. A single ridge readout maps the
reservoir features to the soft-reference residual (truth - ref), trained on
the 880 legitimately knowable rows:

    8 TS symbols x 64 subcarriers (full-grid training sequences)  = 512 rows
    92 data symbols x 4 pilot subcarriers (tracked pilots)        = 368 rows

The predicted residual is added (damped) to the soft reference that feeds
either a nearest-neighbor decision (diagnostic: isolates reference quality)
or the same sliced joint-ML as the linear study (primary).

The ONLY variable between kind="qrc" and kind="esn" is the reservoir that
produces the features: the window-based DTC Floquet QRC (independent windows,
product-state re-preparation) vs a stateful classical ESN stepped
sequentially. Input conditioning, training rows, targets, readout, damping,
and the decision stage are shared code.

Design notes (v2.1):
- ONE shared reservoir instance is applied to every subcarrier's temporal
  sequence (same circuit / same ESN weights, one TS-pooled input encoder for
  the QRC). Features therefore live in a single shared basis, so a readout
  trained on the pilot subcarriers transfers to the data subcarriers.
- train_rows="all" uses the 880 knowable rows; train_rows="pilots" uses only
  the 368 data-symbol pilot rows, whose residual statistics match the data
  symbols (TS residuals are self-fitted by the channel estimator and are not
  representative).
- A classical common-phase-error (CPE) pilot de-rotation baseline (kind="cpe")
  isolates the part of any gain achievable by trivially tracking the phase
  noise common to all subcarriers.
"""
from __future__ import annotations

import numpy as np

import methods_innovative as base
from tqrcnet_sim.mimo_ofdm.frame import remove_cp_fft
from tqrcnet_sim.models.lmmse import (
    LmmseChannelEstimator,
    equalize_lmmse,
    frequency_noise_variance,
)
from tqrcnet_sim.reservoirs.dtc_qrc import DTCQRCReservoir
from tqrcnet_sim.reservoirs.esn import ESNReservoir


def _soft_reference_sequence(received, known_ts_frequency, known_pilots,
                             cfg, variance):
    """Per-symbol gain-normalized LMMSE soft reference [n_total, Nt, Nsc],
    identical pipeline to the reference detector / _sliced_ml internals."""
    pilot_idx = np.asarray(cfg.pilot_indices)
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
    return soft_all


def _subcarrier_inputs(received, soft_all, cfg, lags, include_ref):
    """Per-subcarrier delay-embedded inputs, TS-standardized.

    Returns U [n_total, nsc, D] real with
    D = (lags + 1) * 2 * (Nr + (Nt if include_ref else 0)).
    Standardization: per (k, dim) mean/std over the 8 TS symbols — the same
    statistics both reservoir kinds see, so input conditioning is matched.
    """
    u = np.moveaxis(np.concatenate([received.real, received.imag], axis=1),
                    1, -1)                                # [n_total, nsc, 2Nr]
    if include_ref:
        r = np.moveaxis(np.concatenate([soft_all.real, soft_all.imag], axis=1),
                        1, -1)                            # [n_total, nsc, 2Nt]
        u = np.concatenate([u, r], axis=-1)               # + 2Nt
    if lags > 0:
        lagged = [u]
        for lag in range(1, lags + 1):
            shifted = np.zeros_like(u)
            shifted[lag:] = u[:-lag]
            lagged.append(shifted)
        u = np.concatenate(lagged, axis=-1)
    mu = u[:cfg.n_ts].mean(axis=0, keepdims=True)
    sd = u[:cfg.n_ts].std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (u - mu) / sd


def _bank_features(kind, U, cfg, model_seed, n_qubits, cycles,
                   measurement_count, input_scale, esn_states,
                   esn_spectral_radius):
    """One SHARED reservoir applied to every subcarrier's temporal sequence
    -> features [n_total, nsc, n_feat].

    Sharing one circuit (QRC) / one weight matrix (ESN) across subcarriers
    keeps all features in a single basis, so a readout trained on the pilot
    subcarriers transfers to the data subcarriers. The QRC encoder PCA is
    fitted on the pooled TS rows of all subcarriers; the ESN state is reset
    at the start of each subcarrier's sequence (matching the QRC's
    independent-window, re-prepared-state operation)."""
    n_total, nsc, dim = U.shape
    if kind == "qrc":
        reservoir = DTCQRCReservoir(
            input_dim=dim,
            n_qubits=n_qubits,
            cycles=cycles,
            measurement_count=measurement_count,
            g=0.84,
            seed=model_seed,
            input_encoding="pca_dense_angle",
            input_scale=input_scale,
            include_input=False,
        )
        reservoir.fit_input_encoder(U[:cfg.n_ts].reshape(cfg.n_ts * nsc, dim))
        feats = np.stack([reservoir.forward_sequence(U[:, k])
                          for k in range(nsc)], axis=1)
        return feats
    if kind == "esn":
        esn = ESNReservoir(
            input_dim=dim,
            n_state=esn_states,
            spectral_radius=esn_spectral_radius,
            input_scale=input_scale,
            seed=model_seed,
        )
        feats = np.empty((n_total, nsc, esn.feature_dim), dtype=np.float64)
        for k in range(nsc):
            esn.reset_state()
            for t in range(n_total):
                feats[t, k] = esn.step(U[t, k])
        return feats
    raise ValueError(f"unknown reservoir kind: {kind}")


def _cpe_bias(soft_all, known_pilots, cfg, damp=1.0):
    """Classical common-phase-error (CPE) de-rotation from the tracked pilots.

    Wiener phase noise rotates every subcarrier of a symbol by approximately
    the same angle. Estimate that angle per data symbol from the pilot
    residuals and de-rotate the whole soft reference. This is the trivial
    classical competitor that any reservoir-based phase tracker must beat
    before a "quantum gain" can be claimed."""
    pilot_idx = np.asarray(cfg.pilot_indices)
    est = soft_all[cfg.n_ts:, :, pilot_idx]              # [n_data, Nt, P]
    cross = np.sum(np.conj(est) * known_pilots, axis=(1, 2))
    phi = -np.angle(cross)                               # [n_data]
    rot = np.exp(-1j * phi)[:, None, None]
    bias = soft_all[cfg.n_ts:] * (rot - 1.0)
    return damp * bias


def _residual_bias_bank(kind, received_time, known_ts_frequency,
                        known_pilots, cfg, variance, model_seed,
                        lags=1, include_ref=True, n_qubits=8, cycles=12,
                        measurement_count=4, input_scale=0.5, ridge=1e-3,
                        damp=0.5, esn_states=64, esn_spectral_radius=0.2,
                        train_rows="all"):
    """Reservoir-bank ridge readout -> damped soft-reference bias.

    train_rows: "all" = 880 rows (512 TS + 368 pilots); "pilots" = the 368
    data-symbol pilot rows only (residual statistics match the data symbols).

    Returns (bias [n_data, Nt, Nsc], soft_all [n_total, Nt, Nsc]).
    """
    received = remove_cp_fft(received_time, cfg.ncp, cfg.transform_normalization)
    soft_all = _soft_reference_sequence(received, known_ts_frequency,
                                        known_pilots, cfg, variance)
    if kind == "cpe":
        return _cpe_bias(soft_all, known_pilots, cfg, damp=damp), soft_all
    U = _subcarrier_inputs(received, soft_all, cfg, lags, include_ref)
    F = _bank_features(kind, U, cfg, model_seed, n_qubits, cycles,
                       measurement_count, input_scale, esn_states,
                       esn_spectral_radius)

    pilot_idx = np.asarray(cfg.pilot_indices)

    # Training rows: TS (t<8, all k) + data-symbol pilots (t>=8, pilot k).
    res_ts = known_ts_frequency - soft_all[:cfg.n_ts]      # [8, Nt, Nsc]
    Y_ts = np.moveaxis(res_ts, 1, -1).reshape(cfg.n_ts * cfg.nsc, cfg.nt)
    X_ts = F[:cfg.n_ts].reshape(cfg.n_ts * cfg.nsc, -1)

    res_p = known_pilots - soft_all[cfg.n_ts:, :, pilot_idx]  # [n_data, Nt, 4]
    Y_p = np.moveaxis(res_p, 1, -1).reshape(cfg.n_data * len(pilot_idx), cfg.nt)
    X_p = F[cfg.n_ts:, pilot_idx].reshape(cfg.n_data * len(pilot_idx), -1)

    if train_rows == "all":
        X = np.concatenate([X_ts, X_p], axis=0)
        Y = np.concatenate([Y_ts, Y_p], axis=0)            # [880, Nt]
    elif train_rows == "pilots":
        X, Y = X_p, Y_p                                    # [368, Nt]
    else:
        raise ValueError(f"unknown train_rows: {train_rows}")
    Y = np.concatenate([Y.real, Y.imag], axis=1)           # [rows, 2*Nt]
    # Constant-ones column: PA distortion biases the data-symbol channel
    # estimates, so the residual has a nonzero mean a bias-free readout
    # cannot represent.
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    n_feat = X.shape[1]
    gram = X.T @ X + ridge * np.eye(n_feat)
    W = np.linalg.solve(gram, X.T @ Y)

    X_data = F[cfg.n_ts:].reshape(cfg.n_data * cfg.nsc, -1)
    X_data = np.concatenate([X_data, np.ones((X_data.shape[0], 1))], axis=1)
    pred = X_data @ W                                     # [n_data*Nsc, 2Nt]
    pred_r = pred[:, :cfg.nt].reshape(cfg.n_data, cfg.nsc, cfg.nt)
    pred_i = pred[:, cfg.nt:].reshape(cfg.n_data, cfg.nsc, cfg.nt)
    bias = np.moveaxis(pred_r + 1j * pred_i, 1, -1)       # [n_data, Nt, Nsc]
    return damp * bias, soft_all


def _nearest_neighbor(ref, cfg):
    """Map soft reference [n_data, Nt, Nsc] to nearest 16-QAM symbols."""
    dist = np.abs(ref[:, :, :, None] - base.ALPHABET[None, None, None, :])
    return base.ALPHABET[np.argmin(dist, axis=-1)]


def _make_detector(kind, top, decision="sliced", **bias_kw):
    def detect(received_time, known_ts_frequency, known_pilots, cfg,
               noise_variance, model_seed):
        if kind == "none":
            bias = np.zeros((cfg.n_data, cfg.nt, cfg.nsc), dtype=np.complex128)
            received = remove_cp_fft(received_time, cfg.ncp,
                                     cfg.transform_normalization)
            soft_all = _soft_reference_sequence(received, known_ts_frequency,
                                                known_pilots, cfg,
                                                noise_variance)
        else:
            bias, soft_all = _residual_bias_bank(
                kind, received_time, known_ts_frequency, known_pilots,
                cfg, noise_variance, model_seed, **bias_kw)
        if decision == "nn":
            return _nearest_neighbor(soft_all[cfg.n_ts:] + bias, cfg)
        channels = base._lmmse_channels(received_time, known_ts_frequency,
                                        known_pilots, cfg, noise_variance)
        pred, _ = base._sliced_ml(received_time, known_ts_frequency,
                                  known_pilots, cfg, noise_variance, top=top,
                                  channels=channels,
                                  soft_ref_correction=bias)
        return pred
    detect.__name__ = f"detect_nl_{kind}_{decision}_top{top}"
    return detect


# --- nearest-neighbor diagnostics (reference quality, no ML search) --------
detect_nl_nn_none = _make_detector("none", 0, decision="nn")
detect_nl_nn_qrc = _make_detector("qrc", 0, decision="nn")
detect_nl_nn_esn = _make_detector("esn", 0, decision="nn")
detect_nl_nn_cpe = _make_detector("cpe", 0, decision="nn")

# --- primary matched set at top=4 --------------------------------------------
detect_nl_sliced_top4 = _make_detector("none", 4)
detect_nl_cpe_top4 = _make_detector("cpe", 4)
detect_nl_qrc_top4 = _make_detector("qrc", 4)
detect_nl_esn_top4 = _make_detector("esn", 4)

# --- pilots-only readout training (residual statistics match data) -----------
detect_nl_qrc_top4_ptrain = _make_detector("qrc", 4, train_rows="pilots")
detect_nl_esn_top4_ptrain = _make_detector("esn", 4, train_rows="pilots")

# --- robustness of the conclusion at top=5 ----------------------------------
detect_nl_sliced_top5 = _make_detector("none", 5)
detect_nl_qrc_top5 = _make_detector("qrc", 5)
detect_nl_esn_top5 = _make_detector("esn", 5)

# --- damping ablations -------------------------------------------------------
detect_nl_qrc_top4_damp025 = _make_detector("qrc", 4, damp=0.25)
detect_nl_qrc_top4_damp100 = _make_detector("qrc", 4, damp=1.0)
detect_nl_esn_top4_damp025 = _make_detector("esn", 4, damp=0.25)
detect_nl_esn_top4_damp100 = _make_detector("esn", 4, damp=1.0)

# --- QRC structure ablations -------------------------------------------------
detect_nl_qrc_top4_q6 = _make_detector("qrc", 4, n_qubits=6)
detect_nl_qrc_top4_q4 = _make_detector("qrc", 4, n_qubits=4)
detect_nl_qrc_top4_q2 = _make_detector("qrc", 4, n_qubits=2)
detect_nl_qrc_top4_mc2 = _make_detector("qrc", 4, measurement_count=2)
detect_nl_qrc_top4_cyc8 = _make_detector("qrc", 4, cycles=8)
detect_nl_qrc_top4_lags0 = _make_detector("qrc", 4, lags=0)
detect_nl_qrc_top4_noref = _make_detector("qrc", 4, include_ref=False)

# --- ESN sizing / input ablations (classical control) ------------------------
detect_nl_esn_top4_s32 = _make_detector("esn", 4, esn_states=32)
detect_nl_esn_top4_s128 = _make_detector("esn", 4, esn_states=128)
detect_nl_esn_top4_lags0 = _make_detector("esn", 4, lags=0)
detect_nl_esn_top4_noref = _make_detector("esn", 4, include_ref=False)


METHODS = {
    "nl_nn_none": detect_nl_nn_none,
    "nl_nn_qrc": detect_nl_nn_qrc,
    "nl_nn_esn": detect_nl_nn_esn,
    "nl_sliced_top4": detect_nl_sliced_top4,
    "nl_qrc_top4": detect_nl_qrc_top4,
    "nl_esn_top4": detect_nl_esn_top4,
    "nl_sliced_top5": detect_nl_sliced_top5,
    "nl_nn_cpe": detect_nl_nn_cpe,
    "nl_cpe_top4": detect_nl_cpe_top4,
    "nl_qrc_top4_ptrain": detect_nl_qrc_top4_ptrain,
    "nl_esn_top4_ptrain": detect_nl_esn_top4_ptrain,
    "nl_qrc_top5": detect_nl_qrc_top5,
    "nl_esn_top5": detect_nl_esn_top5,
    "nl_qrc_top4_damp025": detect_nl_qrc_top4_damp025,
    "nl_qrc_top4_damp100": detect_nl_qrc_top4_damp100,
    "nl_esn_top4_damp025": detect_nl_esn_top4_damp025,
    "nl_esn_top4_damp100": detect_nl_esn_top4_damp100,
    "nl_qrc_top4_q6": detect_nl_qrc_top4_q6,
    "nl_qrc_top4_q4": detect_nl_qrc_top4_q4,
    "nl_qrc_top4_q2": detect_nl_qrc_top4_q2,
    "nl_qrc_top4_mc2": detect_nl_qrc_top4_mc2,
    "nl_qrc_top4_cyc8": detect_nl_qrc_top4_cyc8,
    "nl_qrc_top4_lags0": detect_nl_qrc_top4_lags0,
    "nl_qrc_top4_noref": detect_nl_qrc_top4_noref,
    "nl_esn_top4_s32": detect_nl_esn_top4_s32,
    "nl_esn_top4_s128": detect_nl_esn_top4_s128,
    "nl_esn_top4_lags0": detect_nl_esn_top4_lags0,
    "nl_esn_top4_noref": detect_nl_esn_top4_noref,
}
