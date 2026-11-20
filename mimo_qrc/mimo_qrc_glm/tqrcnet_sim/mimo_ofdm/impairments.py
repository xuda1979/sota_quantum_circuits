"""Hardware impairments for the nonlinear-detection study.

`saleh_pa` applies the Saleh (1981) traveling-wave-tube AM/AM + AM/PM model
per transmit antenna to the time-domain waveform [Nsym, Nt, NTD]. The drive
level is set by an input backoff (IBO) relative to the PA's input saturation
amplitude; the output power is renormalized to the input power so the
impairment is pure nonlinear distortion (no net gain change), which keeps the
clean-frame Eb/N0 calibration meaningful.

`wiener_phase_process` / `apply_phase_noise` generate and apply a Wiener
receiver-oscillator phase process exp(j*phi[n]),
phi[n+1] = phi[n] + N(0, sigma_phi^2),
as a common LO broadcast across receiver antennas, applied to the noiseless
channel output before AWGN.
"""
from __future__ import annotations

import numpy as np

# Saleh (1981) TWT parameters (classic textbook values).
SALEH_ALPHA_A = 2.1587
SALEH_BETA_A = 1.1517
SALEH_ALPHA_P = 4.0033
SALEH_BETA_P = 9.1040


def saleh_input_saturation(beta_a: float = SALEH_BETA_A) -> float:
    """Input amplitude at which the AM/AM curve peaks: r_sat = 1/sqrt(beta_a)."""
    return 1.0 / np.sqrt(beta_a)


def saleh_pa(x_time: np.ndarray, ibo_db: float = 6.0,
             alpha_a: float = SALEH_ALPHA_A, beta_a: float = SALEH_BETA_A,
             alpha_p: float = SALEH_ALPHA_P, beta_p: float = SALEH_BETA_P,
             renormalize_power: bool = True) -> np.ndarray:
    """Saleh-model PA applied per transmit antenna.

    ibo_db: input backoff relative to the PA input saturation amplitude
    r_sat = 1/sqrt(beta_a): each antenna's drive is scaled so that
    10*log10(r_sat^2 / rms^2) = ibo_db before the nonlinearity.

    renormalize_power: rescale each antenna's output so its mean power equals
    the input's. The impairment is then pure distortion, and Eb/N0 computed
    on the clean (pre-PA) waveform remains an accurate power reference.
    """
    x = np.asarray(x_time)
    if x.ndim != 3:
        raise ValueError("x_time must be [Nsym, Nt, NTD]")
    out = np.empty(x.shape, dtype=np.complex128)
    r_sat = saleh_input_saturation(beta_a)
    backoff = 10.0 ** (float(ibo_db) / 20.0)
    for tx in range(x.shape[1]):
        sig = x[:, tx, :].astype(np.complex128)
        rms = float(np.sqrt(np.mean(np.abs(sig) ** 2)))
        if rms <= 0.0:
            out[:, tx, :] = sig
            continue
        drive = sig * (r_sat / (rms * backoff))
        r2 = np.abs(drive) ** 2
        amp = alpha_a * drive / (1.0 + beta_a * r2)
        phase = alpha_p * r2 / (1.0 + beta_p * r2)
        y = amp * np.exp(1j * phase)
        if renormalize_power:
            rms_out = float(np.sqrt(np.mean(np.abs(y) ** 2)))
            if rms_out > 0.0:
                y = y * (rms / rms_out)
        out[:, tx, :] = y
    return out


def wiener_phase_process(n_samples: int, sigma_phi: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Wiener phase: cumulative sum of N(0, sigma_phi^2) increments."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if sigma_phi < 0.0 or not np.isfinite(sigma_phi):
        raise ValueError("sigma_phi must be finite and non-negative")
    return np.cumsum(rng.normal(0.0, sigma_phi, size=int(n_samples)))


def apply_phase_noise(noiseless_received: np.ndarray, sigma_phi: float,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply common-LO Wiener phase noise to [Nsym, Nr, NTD].

    One phase process per frame (continuous over Nsym*NTD samples) is
    broadcast across receiver antennas. Returns (impaired_signal, phi) where
    phi has shape [Nsym*NTD]; with sigma_phi <= 0 the input is returned
    unchanged with a zero phase array.
    """
    y = np.asarray(noiseless_received)
    if y.ndim != 3:
        raise ValueError("noiseless_received must be [Nsym, Nr, NTD]")
    n_sym, n_td = y.shape[0], y.shape[2]
    if sigma_phi <= 0.0:
        return y.copy(), np.zeros(n_sym * n_td)
    phi = wiener_phase_process(n_sym * n_td, sigma_phi, rng)
    return y * np.exp(1j * phi.reshape(n_sym, 1, n_td)), phi
