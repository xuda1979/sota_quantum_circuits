# QRC MIMO-OFDM Detector Harness

Goal: find a quantum reservoir pool (QRC) MIMO detector whose payload BER at
each of the seven Eb/N0 points (0,5,10,15,20,25,30 dB) is STRICTLY below the
LMMSE baseline (`tqrcnet_sim.models.lmmse.detect_estimated_csi` with joint
TS-LMMSE + conditional pilot tracking).

Structure:
- `candidates/` — detector programs, each defines
  `detect(received_time, known_ts_frequency, known_pilots, cfg,
  noise_variance, model_seed) -> (92,4,64) frequency-domain symbol estimates`.
- `methods_paper.py` — the paper's original method (DTC-QRC / TRCNet wrapper),
  keeping the exact `initial_program.py` pipeline as a named method plus
  configuration points the paper defines (ESN backend, layer count, window,
  QRC qubits/cycles/measurements).
- `methods_innovative.py` — harness's own novel detector methods. All methods
  use a REAL simulated quantum reservoir (density/statevector simulation of a
  DTC Floquet system) producing input-dependent measured features that
  contribute to payload prediction; classical preprocessing is allowed only
  when the quantum contribution is part of the detector.
- `run_method.py` — CLI to evaluate one method on dev or heldout split,
  comparing against the LMMSE reference at all seven points.
- `search.py` — deterministic local search over the method/settings space,
  using the same scoring as the evaluator (seven points, strict wins).
- `score_report.py` — shared helpers: score candidate files via the package
  evaluator, summarize per-SNR BER vs LMMSE.

The reference numbers are matched exactly with `evaluator.py` from the source
package: a single development frame gives LMMSE seven-point error counts equal
to the historical record; the initial DTC-QRC is included as the `paper` method.

## Evaluation口径 (same as source package)

- Fixed FFT64/CP16, 8 TS + 92 data symbols, 16-QAM, 20 MHz, EPA 20 Hz,
  MATLAB FIR. 70,656 payload bits per frame per point.
- QRC and LMMSE share receive waveform, known TS, pilots and noise variance.
- Seven points share one frame and one noise realization; only noise strength
  changes.
- A point is won iff `qrc_errors < lmmse_errors` (denominator equal).
  `target_met=1` iff all seven points are strict wins.
