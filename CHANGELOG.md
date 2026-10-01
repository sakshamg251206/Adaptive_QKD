# Changelog

## 2.0.0

A substantial overhaul of the original prototype.

### Fixed
- `noise+attack` sessions now honour their sampled parameters. Previously the PNS variant was silently
  replaced by a 30% intercept-resend attack, and clean/noise-only sessions overrode the requested noise level.
- QBER window sequences now follow transmission order; they were previously built from a shuffled sample.
- Key-rate accounting no longer counts the disclosed test bits as key material, and uses the
  Shor–Preskill / GLLP bounds instead of an ad hoc formula.
- Benchmark sessions no longer reuse seeds from the training set.
- The Qiskit "validation" (a single trivial circuit) is replaced by a real BB84 circuit backend.
- README and report figures now come from actual runs; earlier documents quoted unverified numbers.

### Changed
- Code moved into an installable `adaptive_qkd` package with a `adaptive-qkd` CLI.
- Features are length-independent rates computed on partial sessions, and the classifier is trained on
  every protocol checkpoint with a session-level train/test split.
- The protocol re-evaluates its decision at each checkpoint, and an abort is final.
- The benchmark scores every released key against the secure length for the true channel.
- The dashboard was redesigned: guided onboarding, scenario presets, tunable policy, explainer,
  benchmark and model tabs, and in-app training when no model exists.
- Generated data and models are no longer committed; `results/` holds reproducible reports.
- PyTorch and Qiskit are optional extras.

### Added
- A test suite (56 tests), Ruff, strict mypy, GitHub Actions CI, a Makefile, `.env.example` and a license.

## 1.0.0

- Initial BB84 simulator, dataset generator, Random Forest and LSTM classifiers, adaptive protocol
  and Streamlit dashboard.
