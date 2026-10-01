# Technical report

This document describes the models behind Adaptive QKD in more depth than the README: the channel physics,
the features, the classifiers, the key-rate formulas and the evaluation methodology. All numbers quoted
here are produced by `adaptive-qkd pipeline` and stored in [`results/`](../results/).

## 1. Problem

In BB84, Alice encodes random bits in one of two conjugate bases (Z: |0⟩/|1⟩, X: |+⟩/|−⟩). Bob measures in
a random basis, and after basis reconciliation (*sifting*) they disclose a random sample of the sifted bits
to estimate the quantum bit error rate *e*. Error correction and privacy amplification then distil a secret
key whose length depends on *e*.

Two practical difficulties motivate the project:

1. **Noise and attacks look alike.** Channel noise and intercept-resend both raise *e*. A fixed abort
   threshold either discards keys on noisy links or tolerates attacks.
2. **Some attacks are invisible to *e*.** With weak coherent pulses, a fraction of pulses contain several
   photons. A photon-number-splitting (PNS) eavesdropper keeps one photon of each multi-photon pulse and
   learns the bit after basis announcement, without introducing errors.

The project studies whether a classifier over several session statistics can separate these cases, and
whether a protocol driven by it releases safer keys than a QBER threshold.

## 2. Channel model

`simulator.py` models each pulse independently. The parameters are `ChannelParams(noise_prob, intercept_prob, pns_prob)`.

| Effect | Model | Induced QBER |
| --- | --- | --- |
| Depolarizing noise *p* | With probability *p* the qubit becomes I/2, so the outcome is uniformly random | *p*/2 |
| Intercept-resend *q* | Eve measures a fraction *q* in a random basis and re-sends her result in her basis | *q*/4 |
| PNS *s* | Eve blocks a fraction 0.25·*s* of pulses (detection loss) and learns a fraction *s* of detected pulses | 0 |

Noise and interception compose as independent bit flips: *e = a + b − 2ab*, which the tests check
against simulation.

**Qiskit backend.** `quantum.py` builds the same channel as a circuit, with one qubit per pulse. Alice applies
`x`/`h`; an intercepting Eve rotates into her basis, measures mid-circuit and rotates back, which re-prepares
the collapsed state in her basis; noise is an Aer `depolarizing_error` instruction; Bob rotates and measures.
The circuit contains only Clifford gates and Pauli noise, so Aer's stabilizer method simulates thousands of
qubits in one shot. PNS and timing jitter cannot be expressed as qubit gates and are applied classically
by both backends.

**Timing jitter.** Each session draws a scalar jitter value from |N(μ, σ)|, where μ and σ grow linearly
with *p*, *q* and *s*. This stands in for detector telemetry. It is a modelling assumption, and the
classifier relies on it heavily.

## 3. Features and dataset

Features are computed on the first fraction *f* of a session's pulses (`features.extract_features`):

- `qber`: the error rate on disclosed test bits within the prefix, in transmission order;
- `qber_std`, `qber_min`, `qber_max`: statistics of the error rate across windows of test bits. The number
  of windows scales with *f*, so each window holds a similar number of bits at every checkpoint;
- `sifting_ratio`: the share of prefix pulses that were detected and basis-matched;
- `basis_mismatch_rate`: the share of detected prefix pulses measured in different bases;
- `timing_jitter_ns`: the session's jitter value.

When no test bits exist yet, the QBER is set to 0.5, the most pessimistic value.

`dataset.py` simulates 2,000 sessions per class with randomised parameters and lengths of 800–1,200
pulses. The parameter ranges overlap on purpose:

| Class | Parameters |
| --- | --- |
| `clean` | none |
| `noise_only` | *p* ∈ [0.01, 0.15] |
| `intercept_resend` | *q* ∈ [0.10, 1.00] |
| `pns` | *s* ∈ [0.15, 0.90] |
| `noise+attack` | *p* ∈ [0.01, 0.12] plus either *q* ∈ [0.10, 0.80] (60%) or *s* ∈ [0.15, 0.80] (40%) |

Each session contributes one feature row per checkpoint (*f* ∈ {0.25, 0.5, 0.75, 1.0}) and one 20-step
QBER sequence for the LSTM.

## 4. Classifiers

Sessions, not rows, are split 80/20 with stratification, so no checkpoint of a test session appears in
training.

- **Random Forest:** 200 trees, depth ≤ 12, minimum leaf size 2, trained on all checkpoint rows of the
  training sessions. Trees are scale-invariant, so no standardisation is applied.
- **LSTM:** two layers with 64 hidden units over the standardised 20-step QBER sequence, followed by an
  MLP head. It is trained for 25 epochs with AdamW.

| Model | Accuracy | Macro F1 | Attack-detection ROC-AUC |
| --- | --- | --- | --- |
| Random Forest | 80.0% | 0.788 | 0.983 |
| LSTM (sequence only) | 51.6% | 0.388 | 0.720 |

The Random Forest on partial test sessions:

| Session observed | Accuracy | ROC-AUC |
| --- | --- | --- |
| 25% | 71.2% | 0.980 |
| 50% | 76.6% | 0.982 |
| 75% | 78.5% | 0.983 |
| 100% | 80.0% | 0.983 |

Observations:

- `pns` and `clean` are almost perfectly recognised (F1 ≈ 0.98 and 0.97). The main confusion is between
  `intercept_resend` and `noise+attack`, which overlap physically.
- In this simulator, QBER windows are exchangeable: attacks are not bursty, so the order of the windows
  carries no information. The LSTM therefore sees little more than the error-rate distribution. Because
  `clean` and `pns` both have zero errors, it cannot tell them apart.
- Feature importance (mean decrease in impurity) ranks `timing_jitter_ns` first, followed by `qber`,
  `sifting_ratio` and `qber_std`.

## 5. Key rates

`keyrate.py` implements the asymptotic GLLP bound for a fraction Δ of *tagged* pulses (pulses whose value
Eve may know) and error-correction inefficiency *f* (default 1.1):

```
r(e, Δ) = (1 − Δ) · [1 − h(e / (1 − Δ))] − f · h(e)
```

With Δ = 0 this is the Shor–Preskill rate *1 − h(e) − f·h(e)*, which reaches zero near *e* ≈ 10.6% when
*f* = 1.1. The final key length is *r* times the number of sifted bits left after disclosing the test sample.

| Action | Assumed Δ | Key |
| --- | --- | --- |
| `CONTINUE` | 0 | *n*·*r*(*e*, 0) |
| `HARDEN` | 0.5 (configurable) | *n*·*r*(*e*, 0.5) |
| `ABORT` | n/a | 0 |

**Safety scoring.** For each released key, the simulator also computes *n*·*r*(*e*, Δ_true) with
Δ_true = *s* (the true PNS fraction). A release longer than this is **unsafe**. Intercept-resend is
accounted for by *e* itself: in this asymptotic model, error-based privacy amplification removes Eve's
information, so by construction it never produces an unsafe release.

## 6. Adaptive protocol

`protocol.py` evaluates *K* evenly spaced checkpoints (default 4). At checkpoint *k* it computes features on
the first *k/K* of the session, obtains class probabilities, and sums the three attack classes into
P(attack). It then compares that value with the thresholds 0.25 (`HARDEN`) and 0.70 (`ABORT`).

`ABORT` ends the session immediately. Otherwise the action at the final checkpoint, which has the most
evidence, decides post-processing. During development, an earlier variant was trained on full sessions only
and made escalation monotonic. In a spot check of 100 moderate-noise sessions (*p* = 0.06), it hardened 57
and aborted 10 on the strength of noisy early checkpoints. The final design hardens 24% and aborts 0.5% of
the benchmark's moderate-noise sessions. The two samples differ, so treat this as indicative.

## 7. Benchmark

`benchmark.py` simulates 200 sessions of 1,000 pulses for each of six scenarios, using seeds disjoint from
the training data. It post-processes every session under all three policies:

| Scenario | Mean QBER | Secure key rate (none / threshold / adaptive) | Unsafe (none / threshold / adaptive) |
| --- | --- | --- | --- |
| Clean channel | 0.0% | 0.374 / 0.374 / 0.374 | 0% / 0% / 0% |
| Moderate noise (*p* = 0.06) | 2.9% | 0.229 / 0.218 / 0.188 | 0% / 0% / 0% |
| High noise (*p* = 0.14) | 6.9% | 0.097 / 0.041 / 0.017 | 0% / 0% / 0% |
| Intercept-resend (50%) | 12.4% | 0.011 / 0.001 / 0.000 | 0% / 0% / 0% |
| PNS attack (40%) | 0.0% | 0.000 / 0.000 / 0.000 | 100% / 100% / 0% |
| Noise + intercept (*p* = 0.06, 30%) | 10.2% | 0.029 / 0.006 / 0.002 | 0% / 0% / 0% |

The adaptive policy is the only one that never released an unsafe key. Its cost is a lower key rate on
noisy channels, where it hardens or aborts sessions that error-based privacy amplification alone would have
handled safely.

## 8. Threats to validity

- The PNS model and the timing-jitter signal are phenomenological. On real hardware, the separability of
  PNS from clean sessions would depend on what telemetry is available. Decoy-state protocols address PNS
  directly and would be the natural baseline to compare against.
- Key lengths are asymptotic. For sessions of about 1,000 pulses, finite-key effects would shorten all keys
  substantially.
- The classifier is trained and evaluated on the same simulator family. Distribution shift to real
  channels is untested.
