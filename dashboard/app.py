"""Streamlit dashboard for the adaptive QKD project.

Run from the repository root with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow running from a source checkout without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import charts  # noqa: E402

from adaptive_qkd.benchmark import DEFAULT_SCENARIOS, POLICIES, run_benchmark, summarize  # noqa: E402
from adaptive_qkd.config import CLASS_DISPLAY_NAMES, Paths  # noqa: E402
from adaptive_qkd.features import ordered_test_errors, window_error_rates  # noqa: E402
from adaptive_qkd.models import ThreatClassifier  # noqa: E402
from adaptive_qkd.protocol import Action, AdaptiveQKDProtocol, PolicyConfig, SessionResult  # noqa: E402
from adaptive_qkd.quantum import qiskit_available  # noqa: E402
from adaptive_qkd.simulator import BB84Simulator, ChannelParams  # noqa: E402

st.set_page_config(
    page_title="Adaptive QKD · Eavesdropping detection for BB84",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1280px; }
      .verdict { border-radius: 12px; padding: 1rem 1.25rem; border: 1px solid; margin-bottom: 0.5rem; }
      .verdict h3 { margin: 0 0 0.25rem 0; font-size: 1.25rem; }
      .verdict p { margin: 0; }
      .verdict-CONTINUE { background: rgba(22,163,74,0.08); border-color: rgba(22,163,74,0.45); }
      .verdict-HARDEN { background: rgba(217,119,6,0.08); border-color: rgba(217,119,6,0.45); }
      .verdict-ABORT { background: rgba(220,38,38,0.08); border-color: rgba(220,38,38,0.45); }
      .muted { opacity: 0.75; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PATHS = Paths.from_env()

PRESETS: dict[str, ChannelParams] = {
    "Clean channel": ChannelParams(),
    "Noisy fibre": ChannelParams(noise_prob=0.08),
    "Intercept-resend attack": ChannelParams(intercept_prob=0.5),
    "Photon-number-splitting attack": ChannelParams(pns_prob=0.4),
    "Noise + intercept-resend": ChannelParams(noise_prob=0.06, intercept_prob=0.3),
}

ACTION_COPY = {
    Action.CONTINUE: (
        "✅ Key accepted",
        "No sign of an eavesdropper. The key was distilled with standard privacy amplification.",
    ),
    Action.HARDEN: (
        "⚠️ Key hardened",
        "Some evidence of an eavesdropper. The key was distilled assuming part of it may already be known to Eve, "
        "which makes it shorter but safer.",
    ),
    Action.ABORT: ("⛔ Session aborted", "An eavesdropper is likely. The session was stopped and no key was produced."),
}


# --------------------------------------------------------------------------- data access


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _load_classifier(path: str, _mtime_key: float) -> ThreatClassifier | None:
    p = Path(path)
    return ThreatClassifier.load(p) if p.exists() else None


def load_classifier() -> tuple[ThreatClassifier | None, str | None]:
    try:
        return _load_classifier(str(PATHS.rf_model), _mtime(PATHS.rf_model)), None
    except Exception as exc:  # corrupt or incompatible model file
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _read_json(path: str, _mtime_key: float) -> dict[str, Any] | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def _read_csv(path: str, _mtime_key: float) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def train_classifier_now() -> None:
    from adaptive_qkd.dataset import generate_dataset, save_dataset
    from adaptive_qkd.training import train_models

    try:
        with st.status("Training the threat classifier…", expanded=True) as status:
            st.write("Simulating 10,000 labelled BB84 sessions…")
            dataset = generate_dataset()
            save_dataset(dataset, PATHS)
            st.write("Training the Random Forest…")
            train_models(PATHS, dataset=dataset, include_lstm=False, write_reports=False)
            status.update(label="Classifier trained", state="complete", expanded=False)
    except OSError as exc:
        st.error(
            f"Could not write model files to `{PATHS.model_dir}`: {exc}. Set ADAPTIVE_QKD_MODEL_DIR to a writable path."
        )
        return
    _load_classifier.clear()
    st.rerun()


# --------------------------------------------------------------------------- sidebar


def _apply_preset() -> None:
    params = PRESETS[st.session_state["preset"]]
    st.session_state["noise"] = params.noise_prob
    st.session_state["intercept"] = params.intercept_prob
    st.session_state["pns"] = params.pns_prob


def sidebar() -> tuple[ChannelParams, int, int, str, PolicyConfig, bool]:
    if "preset" not in st.session_state:
        st.session_state["preset"] = "Intercept-resend attack"
        _apply_preset()

    with st.sidebar:
        st.header("1 · Choose a channel")
        st.selectbox(
            "Scenario",
            list(PRESETS),
            key="preset",
            on_change=_apply_preset,
            help="Presets fill in the sliders below; adjust them freely.",
        )
        st.slider(
            "Channel noise (depolarizing probability)",
            0.0,
            0.20,
            step=0.01,
            key="noise",
            help="Natural noise in the fibre. Produces a bit error rate of about half this value.",
        )
        st.slider(
            "Intercept-resend fraction",
            0.0,
            1.0,
            step=0.05,
            key="intercept",
            help="Share of pulses Eve measures and re-sends. 100% interception causes ~25% errors.",
        )
        st.slider(
            "PNS attack strength",
            0.0,
            0.9,
            step=0.05,
            key="pns",
            help="Photon-number-splitting: Eve silently copies multi-photon pulses. Causes no extra errors.",
        )
        params = ChannelParams(
            noise_prob=st.session_state["noise"],
            intercept_prob=st.session_state["intercept"],
            pns_prob=st.session_state["pns"],
        )
        st.caption(f"Ground truth for this setting: **{CLASS_DISPLAY_NAMES[params.label]}**")

        with st.expander("Simulation settings"):
            num_qubits = st.slider("Pulses sent by Alice", 400, 3000, 1000, step=100)
            seed = int(
                st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=7,
                    step=1,
                    help="Same seed and settings → same session.",
                )
            )
            has_qiskit = qiskit_available()
            backend = st.radio(
                "Physics backend",
                ["numpy", "qiskit"],
                format_func=lambda b: "Fast (NumPy)" if b == "numpy" else "Quantum circuit (Qiskit Aer)",
                disabled=not has_qiskit,
                help=None if has_qiskit else "Install the 'quantum' extra to enable Qiskit.",
            )

        st.header("2 · Tune the defence")
        with st.expander("Policy settings", expanded=False):
            harden, abort = st.slider("Risk thresholds: HARDEN … ABORT", 0.0, 1.0, (0.25, 0.70), step=0.05)
            checkpoints = st.slider("Checkpoints per session", 1, 8, 4)
            tagged = st.slider("HARDEN assumes this share of the key is known to Eve", 0.0, 0.9, 0.5, step=0.05)
        config = PolicyConfig(
            harden_threshold=harden, abort_threshold=abort, checkpoints=checkpoints, harden_tagged_fraction=tagged
        )

        run = st.button("Run session", type="primary", icon="▶️", width="stretch")
        st.caption("Runs one BB84 session with these settings and lets the adaptive protocol react to it.")
    return params, num_qubits, seed, backend, config, run


# --------------------------------------------------------------------------- tabs


def render_session(result: SessionResult, config: PolicyConfig, uses_model: bool) -> None:
    trace, outcome = result.trace, result.outcome
    final = result.final_checkpoint
    title, body = ACTION_COPY[outcome.action]
    if outcome.action is Action.ABORT and final.index < config.checkpoints:
        body += (
            f" It stopped at checkpoint {final.index} of {config.checkpoints}, "
            f"after {final.progress:.0%} of the pulses."
        )
    st.markdown(
        f'<div class="verdict verdict-{outcome.action.name}"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )

    predicted = final.predicted_class
    truth = trace.label
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"**Actual scenario:** {CLASS_DISPLAY_NAMES[truth]}")
    with cols[1]:
        if uses_model:
            st.markdown(f"**Classifier's best guess:** {CLASS_DISPLAY_NAMES.get(predicted, predicted)}")
        else:
            st.markdown("**Classifier:** not trained (QBER heuristic)")
    with cols[2]:
        if outcome.released_bits == 0:
            st.badge("No key released", color="gray")
        elif outcome.unsafe:
            st.badge("Unsafe: Eve may know part of this key", icon="⚠️", color="red")
        else:
            st.badge("Released key is secure for the true channel", icon="✅", color="green")

    qber = float(trace.error_mask[trace.test_mask].mean()) if trace.test_length else 0.0
    m = st.columns(5)
    m[0].metric("Pulses sent", f"{trace.num_pulses:,}", border=True)
    m[1].metric(
        "Sifted bits", f"{trace.sifted_length:,}", help="Pulses where Alice's and Bob's bases matched.", border=True
    )
    m[2].metric("Measured QBER", f"{qber:.1%}", help="Error rate on the publicly compared test bits.", border=True)
    m[3].metric("P(attack)", f"{final.p_attack:.0%}", help="Classifier's probability that Eve is present.", border=True)
    m[4].metric(
        "Final key",
        f"{outcome.released_bits:,} bits",
        help="After error correction and privacy amplification.",
        border=True,
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Threat level at each checkpoint")
        st.altair_chart(charts.risk_timeline(result, config), width="stretch")
        st.caption("Shaded bands show which action each risk level triggers. An abort stops the session immediately.")
    with right:
        st.subheader("What the classifier sees")
        if final.class_probs:
            st.altair_chart(charts.class_probabilities(final.class_probs, truth), width="stretch")
            st.caption("The teal bar is the scenario that was actually simulated.")
        else:
            st.info("Train the classifier to see per-scenario probabilities.")

    st.subheader("Error rate across the session")
    windows = window_error_rates(ordered_test_errors(trace), 20).tolist()
    st.altair_chart(charts.qber_windows(windows), width="stretch")
    st.caption(
        "Intercept-resend raises the error rate; channel noise raises it less; a PNS attack leaves it untouched, "
        "which is why the classifier also looks at detection rate and timing."
    )

    with st.expander("Checkpoint log"):
        rows = [
            {
                "Checkpoint": cp.index,
                "Session received": cp.progress,
                "QBER so far": cp.features["qber"],
                "Sifting ratio": cp.features["sifting_ratio"],
                "Timing jitter (ns)": cp.features["timing_jitter_ns"],
                "Best guess": CLASS_DISPLAY_NAMES.get(cp.predicted_class, cp.predicted_class),
                "P(attack)": cp.p_attack,
                "Action": cp.action.name,
            }
            for cp in result.checkpoints
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Session received": st.column_config.NumberColumn(format="percent"),
                "QBER so far": st.column_config.NumberColumn(format="percent"),
                "Sifting ratio": st.column_config.NumberColumn(format="%.3f"),
                "Timing jitter (ns)": st.column_config.NumberColumn(format="%.2f"),
                "P(attack)": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
            },
        )


def tab_live(
    params: ChannelParams,
    num_qubits: int,
    seed: int,
    backend: str,
    config: PolicyConfig,
    run: bool,
    protocol: AdaptiveQKDProtocol,
) -> None:
    first_visit = "result" not in st.session_state
    if run or first_visit:
        try:
            with st.spinner("Simulating the session…"):
                trace = BB84Simulator(num_qubits=num_qubits, seed=seed, backend=backend).simulate(params)  # type: ignore[arg-type]
                st.session_state["result"] = (protocol.run(trace), config)
        except Exception as exc:
            st.error(f"The simulation failed: {exc}")
            return

    if first_visit and not run:
        st.info(
            "This is an example session. Pick another scenario in the sidebar and press **Run session**, "
            "or open **How it works** for a two-minute introduction.",
            icon="👋",
        )
    result, used_config = st.session_state["result"]
    render_session(result, used_config, protocol.uses_model)


def tab_how_it_works() -> None:
    st.markdown(
        """
### The problem
**Quantum key distribution (QKD)** lets two parties, Alice and Bob, grow a shared secret key whose secrecy is
protected by physics: measuring a quantum state disturbs it, so an eavesdropper (Eve) leaves traces.
In practice the traces are hard to read. Real fibres are noisy, so *some* errors are always present, and
certain attacks leave **no extra errors at all**.

The usual defence is a fixed rule: *abort if the error rate is above X%*. That rule throws away good keys on
noisy links and does nothing against attacks that do not raise the error rate.

### The approach
This project watches several statistical fingerprints of each session and lets a machine-learning
classifier decide how dangerous the session looks, then adapts the protocol.
"""
    )
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fillcolor="#f1f5f9", color="#94a3b8",
                fontname="Helvetica", fontsize=11];
          edge [color="#64748b"];
          A [label="Alice\\nrandom bits + bases"]; C [label="Quantum channel\\nnoise · Eve"];
          B [label="Bob\\nrandom bases"];
          S [label="Sifting +\\nerror estimation"]; F [label="Features\\nQBER, sifting, jitter"];
          M [label="Classifier\\nP(attack)"];
          D [label="Decision\\nCONTINUE · HARDEN · ABORT", fillcolor="#ccfbf1", color="#0f766e"];
          A -> C -> B -> S -> F -> M -> D;
        }
        """,
        width="stretch",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
#### Channel conditions
| Scenario | What happens | Error rate |
| --- | --- | --- |
| Clean | Nothing disturbs the qubits | 0% |
| Noise | Depolarizing noise in the fibre | ≈ half the noise level |
| Intercept-resend | Eve measures and re-sends pulses | up to 25% |
| Photon-number splitting | Eve copies multi-photon pulses and blocks some others | unchanged |
"""
        )
    with c2:
        st.markdown(
            """
#### Signals the classifier uses
- **QBER** and how it varies across the session (mean, spread, min, max)
- **Sifting ratio**: share of pulses that end up in the key (drops when Eve blocks pulses)
- **Basis-mismatch rate** among detected pulses
- **Timing jitter**: a *synthetic* detector side-channel standing in for hardware telemetry
"""
        )
    st.markdown(
        """
#### Decisions
At each checkpoint the protocol computes the features on the pulses received so far:
- **CONTINUE**: low risk, so standard privacy amplification is applied.
- **HARDEN**: medium risk, so privacy amplification assumes part of the key is already known to Eve
  (shorter key, safer).
- **ABORT**: high risk, so the session stops immediately.

Because this is a simulation, the true channel is known, so every released key can be checked against the length that
is *actually* secure (asymptotic GLLP / Shor-Preskill bounds). A release longer than that is counted as **unsafe**.

> **Scope.** The physics is a faithful qubit-level model of BB84, but photon statistics (for PNS) and timing jitter are
> simplified models, and key lengths use asymptotic bounds without finite-size corrections. Treat the results as a
> research prototype, not a security certification.
"""
    )


def tab_benchmark(protocol: AdaptiveQKDProtocol, config: PolicyConfig) -> None:
    st.markdown(
        "The same simulated sessions are post-processed by three policies: **No defence** (always distil a key), "
        "**QBER threshold** (abort above 5% errors) and **Adaptive (ML)** (this project). "
        "*Secure key rate* counts a key as zero if it was released unsafely."
    )
    saved = _read_csv(
        str(PATHS.results_dir / "benchmark_summary.csv"), _mtime(PATHS.results_dir / "benchmark_summary.csv")
    )

    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        sessions = c1.slider(
            "Sessions per scenario", 10, 200, 30, step=10, help="More sessions give smoother averages but take longer."
        )
        if c2.button("Run benchmark", icon="🔄", width="stretch"):
            with st.spinner(f"Simulating {sessions * len(DEFAULT_SCENARIOS)} sessions under {len(POLICIES)} policies…"):
                st.session_state["bench"] = summarize(run_benchmark(protocol, sessions_per_scenario=sessions))
        st.caption("Runs with the policy settings from the sidebar.")

    if "bench" in st.session_state:
        summary, source = st.session_state["bench"], "your latest run"
    elif saved is not None:
        summary, source = saved, "the saved run in `results/benchmark_summary.csv`"
    else:
        st.info("No benchmark results yet. Press **Run benchmark** to create them.")
        return
    if config != PolicyConfig() and "bench" not in st.session_state:
        st.caption("Saved results use the default policy settings.")

    st.markdown(f"Showing {source}.")
    st.markdown(charts.policy_legend_html(), unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Secure key rate** (bits per pulse, higher is better)")
        st.altair_chart(charts.benchmark_bars(summary, "secure_key_rate", "Secure key rate"), width="stretch")
    with c2:
        st.markdown("**Unsafe key releases** (share of sessions, lower is better)")
        st.altair_chart(charts.benchmark_bars(summary, "unsafe_rate", "Unsafe releases", percent=True), width="stretch")

    st.markdown(
        """
**How to read this**
- Against **intercept-resend** the error rate rises, so privacy amplification based on the measured QBER already removes
  Eve's information; all policies are safe there.
- Against **photon-number splitting** the error rate does not move. Both QBER-based policies release keys that Eve
  partly knows; the adaptive policy recognises the attack from detection-rate and timing signals and aborts.
- The price is caution on **noisy channels**: the classifier sometimes mistakes strong noise for noise plus a
  weak attack and hardens or aborts, lowering the key rate. The thresholds in the sidebar control this trade-off.
"""
    )
    with st.expander("Full table"):
        st.dataframe(
            summary,
            hide_index=True,
            column_config={
                "scenario": "Scenario",
                "policy": "Policy",
                "mean_qber": st.column_config.NumberColumn("Mean QBER", format="percent"),
                "key_rate": st.column_config.NumberColumn("Key rate", format="%.3f"),
                "secure_key_rate": st.column_config.NumberColumn("Secure key rate", format="%.3f"),
                "abort_rate": st.column_config.NumberColumn("Abort", format="percent"),
                "harden_rate": st.column_config.NumberColumn("Harden", format="percent"),
                "unsafe_rate": st.column_config.NumberColumn("Unsafe", format="percent"),
            },
        )


def tab_model() -> None:
    metrics = _read_json(str(PATHS.results_dir / "metrics.json"), _mtime(PATHS.results_dir / "metrics.json"))
    if metrics is None:
        st.info("No evaluation report found. Run `adaptive-qkd train` to create it.")
        return
    st.markdown(
        f"Evaluated on {metrics['test_size']:,} held-out sessions (trained on {metrics['train_size']:,}). "
        "*Attack detection* scores attacked vs. not-attacked sessions with ROC-AUC (1.0 is perfect, 0.5 is chance)."
    )
    cols = st.columns(len(metrics["models"]))
    for col, model in zip(cols, metrics["models"], strict=True):
        with col.container(border=True):
            st.markdown(f"**{model['name']}**")
            a, b = st.columns(2)
            a.metric("5-class accuracy", f"{model['accuracy']:.1%}")
            b.metric("Attack detection AUC", f"{model['binary_roc_auc']:.3f}")

    if metrics.get("by_progress"):
        st.markdown(
            "**Random Forest on partial sessions.** The protocol decides early, so accuracy with less data matters."
        )
        df = pd.DataFrame(
            [
                {"Session observed": float(p), "Accuracy": v["accuracy"], "Attack detection AUC": v["binary_roc_auc"]}
                for p, v in metrics["by_progress"].items()
            ]
        )
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "Session observed": st.column_config.NumberColumn(format="percent"),
                "Accuracy": st.column_config.NumberColumn(format="percent"),
                "Attack detection AUC": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    plots = [
        ("Confusion matrix (Random Forest)", "confusion_matrix_rf.png"),
        ("Attack detection ROC", "roc_curve.png"),
        ("Feature importance", "feature_importance.png"),
    ]
    available = [(t, PATHS.plots_dir / f) for t, f in plots if (PATHS.plots_dir / f).exists()]
    for col, (title, path) in zip(st.columns(len(available) or 1), available, strict=False):
        col.markdown(f"**{title}**")
        col.image(str(path), width="stretch")
    st.caption(
        "Intercept-resend and noise+attack are often confused because a noisy intercept-resend session *is* both. "
        "The LSTM sees only the error-rate sequence, so it cannot tell a clean channel from a PNS attack "
        "(both have zero errors); it is included as an ablation showing why the extra signals matter."
    )


# --------------------------------------------------------------------------- main


def main() -> None:
    st.title("Adaptive QKD")
    st.markdown(
        "#### Detecting eavesdroppers on a quantum key distribution link and adapting the protocol in response\n"
        '<p class="muted">Simulate a BB84 session, let a trained classifier estimate whether someone is listening, '
        "and see how an adaptive protocol keeps, hardens or aborts the key compared with a fixed error-rate rule.</p>",
        unsafe_allow_html=True,
    )

    params, num_qubits, seed, backend, config, run = sidebar()
    classifier, load_error = load_classifier()
    protocol = AdaptiveQKDProtocol(classifier, config)

    if load_error:
        st.error(f"The saved model could not be loaded ({load_error}). Retrain it below.")
    if classifier is None:
        with st.container(border=True):
            st.markdown(
                "**The threat classifier has not been trained yet.** Sessions still run, but decisions use a simple "
                "error-rate heuristic. Training takes about 15 seconds."
            )
            if st.button("Train classifier now", type="primary", icon="🧠"):
                train_classifier_now()

    live, how, bench, model = st.tabs(["Live session", "How it works", "Benchmark", "Model performance"])
    with live:
        tab_live(params, num_qubits, seed, backend, config, run, protocol)
    with how:
        tab_how_it_works()
    with bench:
        tab_benchmark(protocol, config)
    with model:
        tab_model()


main()
