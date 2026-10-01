"""Compare defence policies on identical simulated sessions.

Every session in every scenario is post-processed under three policies:

* **No defence** - always distil a key with standard (Shor-Preskill) privacy
  amplification based on the measured QBER.
* **QBER threshold** - the same, but abort whenever the measured QBER exceeds a
  fixed threshold (5% by default).
* **Adaptive (ML)** - the checkpointed :class:`~adaptive_qkd.protocol.AdaptiveQKDProtocol`.

Because the simulator knows the true channel, each released key can be scored
as safe or unsafe (see :class:`~adaptive_qkd.protocol.KeyOutcome`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adaptive_qkd.protocol import Action, AdaptiveQKDProtocol, finalize_key
from adaptive_qkd.simulator import BB84Simulator, ChannelParams

# Seeds well away from the range used for dataset generation, so benchmark
# sessions are never sessions the classifier was trained on.
BENCHMARK_SEED_OFFSET = 1_000_000

POLICIES: tuple[str, ...] = ("No defence", "QBER threshold", "Adaptive (ML)")


@dataclass(frozen=True)
class Scenario:
    name: str
    params: ChannelParams


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Clean channel", ChannelParams()),
    Scenario("Moderate noise (p=0.06)", ChannelParams(noise_prob=0.06)),
    Scenario("High noise (p=0.14)", ChannelParams(noise_prob=0.14)),
    Scenario("Intercept-resend (50%)", ChannelParams(intercept_prob=0.50)),
    Scenario("PNS attack (40%)", ChannelParams(pns_prob=0.40)),
    Scenario("Noise + intercept (p=0.06, 30%)", ChannelParams(noise_prob=0.06, intercept_prob=0.30)),
)


def run_benchmark(
    protocol: AdaptiveQKDProtocol,
    sessions_per_scenario: int = 200,
    num_qubits: int = 1000,
    qber_threshold: float = 0.05,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    seed: int = BENCHMARK_SEED_OFFSET,
) -> pd.DataFrame:
    """Return one row per (scenario, session, policy)."""
    if sessions_per_scenario < 1:
        raise ValueError("sessions_per_scenario must be positive")
    rows = []
    config = protocol.config
    for scenario_index, scenario in enumerate(scenarios):
        for i in range(sessions_per_scenario):
            session_seed = seed + scenario_index * sessions_per_scenario + i
            trace = BB84Simulator(num_qubits=num_qubits, seed=session_seed).simulate(scenario.params)
            qber = float(trace.error_mask[trace.test_mask].mean()) if trace.test_length else 0.5
            threshold_action = Action.ABORT if qber > qber_threshold else Action.CONTINUE
            outcomes = {
                "No defence": finalize_key(trace, Action.CONTINUE, config),
                "QBER threshold": finalize_key(trace, threshold_action, config),
                "Adaptive (ML)": protocol.run(trace).outcome,
            }
            for policy, outcome in outcomes.items():
                rows.append(
                    {
                        "scenario": scenario.name,
                        "attacked": scenario.params.is_attacked,
                        "session": i,
                        "qber": qber,
                        "policy": policy,
                        "action": outcome.action.name,
                        "key_rate": outcome.key_rate,
                        "secure_key_rate": outcome.secure_key_rate,
                        "unsafe": outcome.unsafe,
                    }
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per scenario and policy, preserving scenario and policy order."""
    summary = (
        results.assign(aborted=results["action"].eq("ABORT"), hardened=results["action"].eq("HARDEN"))
        .groupby(["scenario", "policy"], sort=False)
        .agg(
            mean_qber=("qber", "mean"),
            key_rate=("key_rate", "mean"),
            secure_key_rate=("secure_key_rate", "mean"),
            abort_rate=("aborted", "mean"),
            harden_rate=("hardened", "mean"),
            unsafe_rate=("unsafe", "mean"),
        )
        .reset_index()
    )
    return summary


def render_benchmark_markdown(summary: pd.DataFrame, sessions_per_scenario: int) -> str:
    lines = [
        "# Defence policy benchmark",
        "",
        f"{sessions_per_scenario} sessions of 1000 pulses per scenario. Key rates are in secret bits per",
        "transmitted pulse. *Unsafe* means the policy released more key than is secure for the true channel;",
        "*secure key rate* counts unsafe releases as zero.",
        "",
        "| Scenario | Mean QBER | Policy | Key rate | Secure key rate | Abort | Harden | Unsafe |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.mean_qber:.1%} | {row.policy} | {row.key_rate:.3f} | "
            f"{row.secure_key_rate:.3f} | {row.abort_rate:.0%} | {row.harden_rate:.0%} | {row.unsafe_rate:.0%} |"
        )
    lines += ["", "_Generated by `adaptive-qkd benchmark`._", ""]
    return "\n".join(lines)


def plot_benchmark(summary: pd.DataFrame, path: Path) -> None:
    """Grouped bar charts of secure key rate and unsafe-release rate per scenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"No defence": "#94a3b8", "QBER threshold": "#f59e0b", "Adaptive (ML)": "#0f766e"}
    scenarios = list(dict.fromkeys(summary["scenario"]))
    x = np.arange(len(scenarios))
    width = 0.8 / len(POLICIES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, column, title in (
        (axes[0], "secure_key_rate", "Secure key rate (bits / pulse, higher is better)"),
        (axes[1], "unsafe_rate", "Unsafe key releases (share of sessions, lower is better)"),
    ):
        for k, policy in enumerate(POLICIES):
            values = summary[summary["policy"] == policy].set_index("scenario").loc[scenarios, column]
            ax.bar(x + (k - 1) * width, values, width, label=policy, color=colors[policy])
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, scenarios, rotation=25, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
