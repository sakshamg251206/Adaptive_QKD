"""Altair chart builders for the dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd

from adaptive_qkd.config import CLASS_DISPLAY_NAMES, CLASS_NAMES
from adaptive_qkd.protocol import PolicyConfig, SessionResult

ACTION_COLORS = {"CONTINUE": "#16a34a", "HARDEN": "#d97706", "ABORT": "#dc2626"}
POLICY_COLORS = {"No defence": "#94a3b8", "QBER threshold": "#d97706", "Adaptive (ML)": "#0f766e"}
PRIMARY = "#0f766e"


def risk_timeline(result: SessionResult, config: PolicyConfig) -> alt.LayerChart:
    """P(attack) at each checkpoint over the CONTINUE / HARDEN / ABORT bands."""
    points = pd.DataFrame(
        {
            "Session received": [cp.progress for cp in result.checkpoints],
            "P(attack)": [cp.p_attack for cp in result.checkpoints],
            "Action": [cp.action.name for cp in result.checkpoints],
            "Checkpoint": [cp.index for cp in result.checkpoints],
        }
    )
    bands = pd.DataFrame(
        {
            "start": [0.0, config.harden_threshold, config.abort_threshold],
            "end": [config.harden_threshold, config.abort_threshold, 1.0],
            "Action": ["CONTINUE", "HARDEN", "ABORT"],
        }
    )
    color = alt.Color(
        "Action:N",
        scale=alt.Scale(domain=list(ACTION_COLORS), range=list(ACTION_COLORS.values())),
        legend=alt.Legend(orient="bottom", title=None),
    )
    x = alt.X("Session received:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5))
    band_layer = (
        alt.Chart(bands)
        .mark_rect(opacity=0.10)
        .encode(y=alt.Y("start:Q", scale=alt.Scale(domain=[0, 1])), y2="end:Q", color=color)
    )
    line = alt.Chart(points).mark_line(color="#334155", strokeWidth=2).encode(x=x, y="P(attack):Q")
    dots = (
        alt.Chart(points)
        .mark_circle(size=140, opacity=1)
        .encode(
            x=x,
            y=alt.Y("P(attack):Q", axis=alt.Axis(format="%"), title="P(attack)"),
            color=color,
            tooltip=[
                "Checkpoint",
                alt.Tooltip("Session received:Q", format=".0%"),
                alt.Tooltip("P(attack):Q", format=".1%"),
                "Action",
            ],
        )
    )
    return (band_layer + line + dots).properties(height=280)


def qber_windows(errors_by_window: list[float]) -> alt.Chart:
    data = pd.DataFrame({"Window": range(1, len(errors_by_window) + 1), "QBER": errors_by_window})
    return (
        alt.Chart(data)
        .mark_bar(color=PRIMARY, opacity=0.85, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("Window:O", title="Window of disclosed test bits (in transmission order)"),
            y=alt.Y(
                "QBER:Q",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, max(0.3, max(errors_by_window, default=0.0))]),
                title="Error rate",
            ),
            tooltip=["Window", alt.Tooltip("QBER:Q", format=".1%")],
        )
        .properties(height=240)
    )


def class_probabilities(class_probs: dict[str, float], true_label: str) -> alt.Chart:
    data = pd.DataFrame(
        {
            "Class": [CLASS_DISPLAY_NAMES[c] for c in CLASS_NAMES],
            "Probability": [class_probs.get(c, 0.0) for c in CLASS_NAMES],
            "Ground truth": ["Actual scenario" if c == true_label else "Other" for c in CLASS_NAMES],
        }
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("Class:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color(
                "Ground truth:N",
                scale=alt.Scale(domain=["Actual scenario", "Other"], range=[PRIMARY, "#cbd5e1"]),
                legend=None,
            ),
            tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=260)
    )


def benchmark_bars(summary: pd.DataFrame, column: str, title: str, percent: bool = False) -> alt.Chart:
    axis = alt.Axis(format="%") if percent else alt.Axis(format=".2f")
    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            y=alt.Y("scenario:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            yOffset=alt.YOffset("policy:N", sort=list(POLICY_COLORS)),
            x=alt.X(f"{column}:Q", title=title, axis=axis),
            color=alt.Color(
                "policy:N",
                scale=alt.Scale(domain=list(POLICY_COLORS), range=list(POLICY_COLORS.values())),
                legend=None,  # a shared HTML legend is rendered above the charts
            ),
            tooltip=[
                alt.Tooltip("scenario:N", title="Scenario"),
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip(f"{column}:Q", title=title, format=".1%" if percent else ".3f"),
            ],
        )
        .properties(height=360)
    )


def policy_legend_html() -> str:
    items = "".join(
        f'<span style="margin-right:1.25rem;white-space:nowrap">'
        f'<span style="display:inline-block;width:0.8rem;height:0.8rem;border-radius:2px;background:{color};'
        f'margin-right:0.4rem;vertical-align:-1px"></span>{policy}</span>'
        for policy, color in POLICY_COLORS.items()
    )
    return f'<div style="font-size:0.9rem;margin:0.25rem 0 0.75rem">{items}</div>'
