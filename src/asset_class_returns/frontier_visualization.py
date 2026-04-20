from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import seaborn as sns

from .convex_continuous_efficient_frontier import FrontierPoint, FrontierRun

_MPLCONFIGDIR = Path(
    os.environ.get("MPLCONFIGDIR", "/tmp/asset_class_returns_mplconfig")
)
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "continuous_frontier_runs"
PLOT_TITLE = "Continuous Convex Efficient Frontier"
PLOT_SUBTITLE = (
    "Long-only, cash allowed, inflation excluded, real estate and gold capped at 7.5%"
)
COMPARISON_PLOT_TITLE = "Efficient Frontier Comparison"


def create_run_output_dir(base_dir: Path | None = None) -> Path:
    """Create a timestamped directory for one optimization run."""
    root = DEFAULT_OUTPUT_ROOT if base_dir is None else base_dir
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / timestamp
    suffix = 1
    while output_dir.exists():
        output_dir = root / f"{timestamp}_{suffix:02d}"
        suffix += 1

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_frontier_summary_text(run: FrontierRun, output_dir: Path) -> str:
    """Build a compact human-readable summary for the run."""
    return build_multi_frontier_summary_text(
        sections=[(PLOT_TITLE, run, None)],
        output_dir=output_dir,
        artifacts=[
            ("Output directory", output_dir),
            ("Summary", output_dir / "frontier_summary.txt"),
            ("Full CSV", output_dir / "frontier_full.csv"),
            ("Static plot", output_dir / "efficient_frontier.png"),
            ("Interactive plot", output_dir / "efficient_frontier.html"),
        ],
    )


def build_multi_frontier_summary_text(
    sections: Sequence[tuple[str, FrontierRun | None, str | None]],
    output_dir: Path,
    artifacts: Sequence[tuple[str, Path]] | None = None,
) -> str:
    """Build a compact human-readable summary for one or more frontier runs."""
    lines = [COMPARISON_PLOT_TITLE]

    for section_index, (label, run, error_message) in enumerate(sections):
        lines.extend(
            [
                "",
                "==========================",
                f"Method: {label}",
                "==========================",
            ]
        )

        if run is None:
            lines.append("No optimization run was produced.")
            if error_message:
                lines.append(f"Error: {error_message}")
        else:
            lines.extend(_build_run_section_lines(run))
            if error_message:
                lines.append(f"Note: {error_message}")

        if section_index == len(sections) - 1:
            lines.append("")

    artifact_items = (
        [
            ("Output directory", output_dir),
            ("Summary", output_dir / "frontier_summary.txt"),
        ]
        if artifacts is None
        else list(artifacts)
    )
    lines.append("Artifacts:")
    for artifact_label, artifact_path in artifact_items:
        lines.append(f"  {artifact_label}: {artifact_path}")

    return "\n".join(lines)


def save_frontier_summary(summary_text: str, output_dir: Path) -> Path:
    """Save the run summary text file."""
    summary_path = output_dir / "frontier_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    return summary_path


def plot_frontier_static(
    run: FrontierRun,
    output_path: Path,
    title: str = PLOT_TITLE,
    subtitle: str = PLOT_SUBTITLE,
    method_label: str | None = None,
) -> Path:
    """Save a static Matplotlib/Seaborn frontier chart."""
    solved_points = _solved_points(run)
    if not solved_points:
        raise ValueError("No solved frontier points are available to plot.")

    sns.set_theme(style="whitegrid", context="talk")
    volatility = np.array([point.volatility for point in solved_points], dtype=float)
    achieved_return = np.array(
        [point.achieved_return for point in solved_points],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#fbfaf7")
    ax.set_facecolor("#fbfaf7")

    ax.plot(
        volatility,
        achieved_return,
        color="#1f4e79",
        linewidth=2.8,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.3,
        markeredgecolor="#1f4e79",
        zorder=2,
    )

    start_point = solved_points[0]
    end_point = solved_points[-1]
    ax.scatter(
        start_point.volatility,
        start_point.achieved_return,
        s=110,
        color="#2e8b57",
        edgecolor="white",
        linewidth=1.4,
        zorder=4,
    )
    ax.scatter(
        end_point.volatility,
        end_point.achieved_return,
        s=110,
        color="#b03a2e",
        edgecolor="white",
        linewidth=1.4,
        zorder=4,
    )

    ax.annotate(
        f"Start: {_format_pct(start_point.target_return)} target",
        xy=(start_point.volatility, start_point.achieved_return),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        color="#2e8b57",
    )
    ax.annotate(
        f"End: {_format_pct(end_point.target_return)} target",
        xy=(end_point.volatility, end_point.achieved_return),
        xytext=(-110, 12),
        textcoords="offset points",
        fontsize=10,
        color="#b03a2e",
    )

    ax.set_xlabel("Volatility (standard deviation)")
    ax.set_ylabel("Expected return")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(color="#d8d8d8", alpha=0.7, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.06, y=0.08)

    fig.suptitle(
        title,
        x=0.125,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.125,
        0.925,
        subtitle,
        ha="left",
        fontsize=11,
        color="#4f4f4f",
    )

    method_text = "" if method_label is None else f"Method: {method_label}\n"
    annotation_text = (
        f"{method_text}Solver: {run.solver}\n"
        f"Solved targets: {len(solved_points)}/{len(run.points)}\n"
        f"Total runtime: {run.total_solve_time_seconds:.3f}s"
    )
    ax.text(
        0.02,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#d5d5d5",
            "alpha": 0.95,
        },
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return output_path


def plot_frontier_interactive(run: FrontierRun, output_path: Path) -> Path:
    """Save an interactive Plotly frontier chart."""
    solved_points = _solved_points(run)
    if not solved_points:
        raise ValueError("No solved frontier points are available to plot.")

    volatility = np.array([point.volatility for point in solved_points], dtype=float)
    achieved_return = np.array(
        [point.achieved_return for point in solved_points],
        dtype=float,
    )
    customdata = np.column_stack(
        [
            np.array([point.target_return for point in solved_points], dtype=float),
            np.array([point.invested_weight for point in solved_points], dtype=float),
            np.array([point.holdings_count for point in solved_points], dtype=float),
            1000.0
            * np.array(
                [point.solve_time_seconds for point in solved_points], dtype=float
            ),
        ]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=volatility,
            y=achieved_return,
            mode="lines+markers",
            name="Frontier",
            customdata=customdata,
            line={"color": "#1f4e79", "width": 3},
            marker={
                "size": 8,
                "color": "#1f4e79",
                "line": {"color": "white", "width": 1},
            },
            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Achieved return: %{y:.2%}<br>"
                "Target return: %{customdata[0]:.2%}<br>"
                "Invested weight: %{customdata[1]:.2%}<br>"
                "Holdings count: %{customdata[2]:.0f}<br>"
                "Solve time: %{customdata[3]:.3f} ms"
                "<extra></extra>"
            ),
        )
    )

    start_point = solved_points[0]
    end_point = solved_points[-1]
    figure.add_trace(
        go.Scatter(
            x=[start_point.volatility],
            y=[start_point.achieved_return],
            mode="markers",
            name="Start target",
            marker={
                "size": 12,
                "color": "#2e8b57",
                "line": {"color": "white", "width": 1.5},
            },
            hovertemplate=(
                f"Start target: {_format_pct(start_point.target_return)}<br>"
                f"Volatility: {_format_pct(start_point.volatility)}<br>"
                f"Achieved return: {_format_pct(start_point.achieved_return)}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[end_point.volatility],
            y=[end_point.achieved_return],
            mode="markers",
            name="End target",
            marker={
                "size": 12,
                "color": "#b03a2e",
                "line": {"color": "white", "width": 1.5},
            },
            hovertemplate=(
                f"End target: {_format_pct(end_point.target_return)}<br>"
                f"Volatility: {_format_pct(end_point.volatility)}<br>"
                f"Achieved return: {_format_pct(end_point.achieved_return)}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        template="plotly_white",
        width=980,
        height=620,
        hovermode="closest",
        title={
            "text": (
                f"{PLOT_TITLE}<br>"
                f"<sup>{PLOT_SUBTITLE} | Solver: {run.solver} | "
                f"Total runtime: {run.total_solve_time_seconds:.3f}s</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        margin={"l": 70, "r": 30, "t": 110, "b": 70},
    )
    figure.update_xaxes(title="Volatility (standard deviation)", tickformat=".1%")
    figure.update_yaxes(title="Expected return", tickformat=".1%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def plot_frontier_interactive_comparison(
    runs: Sequence[tuple[str, FrontierRun]],
    output_path: Path,
) -> Path:
    """Save one interactive Plotly chart containing multiple frontier traces."""
    color_cycle = ["#1f4e79", "#b03a2e", "#2e8b57", "#8e44ad", "#c97900"]
    figure = go.Figure()
    plotted_count = 0
    method_summaries: list[str] = []

    for index, (label, run) in enumerate(runs):
        solved_points = _solved_points(run)
        if not solved_points:
            continue

        plotted_count += 1
        color = color_cycle[index % len(color_cycle)]
        volatility = np.array(
            [point.volatility for point in solved_points], dtype=float
        )
        achieved_return = np.array(
            [point.achieved_return for point in solved_points],
            dtype=float,
        )
        customdata = np.column_stack(
            [
                np.array([point.target_return for point in solved_points], dtype=float),
                np.array(
                    [point.invested_weight for point in solved_points], dtype=float
                ),
                np.array(
                    [point.holdings_count for point in solved_points], dtype=float
                ),
                1000.0
                * np.array(
                    [point.solve_time_seconds for point in solved_points],
                    dtype=float,
                ),
            ]
        )

        figure.add_trace(
            go.Scatter(
                x=volatility,
                y=achieved_return,
                mode="lines+markers",
                name=f"{label} frontier",
                customdata=customdata,
                line={"color": color, "width": 3},
                marker={
                    "size": 8,
                    "color": color,
                    "line": {"color": "white", "width": 1},
                },
                hovertemplate=(
                    "Method: "
                    + label
                    + "<br>"
                    + "Volatility: %{x:.2%}<br>"
                    + "Achieved return: %{y:.2%}<br>"
                    + "Target return: %{customdata[0]:.2%}<br>"
                    + "Invested weight: %{customdata[1]:.2%}<br>"
                    + "Holdings count: %{customdata[2]:.0f}<br>"
                    + "Solve time: %{customdata[3]:.3f} ms"
                    + "<extra></extra>"
                ),
            )
        )

        method_summaries.append(
            f"{label} ({len(solved_points)}/{len(run.points)} targets)"
        )

    if plotted_count == 0:
        raise ValueError("No solved frontier points are available to plot.")

    figure.update_layout(
        template="plotly_white",
        width=980,
        height=620,
        hovermode="closest",
        title={
            "text": (
                f"{COMPARISON_PLOT_TITLE}<br><sup>{' | '.join(method_summaries)}</sup>"
            ),
            "x": 0.01,
            "xanchor": "left",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        margin={"l": 70, "r": 30, "t": 110, "b": 70},
    )
    figure.update_xaxes(title="Volatility (standard deviation)", tickformat=".1%")
    figure.update_yaxes(title="Expected return", tickformat=".1%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


def _solved_points(run: FrontierRun) -> list[FrontierPoint]:
    return [
        point
        for point in run.points
        if point.weights is not None
        and point.achieved_return is not None
        and point.volatility is not None
        and point.invested_weight is not None
        and point.holdings_count is not None
    ]


def _build_run_section_lines(run: FrontierRun) -> list[str]:
    solved_points = _solved_points(run)
    solved_count = len(solved_points)
    total_count = len(run.points)

    lines = [
        f"Solver: {run.solver}",
        f"Solved targets: {solved_count}/{total_count}",
        f"Total solve time (s): {run.total_solve_time_seconds:.6f}",
    ]

    if solved_points:
        average_solve_time_ms = 1000.0 * mean(
            point.solve_time_seconds for point in solved_points
        )
        first_point = solved_points[0]
        last_point = solved_points[-1]
        volatility_min = min(point.volatility for point in solved_points)
        volatility_max = max(point.volatility for point in solved_points)
        achieved_min = min(point.achieved_return for point in solved_points)
        achieved_max = max(point.achieved_return for point in solved_points)

        lines.extend(
            [
                f"Average solve time per target (ms): {average_solve_time_ms:.3f}",
                f"Volatility range: {_format_pct(volatility_min)} to {_format_pct(volatility_max)}",
                f"Achieved return range: {_format_pct(achieved_min)} to {_format_pct(achieved_max)}",
                "",
                "First frontier point:",
                _format_point_summary(first_point),
                "Last frontier point:",
                _format_point_summary(last_point),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No frontier points were solved successfully, so plots may be unavailable.",
            ]
        )

    return lines


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _format_point_summary(point: FrontierPoint) -> str:
    return (
        f"  target={_format_pct(point.target_return)}, "
        f"achieved={_format_pct(point.achieved_return)}, "
        f"volatility={_format_pct(point.volatility)}, "
        f"invested={_format_pct(point.invested_weight)}, "
        f"holdings={point.holdings_count}"
    )
