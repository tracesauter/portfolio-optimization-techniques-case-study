from pathlib import Path
from time import perf_counter

from . import convex_continuous_efficient_frontier as continuous
from . import mip_efficient_frontier_ideal_solution as mip
from .convex_continuous_efficient_frontier import (
    FrontierPoint as ContinuousFrontierPoint,
    FrontierRun,
)
from .frontier_visualization import (
    build_multi_frontier_summary_text,
    create_run_output_dir,
    plot_frontier_interactive_comparison,
    plot_frontier_static,
    save_frontier_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "continuous_frontier_runs"


def _to_continuous_points(
    points: list[mip.FrontierPoint],
    solve_time_seconds: float,
) -> list[ContinuousFrontierPoint]:
    """Adapt MIP frontier points to the shared plotting/summary point schema."""
    return [
        ContinuousFrontierPoint(
            target_return=point.target_return,
            status=point.status,
            achieved_return=point.achieved_return,
            variance=point.variance,
            volatility=point.volatility,
            invested_weight=point.invested_weight,
            holdings_count=point.holdings_count,
            solve_time_seconds=solve_time_seconds,
            weights=None if point.weights is None else point.weights.copy(),
        )
        for point in points
    ]


def main() -> None:
    asset_names, expected_returns, covariance = continuous.prepare_project_inputs()
    continuous_run = continuous.solve_efficient_frontier(
        expected_returns=expected_returns,
        covariance=covariance,
        asset_names=asset_names,
    )
    continuous_frontier_frame = continuous.frontier_to_frame(continuous_run)

    mip_run: FrontierRun | None = None
    mip_run_error: str | None = None
    mip_frontier_frame = None
    mip_plot_error: str | None = None

    try:
        mip_start_time = perf_counter()
        mip_points = mip.solve_efficient_frontier(
            expected_returns=expected_returns,
            covariance=covariance,
        )
        mip_elapsed_seconds = perf_counter() - mip_start_time
        mip_point_solve_seconds = (
            mip_elapsed_seconds / len(mip_points) if mip_points else 0.0
        )
        mip_frontier_frame = mip.frontier_to_frame(mip_points, asset_names=asset_names)
        mip_run = FrontierRun(
            solver="SCIP",
            total_solve_time_seconds=mip_elapsed_seconds,
            asset_names=list(asset_names),
            points=_to_continuous_points(mip_points, mip_point_solve_seconds),
        )
    except Exception as exc:
        mip_run_error = str(exc)

    output_dir = create_run_output_dir(OUTPUTS_ROOT)
    continuous_csv_path = output_dir / "frontier_full_continuous.csv"
    mip_csv_path = output_dir / "frontier_full_mip.csv"
    continuous_png_path = output_dir / "efficient_frontier_continuous.png"
    mip_png_path = output_dir / "efficient_frontier_mip.png"
    comparison_html_path = output_dir / "efficient_frontier_comparison.html"

    continuous_frontier_frame.to_csv(continuous_csv_path, index=False)
    if mip_frontier_frame is not None:
        mip_frontier_frame.to_csv(mip_csv_path, index=False)

    plot_frontier_static(
        continuous_run,
        continuous_png_path,
        method_label="Continuous Convex",
    )
    if mip_run is not None:
        try:
            plot_frontier_static(
                mip_run,
                mip_png_path,
                title="Mixed Integer Efficient Frontier",
                subtitle="Long-only with cardinality and lot-size constraints",
                method_label="Mixed Integer (Discrete Constraints)",
            )
        except ValueError as exc:
            mip_plot_error = str(exc)

    comparison_runs: list[tuple[str, FrontierRun]] = [
        ("Continuous Convex", continuous_run),
    ]
    if mip_run is not None:
        comparison_runs.append(("Mixed Integer (Discrete Constraints)", mip_run))
    plot_frontier_interactive_comparison(comparison_runs, comparison_html_path)

    mip_notes = [message for message in (mip_run_error, mip_plot_error) if message]
    summary_text = build_multi_frontier_summary_text(
        sections=[
            ("Continuous Convex", continuous_run, None),
            (
                "Mixed Integer (Discrete Constraints)",
                mip_run,
                " | ".join(mip_notes) if mip_notes else None,
            ),
        ],
        output_dir=output_dir,
        artifacts=[
            ("Output directory", output_dir),
            ("Summary", output_dir / "frontier_summary.txt"),
            ("Continuous full CSV", continuous_csv_path),
            ("MIP full CSV", mip_csv_path),
            ("Continuous static plot", continuous_png_path),
            ("MIP static plot", mip_png_path),
            ("Interactive comparison plot", comparison_html_path),
        ],
    )
    save_frontier_summary(summary_text, output_dir)
    print(summary_text)


if __name__ == "__main__":
    main()
