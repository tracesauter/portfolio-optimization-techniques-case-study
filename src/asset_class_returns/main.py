from pathlib import Path

from .convex_continuous_efficient_frontier import (
    frontier_to_frame,
    prepare_project_inputs,
    solve_efficient_frontier,
)
from .frontier_visualization import (
    build_frontier_summary_text,
    create_run_output_dir,
    plot_frontier_interactive,
    plot_frontier_static,
    save_frontier_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "continuous_frontier_runs"


def main() -> None:
    asset_names, expected_returns, covariance = prepare_project_inputs()
    run = solve_efficient_frontier(
        expected_returns=expected_returns,
        covariance=covariance,
        asset_names=asset_names,
    )
    frontier_frame = frontier_to_frame(run)

    output_dir = create_run_output_dir(OUTPUTS_ROOT)
    csv_path = output_dir / "frontier_full.csv"
    png_path = output_dir / "efficient_frontier.png"
    html_path = output_dir / "efficient_frontier.html"

    frontier_frame.to_csv(csv_path, index=False)
    plot_frontier_static(run, png_path)
    plot_frontier_interactive(run, html_path)

    summary_text = build_frontier_summary_text(run, output_dir)
    save_frontier_summary(summary_text, output_dir)
    print(summary_text)


if __name__ == "__main__":
    main()
