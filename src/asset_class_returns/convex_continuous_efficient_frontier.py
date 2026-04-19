from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence, TypeAlias

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .estimate_covariance import (
    construct_conservative_covariance,
    exponential_decay_weights,
)


Array1D: TypeAlias = NDArray[np.float64]
Array2D: TypeAlias = NDArray[np.float64]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class FrontierPoint:
    target_return: float
    status: str
    achieved_return: float | None
    variance: float | None
    volatility: float | None
    invested_weight: float | None
    holdings_count: int | None
    solve_time_seconds: float
    weights: Array1D | None


@dataclass(frozen=True)
class FrontierRun:
    solver: str
    total_solve_time_seconds: float
    asset_names: list[str]
    points: list[FrontierPoint]


@dataclass(frozen=True)
class FrontierModel:
    problem: cp.Problem
    target_return_param: cp.Parameter
    weights: cp.Variable
    capped_indices: tuple[int, int]
    inflation_index: int


def make_target_returns(
    start: float = 0.04,
    stop: float = 0.15,
    step: float = 0.005,
) -> Array1D:
    """Create target returns in decimal form."""
    if step <= 0.0:
        raise ValueError("step must be positive.")
    if stop < start:
        raise ValueError("stop must be greater than or equal to start.")

    n_steps = int(round((stop - start) / step)) + 1
    targets = start + step * np.arange(n_steps, dtype=np.float64)
    return np.round(targets, 10)


def prepare_project_inputs(
    csv_path: Path | None = None,
    decay: float = 0.9925,
) -> tuple[list[str], Array1D, Array2D]:
    """Load the project data and prepare mean returns and covariance."""
    if csv_path is None:
        csv_path = DATA_DIR / "asset_class_returns.csv"

    df = pd.read_csv(csv_path)
    returns_df = df.drop(columns=["Year"])
    asset_names = returns_df.columns.tolist()
    returns = returns_df.to_numpy(dtype=np.float64)

    observation_weights = exponential_decay_weights(returns.shape[0], decay=decay)
    expected_returns = np.average(returns, axis=0, weights=observation_weights).astype(
        np.float64
    )
    covariance = construct_conservative_covariance(
        returns,
        observation_weights=observation_weights,
    ).conservative_covariance

    return asset_names, expected_returns, np.asarray(covariance, dtype=np.float64)


def validate_inputs(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    inflation_index: int = 7,
    capped_indices: tuple[int, int] = (5, 6),
    psd_tolerance: float = 1e-10,
) -> tuple[Array1D, Array2D]:
    """Validate dimensions and basic numerical assumptions."""
    mu = np.asarray(expected_returns, dtype=np.float64).reshape(-1)
    sigma = np.asarray(covariance, dtype=np.float64)

    if mu.size == 0:
        raise ValueError("expected_returns must be non-empty.")
    if sigma.shape != (mu.size, mu.size):
        raise ValueError(
            f"covariance must have shape {(mu.size, mu.size)}, got {sigma.shape}."
        )
    if not np.isfinite(mu).all():
        raise ValueError("expected_returns contains NaN or infinite values.")
    if not np.isfinite(sigma).all():
        raise ValueError("covariance contains NaN or infinite values.")
    if not (0 <= inflation_index < mu.size):
        raise ValueError("inflation_index is out of bounds.")
    if len(capped_indices) != 2:
        raise ValueError("capped_indices must contain exactly two asset indices.")
    if len(set(capped_indices)) != len(capped_indices):
        raise ValueError("capped_indices must be unique.")
    if inflation_index in capped_indices:
        raise ValueError("inflation_index cannot also be capped.")
    if any(index < 0 or index >= mu.size for index in capped_indices):
        raise ValueError("capped_indices contains an out-of-bounds index.")

    sigma = 0.5 * (sigma + sigma.T)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(sigma)))
    if min_eigenvalue < -psd_tolerance:
        raise ValueError(
            "covariance must be positive semidefinite for a convex variance objective. "
            f"Minimum eigenvalue was {min_eigenvalue:.6e}."
        )

    return mu, sigma


def build_portfolio_qp(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    inflation_index: int = 7,
    capped_indices: tuple[int, int] = (5, 6),
    capped_weight: float = 0.075,
) -> FrontierModel:
    """Build one reusable convex QP model for the frontier sweep."""
    if not (0.0 <= capped_weight <= 1.0):
        raise ValueError("capped_weight must be between 0 and 1.")

    mu, sigma = validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        inflation_index=inflation_index,
        capped_indices=capped_indices,
    )

    weights = cp.Variable(mu.size, nonneg=True, name="weights")
    target_return = cp.Parameter(name="target_return")

    constraints: list[cp.Constraint] = [
        cp.sum(weights) <= 1.0,
        mu @ weights >= target_return,
        weights[inflation_index] == 0.0,
    ]
    for asset_index in capped_indices:
        constraints.append(weights[asset_index] <= capped_weight)

    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, constraints)

    return FrontierModel(
        problem=problem,
        target_return_param=target_return,
        weights=weights,
        capped_indices=capped_indices,
        inflation_index=inflation_index,
    )


def solve_efficient_frontier(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    asset_names: Sequence[str] | None = None,
    target_returns: Iterable[float] | None = None,
    inflation_index: int = 7,
    capped_indices: tuple[int, int] = (5, 6),
    capped_weight: float = 0.075,
    solver: str = "OSQP",
    verbose: bool = False,
    nonzero_tolerance: float = 1e-8,
    **solver_kwargs: object,
) -> FrontierRun:
    """Solve the minimum-variance continuous frontier for each target return."""
    if nonzero_tolerance < 0.0:
        raise ValueError("nonzero_tolerance must be nonnegative.")

    mu, sigma = validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        inflation_index=inflation_index,
        capped_indices=capped_indices,
    )

    if target_returns is None:
        target_returns = make_target_returns()

    labels = (
        [f"asset_{i}" for i in range(mu.size)]
        if asset_names is None
        else list(asset_names)
    )
    if len(labels) != mu.size:
        raise ValueError("asset_names must have the same length as expected_returns.")

    installed_solvers = set(cp.installed_solvers())
    if solver not in installed_solvers:
        raise RuntimeError(
            f"Requested solver '{solver}' is not installed. "
            f"Installed solvers: {sorted(installed_solvers)}"
        )

    effective_solver_kwargs = dict(solver_kwargs)
    if solver == "OSQP":
        effective_solver_kwargs.setdefault("eps_abs", 1e-10)
        effective_solver_kwargs.setdefault("eps_rel", 1e-10)
        effective_solver_kwargs.setdefault("polish", True)

    model = build_portfolio_qp(
        expected_returns=mu,
        covariance=sigma,
        inflation_index=inflation_index,
        capped_indices=capped_indices,
        capped_weight=capped_weight,
    )

    points: list[FrontierPoint] = []
    frontier_start_time = perf_counter()

    for target_return in target_returns:
        model.target_return_param.value = float(target_return)

        solve_start_time = perf_counter()
        model.problem.solve(
            solver=solver,
            warm_start=True,
            verbose=verbose,
            **effective_solver_kwargs,
        )
        solve_time_seconds = perf_counter() - solve_start_time

        status = model.problem.status
        if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            weights = np.asarray(model.weights.value, dtype=np.float64).reshape(-1)
            weights = weights.copy()
            weights[weights < 0.0] = np.maximum(weights[weights < 0.0], 0.0)
            weights[np.abs(weights) < nonzero_tolerance] = 0.0
            weights[inflation_index] = 0.0

            achieved_return = float(mu @ weights)
            variance = float(weights @ sigma @ weights)
            invested_weight = float(np.sum(weights))
            holdings_count = int(np.count_nonzero(weights > nonzero_tolerance))

            points.append(
                FrontierPoint(
                    target_return=float(target_return),
                    status=status,
                    achieved_return=achieved_return,
                    variance=variance,
                    volatility=float(np.sqrt(max(variance, 0.0))),
                    invested_weight=invested_weight,
                    holdings_count=holdings_count,
                    solve_time_seconds=solve_time_seconds,
                    weights=weights,
                )
            )
        else:
            points.append(
                FrontierPoint(
                    target_return=float(target_return),
                    status=status,
                    achieved_return=None,
                    variance=None,
                    volatility=None,
                    invested_weight=None,
                    holdings_count=None,
                    solve_time_seconds=solve_time_seconds,
                    weights=None,
                )
            )

    total_solve_time_seconds = perf_counter() - frontier_start_time
    return FrontierRun(
        solver=solver,
        total_solve_time_seconds=total_solve_time_seconds,
        asset_names=labels,
        points=points,
    )


def frontier_to_frame(run: FrontierRun) -> pd.DataFrame:
    """Convert frontier results to a tabular view."""
    rows: list[dict[str, float | int | str | None]] = []

    for point in run.points:
        row: dict[str, float | int | str | None] = {
            "target_return": point.target_return,
            "status": point.status,
            "achieved_return": point.achieved_return,
            "variance": point.variance,
            "volatility": point.volatility,
            "invested_weight": point.invested_weight,
            "holdings_count": point.holdings_count,
            "solve_time_seconds": point.solve_time_seconds,
        }

        if point.weights is not None:
            if len(run.asset_names) != point.weights.size:
                raise ValueError(
                    "asset_names must have the same length as each weight vector."
                )
            for label, weight in zip(run.asset_names, point.weights):
                row[f"weight_{label}"] = float(weight)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Build the frontier from the project CSV and print a compact summary."""
    asset_names, expected_returns, covariance = prepare_project_inputs()
    run = solve_efficient_frontier(
        expected_returns=expected_returns,
        covariance=covariance,
        asset_names=asset_names,
    )

    summary = frontier_to_frame(run)
    columns = [
        "target_return",
        "status",
        "achieved_return",
        "volatility",
        "invested_weight",
        "holdings_count",
        "solve_time_seconds",
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nSolver: {run.solver}")
    print(f"Total solve time (s): {run.total_solve_time_seconds:.6f}")


if __name__ == "__main__":
    main()
