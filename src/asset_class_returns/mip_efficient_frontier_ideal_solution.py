from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    weights: Array1D | None


@dataclass(frozen=True)
class FrontierModel:
    problem: cp.Problem
    target_return_param: cp.Parameter
    weights: cp.Variable
    selected: cp.Variable
    standard_units: cp.Variable
    special_units: cp.Variable
    standard_indices: tuple[int, ...]
    special_indices: tuple[int, int]
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
    special_indices: tuple[int, int] = (5, 6),
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
    if len(special_indices) != 2:
        raise ValueError("special_indices must contain exactly two asset indices.")
    if len(set(special_indices)) != len(special_indices):
        raise ValueError("special_indices must be unique.")
    if inflation_index in special_indices:
        raise ValueError("inflation_index cannot also be a special index.")
    if any(index < 0 or index >= mu.size for index in special_indices):
        raise ValueError("special_indices contains an out-of-bounds index.")

    sigma = 0.5 * (sigma + sigma.T)
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(sigma)))
    if min_eigenvalue < -psd_tolerance:
        raise ValueError(
            "covariance must be positive semidefinite for a convex variance objective. "
            f"Minimum eigenvalue was {min_eigenvalue:.6e}."
        )

    return mu, sigma


def build_portfolio_miqp(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    inflation_index: int = 7,
    special_indices: tuple[int, int] = (5, 6),
    max_nonzero_assets: int = 6,
) -> FrontierModel:
    """Build one reusable MIQP model for the whole frontier sweep."""
    if max_nonzero_assets < 1:
        raise ValueError("max_nonzero_assets must be at least 1.")

    mu, sigma = validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        inflation_index=inflation_index,
        special_indices=special_indices,
    )

    n_assets = mu.size
    excluded = set(special_indices) | {inflation_index}
    standard_indices = tuple(i for i in range(n_assets) if i not in excluded)

    weights = cp.Variable(n_assets, nonneg=True, name="weights")
    selected = cp.Variable(n_assets, boolean=True, name="selected_assets")
    standard_units = cp.Variable(
        len(standard_indices),
        integer=True,
        name="standard_units",
    )
    special_units = cp.Variable(
        len(special_indices),
        integer=True,
        name="special_units",
    )
    target_return = cp.Parameter(name="target_return")

    constraints: list[cp.Constraint] = [
        cp.sum(weights) <= 1.0,
        mu @ weights >= target_return,
        cp.sum(selected) <= max_nonzero_assets,
        cp.sum(selected) >= 1,
        weights[inflation_index] == 0.0,
        selected[inflation_index] == 0.0,
    ]

    # Integer unit counts keep the weights exactly on the allowed grids.
    for k, asset_index in enumerate(standard_indices):
        constraints.extend(
            [
                standard_units[k] >= 0,
                standard_units[k] <= 100,
                weights[asset_index] == 0.01 * standard_units[k],
                standard_units[k] <= 100 * selected[asset_index],
                standard_units[k] >= selected[asset_index],
            ]
        )

    for k, asset_index in enumerate(special_indices):
        constraints.extend(
            [
                special_units[k] >= 0,
                special_units[k] <= 3,
                weights[asset_index] == 0.025 * special_units[k],
                special_units[k] <= 3 * selected[asset_index],
                special_units[k] >= selected[asset_index],
            ]
        )

    objective = cp.Minimize(cp.quad_form(weights, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, constraints)

    return FrontierModel(
        problem=problem,
        target_return_param=target_return,
        weights=weights,
        selected=selected,
        standard_units=standard_units,
        special_units=special_units,
        standard_indices=standard_indices,
        special_indices=special_indices,
        inflation_index=inflation_index,
    )


def reconstruct_weights(model: FrontierModel, n_assets: int) -> Array1D:
    """Reconstruct exact weights from the integer unit variables."""
    if model.standard_units.value is None or model.special_units.value is None:
        raise ValueError("Model has not been solved yet.")

    weights = np.zeros(n_assets, dtype=np.float64)
    standard_units = np.rint(
        np.asarray(model.standard_units.value, dtype=np.float64).reshape(-1)
    ).astype(int)
    special_units = np.rint(
        np.asarray(model.special_units.value, dtype=np.float64).reshape(-1)
    ).astype(int)

    for asset_index, units in zip(model.standard_indices, standard_units):
        weights[asset_index] = 0.01 * float(units)

    for asset_index, units in zip(model.special_indices, special_units):
        weights[asset_index] = 0.025 * float(units)

    weights[model.inflation_index] = 0.0
    return weights


def solve_efficient_frontier(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    target_returns: Iterable[float] | None = None,
    inflation_index: int = 7,
    special_indices: tuple[int, int] = (5, 6),
    max_nonzero_assets: int = 6,
    solver: str = "SCIP",
    verbose: bool = False,
    **solver_kwargs: object,
) -> list[FrontierPoint]:
    """Solve the minimum-variance portfolio for each target return."""
    if solver != "SCIP":
        raise ValueError(
            "This module is written for the SCIP MIQP path only. "
            "Use solver='SCIP'."
        )

    installed_solvers = set(cp.installed_solvers())
    if "SCIP" not in installed_solvers:
        raise RuntimeError(
            "SCIP is not available. This module requires PySCIPOpt so CVXPY can "
            "solve the mixed-integer quadratic program. Try `uv add pyscipopt==5.4.0`. "
            "If that falls back to a source build, you will also need a compatible "
            "SCIP installation with headers."
        )

    mu, sigma = validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        inflation_index=inflation_index,
        special_indices=special_indices,
    )

    if target_returns is None:
        target_returns = make_target_returns()

    model = build_portfolio_miqp(
        expected_returns=mu,
        covariance=sigma,
        inflation_index=inflation_index,
        special_indices=special_indices,
        max_nonzero_assets=max_nonzero_assets,
    )

    results: list[FrontierPoint] = []

    for target_return in target_returns:
        model.target_return_param.value = float(target_return)
        model.problem.solve(
            solver=cp.SCIP,
            warm_start=True,
            verbose=verbose,
            **solver_kwargs,
        )

        status = model.problem.status
        if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            weights = reconstruct_weights(model, n_assets=mu.size)
            achieved_return = float(mu @ weights)
            variance = float(weights @ sigma @ weights)
            invested_weight = float(np.sum(weights))
            holdings_count = int(np.count_nonzero(weights > 0.0))

            results.append(
                FrontierPoint(
                    target_return=float(target_return),
                    status=status,
                    achieved_return=achieved_return,
                    variance=variance,
                    volatility=float(np.sqrt(max(variance, 0.0))),
                    invested_weight=invested_weight,
                    holdings_count=holdings_count,
                    weights=weights,
                )
            )
        else:
            results.append(
                FrontierPoint(
                    target_return=float(target_return),
                    status=status,
                    achieved_return=None,
                    variance=None,
                    volatility=None,
                    invested_weight=None,
                    holdings_count=None,
                    weights=None,
                )
            )

    return results


def frontier_to_frame(
    points: Sequence[FrontierPoint],
    asset_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert frontier results to a tabular view."""
    rows: list[dict[str, float | int | str | None]] = []

    for point in points:
        row: dict[str, float | int | str | None] = {
            "target_return": point.target_return,
            "status": point.status,
            "achieved_return": point.achieved_return,
            "variance": point.variance,
            "volatility": point.volatility,
            "invested_weight": point.invested_weight,
            "holdings_count": point.holdings_count,
        }

        if point.weights is not None:
            if asset_names is None:
                labels = [f"asset_{i}" for i in range(point.weights.size)]
            else:
                labels = list(asset_names)
                if len(labels) != point.weights.size:
                    raise ValueError(
                        "asset_names must have the same length as each weight vector."
                    )

            for label, weight in zip(labels, point.weights):
                row[f"weight_{label}"] = float(weight)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Build the frontier from the project CSV and print a compact summary."""
    asset_names, expected_returns, covariance = prepare_project_inputs()

    try:
        points = solve_efficient_frontier(
            expected_returns=expected_returns,
            covariance=covariance,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    summary = frontier_to_frame(points, asset_names=asset_names)
    columns = [
        "target_return",
        "status",
        "achieved_return",
        "volatility",
        "invested_weight",
        "holdings_count",
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
