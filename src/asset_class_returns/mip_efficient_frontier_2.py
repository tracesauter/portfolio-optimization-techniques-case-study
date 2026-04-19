from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeAlias

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray


Array1D: TypeAlias = NDArray[np.float64]
Array2D: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class FrontierPoint:
    target_return: float
    status: str
    expected_return: float | None
    variance: float | None
    volatility: float | None
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
    special_indices: tuple[int, ...]
    inflation_index: int


def make_target_returns(
    start: float = 0.04,
    stop: float = 0.15,
    step: float = 0.005,
) -> Array1D:
    """
    Create target returns in decimal form.
    Example: 0.04 means 4%.
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    n_steps = int(round((stop - start) / step)) + 1
    targets = start + step * np.arange(n_steps, dtype=float)
    return np.round(targets, 10)


def validate_inputs(
    expected_returns: Sequence[float],
    covariance: Sequence[Sequence[float]],
    inflation_index: int = 7,
    special_indices: tuple[int, int] = (5, 6),
    psd_tolerance: float = 1e-10,
) -> tuple[Array1D, Array2D]:
    """
    Validate dimensions and ensure the covariance matrix is symmetric PSD
    up to a small numerical tolerance.
    """
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    sigma = np.asarray(covariance, dtype=float)

    if sigma.shape != (mu.size, mu.size):
        raise ValueError(
            f"covariance must have shape {(mu.size, mu.size)}, got {sigma.shape}."
        )

    if not (0 <= inflation_index < mu.size):
        raise ValueError("inflation_index is out of bounds.")

    if len(set(special_indices)) != len(special_indices):
        raise ValueError("special_indices must be unique.")

    if inflation_index in special_indices:
        raise ValueError("inflation_index cannot also be a special index.")

    # Symmetrize in case of tiny numerical asymmetry.
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
    """
    Build the MIQP for variance minimization subject to:
      - target return
      - total invested weight <= 1
      - no shorting
      - inflation asset weight == 0
      - indices in special_indices have weights in {0, 0.025, 0.05, 0.075}
      - all other investable assets have weights in {0, 0.01, ..., 1.00}
      - at most max_nonzero_assets nonzero positions
    """
    mu, sigma = validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        inflation_index=inflation_index,
        special_indices=special_indices,
    )

    n_assets = mu.size
    excluded = set(special_indices) | {inflation_index}
    standard_indices = tuple(i for i in range(n_assets) if i not in excluded)

    # Decision variables
    w = cp.Variable(n_assets, nonneg=True, name="weights")
    z = cp.Variable(n_assets, boolean=True, name="selected_assets")

    # Integer unit variables
    # Standard assets: whole percentages -> 0.01 * unit, unit in {0, ..., 100}
    standard_units = cp.Variable(len(standard_indices), integer=True, name="standard_units")

    # Special assets: 2.5% increments -> 0.025 * unit, unit in {0, 1, 2, 3}
    special_units = cp.Variable(len(special_indices), integer=True, name="special_units")

    target_return = cp.Parameter(name="target_return")

    constraints: list[cp.Constraint] = [
        cp.sum(w) <= 1.0,
        mu @ w >= target_return,
        cp.sum(z) <= max_nonzero_assets,
        cp.sum(z) >= 1,            # avoids the all-zero portfolio
        w[inflation_index] == 0.0,
        z[inflation_index] == 0.0,
    ]

    # Standard assets: weights in {0, 0.01, 0.02, ..., 1.00}
    for k, asset_idx in enumerate(standard_indices):
        constraints.extend(
            [
                standard_units[k] >= 0,
                standard_units[k] <= 100,
                w[asset_idx] == 0.01 * standard_units[k],
                standard_units[k] <= 100 * z[asset_idx],
                standard_units[k] >= z[asset_idx],
            ]
        )

    # Special assets: weights in {0, 0.025, 0.05, 0.075}
    for k, asset_idx in enumerate(special_indices):
        constraints.extend(
            [
                special_units[k] >= 0,
                special_units[k] <= 3,
                w[asset_idx] == 0.025 * special_units[k],
                special_units[k] <= 3 * z[asset_idx],
                special_units[k] >= z[asset_idx],
            ]
        )

    objective = cp.Minimize(cp.quad_form(w, sigma))
    problem = cp.Problem(objective, constraints)

    return FrontierModel(
        problem=problem,
        target_return_param=target_return,
        weights=w,
        selected=z,
        standard_units=standard_units,
        special_units=special_units,
        standard_indices=standard_indices,
        special_indices=special_indices,
        inflation_index=inflation_index,
    )


def reconstruct_weights(
    model: FrontierModel,
    n_assets: int,
) -> Array1D:
    """
    Reconstruct exact weights from the integer unit variables so that
    the returned weights sit exactly on the allowed grids.
    """
    if model.standard_units.value is None or model.special_units.value is None:
        raise ValueError("Model has not been solved yet.")

    weights = np.zeros(n_assets, dtype=float)

    standard_units = np.rint(np.asarray(model.standard_units.value).reshape(-1)).astype(int)
    special_units = np.rint(np.asarray(model.special_units.value).reshape(-1)).astype(int)

    for asset_idx, units in zip(model.standard_indices, standard_units):
        weights[asset_idx] = 0.01 * units

    for asset_idx, units in zip(model.special_indices, special_units):
        weights[asset_idx] = 0.025 * units

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
    """
    Solve the variance-minimizing portfolio for each target return level.

    Parameters
    ----------
    expected_returns
        Mean returns in decimal form, e.g. 0.08 for 8%.
    covariance
        Covariance matrix for the same assets.
    target_returns
        Iterable of target returns in decimal form. If None, uses 4% to 15%
        in 0.5% increments.
    solver
        Mixed-integer solver to use. Defaults to SCIP.
    verbose
        Whether to print solver output.
    solver_kwargs
        Additional keyword args passed to problem.solve(...).

    Returns
    -------
    list[FrontierPoint]
        One result per target return. Infeasible targets will return a point
        whose weights / variance / volatility are None.
    """
    mu, sigma = validate_inputs(
        expected_returns=expected_returns,
        covariance=covariance,
        inflation_index=inflation_index,
        special_indices=special_indices,
    )

    if target_returns is None:
        target_returns = make_target_returns()

    installed = set(cp.installed_solvers())
    if solver not in installed:
        raise RuntimeError(
            f"Requested solver '{solver}' is not installed. "
            f"Installed solvers: {sorted(installed)}"
        )

    model = build_portfolio_miqp(
        expected_returns=mu,
        covariance=sigma,
        inflation_index=inflation_index,
        special_indices=special_indices,
        max_nonzero_assets=max_nonzero_assets,
    )

    results: list[FrontierPoint] = []

    for target in target_returns:
        model.target_return_param.value = float(target)

        model.problem.solve(
            solver=solver,
            warm_start=True,
            verbose=verbose,
            **solver_kwargs,
        )

        status = model.problem.status
        if status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            weights = reconstruct_weights(model=model, n_assets=mu.size)
            expected_return = float(mu @ weights)
            variance = float(weights @ sigma @ weights)
            volatility = float(np.sqrt(max(variance, 0.0)))

            results.append(
                FrontierPoint(
                    target_return=float(target),
                    status=status,
                    expected_return=expected_return,
                    variance=variance,
                    volatility=volatility,
                    weights=weights,
                )
            )
        else:
            results.append(
                FrontierPoint(
                    target_return=float(target),
                    status=status,
                    expected_return=None,
                    variance=None,
                    volatility=None,
                    weights=None,
                )
            )

    return results