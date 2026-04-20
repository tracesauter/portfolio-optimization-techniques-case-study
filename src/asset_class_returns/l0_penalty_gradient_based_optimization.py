from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeAlias, Tuple

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


def find_basis_for_0_return_space(expected_returns: Array1D) -> Tuple[Array2D, Array2D]:
    num_asset_classes = expected_returns.shape[0]
    constraints = np.vstack([expected_returns, np.ones(num_asset_classes)])
    V = np.linalg.svd(constraints).T
    num_vectors_in_basis = V.shape[1] - 2
    basis = V[:, -num_vectors_in_basis:]
    projection_matrix_onto_basis = basis @ basis.T
    return basis, projection_matrix_onto_basis


def sample_random_long_only(n_asset_classes: int, n_samples: int) -> Array2D:
    dirichlet_samples_including_cash = np.random.dirichlet(
        np.ones(
            n_asset_classes + 1,
        ),
        n_samples,
    )
    return dirichlet_samples_including_cash[:, :-1]
