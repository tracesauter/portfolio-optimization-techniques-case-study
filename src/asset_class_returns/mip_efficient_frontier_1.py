"""
Mixed-integer minimum-variance efficient frontier with discrete weight grids,
a no-shorting budget constraint, a forbidden "inflation" asset, and a
cardinality limit on the number of non-zero holdings.

Formulation (one MIQP per target return r*)
-------------------------------------------

    minimize    wᵀ Σ w
    subject to  μᵀ w ≥ r*
                Σ_i w_i ≤ 1,            w ≥ 0       # long-only, cash allowed
                w[inflation_index] = 0
                w_i ∈ {0, 0.025, 0.050, 0.075}      for i in alt_indices
                w_i ∈ {0.00, 0.01, ..., 1.00}       for the other held assets
                Σ_i 1{w_i > 0} ≤ max_holdings

Discrete weight grids are implemented by introducing integer "unit-count"
variables linked to w via linear equalities. The cardinality limit is
enforced with per-asset binary indicators z_i and the big-M link
w_i ≤ U_i · z_i, where U_i is the asset's weight upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import cvxpy as cp
import numpy as np
import numpy.typing as npt


FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FrontierPoint:
    """Outcome of a single minimum-variance optimization at a given target."""

    target_return: float
    achieved_return: float
    variance: float
    volatility: float
    weights: FloatArray
    status: str


def solve_min_variance(
    mu: FloatArray,
    sigma: FloatArray,
    target_return: float,
    *,
    inflation_index: int = 7,
    alt_indices: Sequence[int] = (5, 6),
    alt_grid_step: float = 0.025,
    alt_grid_max_units: int = 3,  # yields {0, .025, .05, .075}
    regular_grid_step: float = 0.01,  # yields {0, .01, ..., 1.00}
    max_holdings: int = 6,
    solver: str | None = cp.SCIP,
    solver_kwargs: dict | None = None,
) -> FrontierPoint:
    """Solve the single-period MIQP for one target return."""
    n = int(mu.shape[0])
    alt_set = set(alt_indices)
    regular_indices: list[int] = [
        i for i in range(n) if i != inflation_index and i not in alt_set
    ]

    # --- Decision variables ------------------------------------------------
    w = cp.Variable(n, name="weights")
    z = cp.Variable(n, boolean=True, name="selected")

    # Integer "unit" variables: weight = unit_count * grid_step.
    n_reg = cp.Variable(len(regular_indices), integer=True, name="units_regular")
    n_alt = cp.Variable(len(alt_indices), integer=True, name="units_alt")

    max_reg_units: int = int(round(1.0 / regular_grid_step))  # 100
    max_alt_weight: float = alt_grid_step * alt_grid_max_units  # 0.075

    constraints: list[cp.Constraint] = []

    # Link continuous weights to integer unit counts.
    for k, i in enumerate(regular_indices):
        constraints.append(w[i] == regular_grid_step * n_reg[k])
    for k, i in enumerate(alt_indices):
        constraints.append(w[i] == alt_grid_step * n_alt[k])

    constraints += [n_reg >= 0, n_reg <= max_reg_units]
    constraints += [n_alt >= 0, n_alt <= alt_grid_max_units]

    # Inflation cannot be held.
    constraints += [w[inflation_index] == 0, z[inflation_index] == 0]

    # Budget + long-only.
    constraints += [w >= 0, cp.sum(w) <= 1.0]

    # Selection indicators: z_i = 0  ⇒  w_i = 0.
    # Big-M tightened to each asset's weight upper bound.
    for i in regular_indices:
        constraints.append(w[i] <= z[i])  # U_i = 1
    for i in alt_indices:
        constraints.append(w[i] <= max_alt_weight * z[i])  # U_i = 0.075

    # Cardinality.
    constraints.append(cp.sum(z) <= max_holdings)

    # Expected-return floor.
    constraints.append(mu @ w >= target_return)

    # --- Objective ---------------------------------------------------------
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, **(solver_kwargs or {}))

    status = problem.status
    if status not in {"optimal", "optimal_inaccurate"}:
        return FrontierPoint(
            target_return=target_return,
            achieved_return=float("nan"),
            variance=float("nan"),
            volatility=float("nan"),
            weights=np.full(n, np.nan),
            status=status,
        )

    # Snap away solver-tolerance noise and re-project onto the discrete grid.
    w_val = _snap_to_grid(
        np.asarray(w.value, dtype=float).ravel(),
        regular_indices=regular_indices,
        regular_step=regular_grid_step,
        alt_indices=list(alt_indices),
        alt_step=alt_grid_step,
    )
    variance = float(w_val @ sigma @ w_val)

    return FrontierPoint(
        target_return=target_return,
        achieved_return=float(mu @ w_val),
        variance=variance,
        volatility=float(np.sqrt(max(variance, 0.0))),
        weights=w_val,
        status=status,
    )


def _snap_to_grid(
    w: FloatArray,
    *,
    regular_indices: Sequence[int],
    regular_step: float,
    alt_indices: Sequence[int],
    alt_step: float,
) -> FloatArray:
    """Round each weight to its prescribed grid, zero out numerical dust."""
    out = w.copy()
    for i in regular_indices:
        out[i] = round(out[i] / regular_step) * regular_step
    for i in alt_indices:
        out[i] = round(out[i] / alt_step) * alt_step
    out[np.abs(out) < 1e-12] = 0.0
    return out


def build_efficient_frontier(
    mu: FloatArray,
    sigma: FloatArray,
    *,
    return_min: float = 0.04,
    return_max: float = 0.15,
    return_step: float = 0.005,
    **solver_kwargs,
) -> list[FrontierPoint]:
    """Sweep the target-return grid and solve a MIQP at each point."""
    targets = np.round(
        np.arange(return_min, return_max + 0.5 * return_step, return_step), 6
    )
    return [solve_min_variance(mu, sigma, float(r), **solver_kwargs) for r in targets]


if __name__ == "__main__":
    # Illustrative example — replace mu / sigma with your estimates.
    rng = np.random.default_rng(0)
    k = 8
    mu_example: FloatArray = np.array([0.08, 0.10, 0.06, 0.12, 0.09, 0.14, 0.11, 0.03])
    rnd = rng.standard_normal((k, k))
    sigma_example: FloatArray = rnd @ rnd.T / k + np.diag([0.04] * k)

    frontier = build_efficient_frontier(mu_example, sigma_example, solver=cp.SCIP)
    for pt in frontier:
        if pt.status.startswith("optimal"):
            held = int(np.sum(pt.weights > 0))
            print(
                f"r*={pt.target_return:5.3f}  "
                f"vol={pt.volatility:6.4f}  "
                f"held={held}  "
                f"weights={np.round(pt.weights, 4).tolist()}"
            )
        else:
            print(f"r*={pt.target_return:5.3f}  status={pt.status}")
