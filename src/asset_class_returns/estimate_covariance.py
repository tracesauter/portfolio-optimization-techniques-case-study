from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import cvxpy as cp
import numpy as np
import numpy.typing as npt


Array1D: TypeAlias = npt.NDArray[np.float64]
Array2D: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ConservativeCovarianceResult:
    observation_weights: Array1D
    base_covariance: Array2D
    base_correlation: Array2D
    base_variances: Array1D
    edited_correlation: Array2D
    repaired_correlation: Array2D
    conservative_variances: Array1D
    conservative_covariance: Array2D


def _as_float_2d_array(x: npt.ArrayLike) -> Array2D:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError("Need at least 2 observations.")
    if arr.shape[1] < 1:
        raise ValueError("Need at least 1 asset.")
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or infinite values.")
    return arr


def _as_float_1d_array(x: npt.ArrayLike, expected_length: int, name: str) -> Array1D:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != expected_length:
        raise ValueError(f"{name} must be a 1D array of length {expected_length}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _symmetrize(matrix: Array2D) -> Array2D:
    return ((matrix + matrix.T) / 2.0).astype(np.float64)


def is_positive_semidefinite(matrix: npt.ArrayLike, tol: float = 1e-10) -> bool:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square.")
    eigvals = np.linalg.eigvalsh(_symmetrize(mat))
    return bool(np.min(eigvals) >= -tol)


def exponential_decay_weights(
    num_observations: int,
    decay: float | None = None,
    half_life: float | None = None,
) -> Array1D:
    """
    Create normalized weights with the most recent observation receiving
    the largest weight.

    Assumes rows are ordered from oldest to newest.

    Parameters
    ----------
    num_observations
        Number of rows in the returns matrix.
    decay
        Per-step decay factor in (0, 1]. If provided, newest gets weight 1,
        previous gets decay, previous gets decay^2, etc.
    half_life
        Alternative to decay. If provided, decay = 0.5 ** (1 / half_life).

    Returns
    -------
    weights : Array1D
        Normalized weights summing to 1.
    """
    if num_observations < 1:
        raise ValueError("num_observations must be positive.")

    if (decay is None) == (half_life is None):
        raise ValueError("Provide exactly one of decay or half_life.")

    if half_life is not None:
        if half_life <= 0:
            raise ValueError("half_life must be positive.")
        decay = 0.5 ** (1.0 / half_life)

    assert decay is not None
    if not (0.0 < decay <= 1.0):
        raise ValueError("decay must be in (0, 1].")

    exponents = np.arange(num_observations - 1, -1, -1, dtype=np.float64)
    raw = decay ** exponents
    return raw / raw.sum()


def normalize_observation_weights(
    num_observations: int,
    observation_weights: npt.ArrayLike | None = None,
) -> Array1D:
    """
    Normalize observation weights to sum to 1.

    If no weights are provided, returns equal weights.
    """
    if observation_weights is None:
        return np.full(num_observations, 1.0 / num_observations, dtype=np.float64)

    weights = _as_float_1d_array(observation_weights, num_observations, "observation_weights")
    if np.any(weights < 0.0):
        raise ValueError("observation_weights must be nonnegative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("observation_weights must sum to a positive number.")
    return weights / total


def estimate_covariance_from_returns(
    returns: npt.ArrayLike,
    observation_weights: npt.ArrayLike | None = None,
) -> tuple[Array2D, Array1D]:
    """
    Estimate a covariance matrix from an m x n matrix of returns.

    Uses an unbiased weighted covariance formula:
        cov = sum_i w_i (x_i - mu)(x_i - mu)^T / (1 - sum_i w_i^2)
    where weights sum to 1.

    For equal weights, this reduces to the usual sample covariance with
    denominator (m - 1).
    """
    x = _as_float_2d_array(returns)
    m, _ = x.shape
    weights = normalize_observation_weights(m, observation_weights)

    mean = np.sum(x * weights[:, None], axis=0)
    centered = x - mean
    scatter = centered.T @ (centered * weights[:, None])

    correction = 1.0 - float(np.sum(weights**2))
    if correction <= 0.0:
        raise ValueError("Degenerate weights: unable to compute unbiased weighted covariance.")

    cov = scatter / correction
    cov = _symmetrize(cov)
    return cov, weights


def covariance_to_correlation(covariance: npt.ArrayLike) -> tuple[Array2D, Array1D]:
    """
    Convert a covariance matrix to a correlation matrix and return the variances.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    cov = _symmetrize(cov)

    variances = np.diag(cov).copy()
    if np.any(variances <= 0.0):
        raise ValueError("All variances must be strictly positive to form correlations.")

    std = np.sqrt(variances)
    denom = np.outer(std, std)
    corr = cov / denom
    corr = _symmetrize(corr)
    np.fill_diagonal(corr, 1.0)

    # small numerical clean-up
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    return corr, variances


def correlation_to_covariance(correlation: npt.ArrayLike, variances: npt.ArrayLike) -> Array2D:
    """
    Build covariance = D * correlation * D, where D = diag(sqrt(variances)).
    """
    corr = np.asarray(correlation, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be a square matrix.")
    corr = _symmetrize(corr)

    vars_ = _as_float_1d_array(variances, corr.shape[0], "variances")
    if np.any(vars_ <= 0.0):
        raise ValueError("variances must be strictly positive.")

    std = np.sqrt(vars_)
    cov = np.diag(std) @ corr @ np.diag(std)
    return _symmetrize(cov)


def apply_asymmetric_correlation_shrink(
    correlation: npt.ArrayLike,
    positive_scale: float = 0.90,
    negative_scale: float = 0.75,
) -> Array2D:
    """
    Shrink off-diagonal correlations toward zero, using different scales for
    positive and negative entries.

    - positive off-diagonals: multiply by positive_scale
    - negative off-diagonals: multiply by negative_scale
    - diagonal remains exactly 1
    """
    corr = np.asarray(correlation, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be a square matrix.")
    corr = _symmetrize(corr)

    if not (0.0 <= positive_scale <= 1.0):
        raise ValueError("positive_scale must be in [0, 1].")
    if not (0.0 <= negative_scale <= 1.0):
        raise ValueError("negative_scale must be in [0, 1].")

    edited = corr.copy()
    n = edited.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)

    pos_mask = (edited > 0.0) & off_diag_mask
    neg_mask = (edited < 0.0) & off_diag_mask

    edited[pos_mask] *= positive_scale
    edited[neg_mask] *= negative_scale

    edited = _symmetrize(edited)
    np.fill_diagonal(edited, 1.0)
    return edited


def inflate_variances(
    variances: npt.ArrayLike,
    above_or_equal_mean_scale: float = 1.025,
    below_mean_scale: float = 1.05,
) -> Array1D:
    """
    Inflate variances using the rule:

    - if variance >= mean(variances), multiply by above_or_equal_mean_scale
    - else multiply by below_mean_scale
    """
    vars_ = np.asarray(variances, dtype=np.float64)
    if vars_.ndim != 1:
        raise ValueError("variances must be a 1D array.")
    if not np.isfinite(vars_).all():
        raise ValueError("variances contains NaN or infinite values.")
    if np.any(vars_ <= 0.0):
        raise ValueError("variances must be strictly positive.")

    if above_or_equal_mean_scale < 1.0:
        raise ValueError("above_or_equal_mean_scale must be >= 1.")
    if below_mean_scale < 1.0:
        raise ValueError("below_mean_scale must be >= 1.")

    mean_var = float(np.mean(vars_))
    scales = np.where(vars_ >= mean_var, above_or_equal_mean_scale, below_mean_scale)
    return (vars_ * scales).astype(np.float64)


def nearest_psd_with_fixed_diagonal(
    matrix: npt.ArrayLike,
    diagonal: npt.ArrayLike,
    solver: str = "SCS",
    solver_options: dict[str, Any] | None = None,
) -> Array2D:
    """
    Solve:
        minimize ||X - A||_F^2
        subject to X is PSD
                   diag(X) = diagonal

    This is a convex semidefinite optimization problem.

    Parameters
    ----------
    matrix
        Target square matrix A.
    diagonal
        Desired diagonal of X.
    solver
        CVXPY solver name. "SCS" is a practical default.
    solver_options
        Optional kwargs passed to problem.solve(...).

    Returns
    -------
    Array2D
        The nearest PSD matrix with the requested diagonal.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square.")
    a = _symmetrize(a)

    diag_target = _as_float_1d_array(diagonal, a.shape[0], "diagonal")

    x = cp.Variable(a.shape, PSD=True)
    objective = cp.Minimize(cp.sum_squares(x - a))
    constraints = [cp.diag(x) == diag_target]
    problem = cp.Problem(objective, constraints)

    solve_kwargs = dict(solver_options or {})
    cvxpy_solver = getattr(cp, solver.upper(), None)
    if cvxpy_solver is None:
        raise ValueError(f"Unknown CVXPY solver: {solver}")

    problem.solve(solver=cvxpy_solver, **solve_kwargs)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Nearest-PSD solve failed with status: {problem.status}")

    result = np.asarray(x.value, dtype=np.float64)
    result = _symmetrize(result)
    np.fill_diagonal(result, diag_target)
    return result


def nearest_correlation_matrix(
    correlation: npt.ArrayLike,
    solver: str = "SCS",
    solver_options: dict[str, Any] | None = None,
) -> Array2D:
    """
    Find the nearest valid correlation matrix in Frobenius norm.

    Equivalent to nearest_psd_with_fixed_diagonal(..., diagonal=np.ones(n)).
    """
    corr = np.asarray(correlation, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be square.")
    corr = _symmetrize(corr)

    n = corr.shape[0]
    repaired = nearest_psd_with_fixed_diagonal(
        matrix=corr,
        diagonal=np.ones(n, dtype=np.float64),
        solver=solver,
        solver_options=solver_options,
    )

    # small numerical clean-up
    repaired = np.clip(repaired, -1.0, 1.0)
    repaired = _symmetrize(repaired)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def construct_conservative_covariance(
    returns: npt.ArrayLike,
    observation_weights: npt.ArrayLike | None = None,
    positive_correlation_scale: float = 0.90,
    negative_correlation_scale: float = 0.75,
    above_or_equal_mean_variance_scale: float = 1.025,
    below_mean_variance_scale: float = 1.05,
    repair_correlation: bool = True,
    solver: str = "SCS",
    solver_options: dict[str, Any] | None = None,
) -> ConservativeCovarianceResult:
    """
    Full conservative-covariance pipeline.

    Steps
    -----
    1. Estimate base covariance from returns.
    2. Convert to correlation + variances.
    3. Shrink positive and negative off-diagonal correlations differently.
    4. Repair the correlation matrix to the nearest valid correlation matrix.
    5. Inflate variances by the specified rule.
    6. Rebuild the conservative covariance matrix.

    Parameters
    ----------
    returns
        m x n matrix of joint returns, rows = observations, cols = assets.
    observation_weights
        Optional observation weights. Must have length m. If None, equal weights
        are used. You can also pass exponential_decay_weights(m, ...).
    positive_correlation_scale
        Multiplier for positive off-diagonal correlations.
    negative_correlation_scale
        Multiplier for negative off-diagonal correlations.
    above_or_equal_mean_variance_scale
        Multiplier for variances >= mean variance.
    below_mean_variance_scale
        Multiplier for variances < mean variance.
    repair_correlation
        If True, repair the edited correlation matrix via nearest correlation.
        Strongly recommended.
    solver
        CVXPY solver name.
    solver_options
        Optional kwargs for CVXPY's solve().

    Returns
    -------
    ConservativeCovarianceResult
        Includes all intermediate matrices/vectors plus the final covariance.
    """
    base_cov, weights = estimate_covariance_from_returns(
        returns=returns,
        observation_weights=observation_weights,
    )

    base_corr, base_variances = covariance_to_correlation(base_cov)

    edited_corr = apply_asymmetric_correlation_shrink(
        correlation=base_corr,
        positive_scale=positive_correlation_scale,
        negative_scale=negative_correlation_scale,
    )

    if repair_correlation:
        repaired_corr = nearest_correlation_matrix(
            correlation=edited_corr,
            solver=solver,
            solver_options=solver_options,
        )
    else:
        repaired_corr = edited_corr
        if not is_positive_semidefinite(repaired_corr):
            raise ValueError(
                "Edited correlation matrix is not PSD. "
                "Set repair_correlation=True or use milder edits."
            )

    conservative_variances = inflate_variances(
        variances=base_variances,
        above_or_equal_mean_scale=above_or_equal_mean_variance_scale,
        below_mean_scale=below_mean_variance_scale,
    )

    conservative_cov = correlation_to_covariance(
        correlation=repaired_corr,
        variances=conservative_variances,
    )

    return ConservativeCovarianceResult(
        observation_weights=weights,
        base_covariance=base_cov,
        base_correlation=base_corr,
        base_variances=base_variances,
        edited_correlation=edited_corr,
        repaired_correlation=repaired_corr,
        conservative_variances=conservative_variances,
        conservative_covariance=conservative_cov,
    )