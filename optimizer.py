"""
Modern Portfolio Theory helpers.

Everything here works off daily returns for a set of assets:
- annualize(): turns daily mean/covariance into annualized figures
- random_portfolios(): Monte Carlo cloud of long-only portfolios (illustrative)
- min_volatility() / max_sharpe(): the two classic reference portfolios
- efficient_frontier(): the actual frontier curve, via constrained optimization
- portfolio_for_target_return() / portfolio_for_target_risk(): user-driven picks

All optimized portfolios (everything except the Monte Carlo cloud) honor
an optional per-asset weight cap, used as a simple diversification control:
capping any single stock's weight forces the optimizer to spread risk
across more names instead of concentrating in whichever asset has the
best historical return/risk trade-off.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def annualize(daily_returns: pd.DataFrame):
    """Annualized mean return (Series) and covariance matrix (DataFrame)."""
    mean_daily = daily_returns.mean()
    cov_daily = daily_returns.cov()
    return mean_daily * TRADING_DAYS, cov_daily * TRADING_DAYS


def portfolio_performance(weights, mean_returns, cov_matrix, rf=0.0):
    """Return (expected return, volatility, Sharpe ratio) for one weight vector."""
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mean_returns.values)
    vol = float(np.sqrt(w @ cov_matrix.values @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def random_portfolios(n_portfolios, mean_returns, cov_matrix, rf, n_assets, seed=42):
    """
    Monte Carlo cloud of long-only, fully-invested portfolios (weights sum to 1,
    no per-asset cap). This is the illustrative scatter behind the frontier —
    it always uses the full long-only space regardless of any diversification
    cap applied to the optimized portfolios, so the cap's effect is visible as
    the true frontier sitting inside the cloud rather than hugging its edge.
    """
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)
    rets = weights @ mean_returns.values
    cov = cov_matrix.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(vols > 0, (rets - rf) / vols, np.nan)
    return rets, vols, sharpe, weights


def _bounds_and_constraints(n_assets, max_weight, allow_short):
    lower = -max_weight if allow_short else 0.0
    bounds = tuple((lower, max_weight) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return bounds, constraints


def min_volatility(mean_returns, cov_matrix, max_weight=1.0, allow_short=False):
    n = len(mean_returns)
    bounds, constraints = _bounds_and_constraints(n, max_weight, allow_short)
    x0 = np.repeat(1 / n, n)

    result = minimize(
        lambda w: w @ cov_matrix.values @ w,
        x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x


def max_sharpe(mean_returns, cov_matrix, rf=0.0, max_weight=1.0, allow_short=False):
    n = len(mean_returns)
    bounds, constraints = _bounds_and_constraints(n, max_weight, allow_short)
    x0 = np.repeat(1 / n, n)

    def neg_sharpe(w):
        _, _, sharpe = portfolio_performance(w, mean_returns, cov_matrix, rf)
        return -sharpe if np.isfinite(sharpe) else 1e6

    result = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x


def efficient_frontier(mean_returns, cov_matrix, n_points=40, max_weight=1.0, allow_short=False):
    """Sweep target returns and minimize volatility at each — the frontier curve."""
    n = len(mean_returns)
    bounds, base_constraints = _bounds_and_constraints(n, max_weight, allow_short)

    mv_weights = min_volatility(mean_returns, cov_matrix, max_weight, allow_short)
    ret_lo, _, _ = portfolio_performance(mv_weights, mean_returns, cov_matrix)
    ret_hi = mean_returns.max() * (1.2 if allow_short else 1.0)
    if ret_hi <= ret_lo:
        ret_hi = ret_lo * 1.1 if ret_lo > 0 else ret_lo + 0.01
    target_returns = np.linspace(ret_lo, ret_hi, n_points)

    frontier = []
    x0 = mv_weights.copy()
    for target in target_returns:
        constraints = base_constraints + [
            {"type": "eq", "fun": lambda w, t=target: w @ mean_returns.values - t}
        ]
        result = minimize(
            lambda w: w @ cov_matrix.values @ w,
            x0, method="SLSQP", bounds=bounds, constraints=constraints,
        )
        if result.success:
            vol = float(np.sqrt(result.fun))
            frontier.append((float(target), vol, result.x))
            x0 = result.x  # warm-start the next point

    return frontier


def portfolio_for_target_return(mean_returns, cov_matrix, target_return, max_weight=1.0, allow_short=False):
    """Minimum-volatility portfolio achieving at least the target return."""
    n = len(mean_returns)
    bounds, constraints = _bounds_and_constraints(n, max_weight, allow_short)
    constraints = constraints + [
        {"type": "ineq", "fun": lambda w: w @ mean_returns.values - target_return}
    ]
    x0 = np.repeat(1 / n, n)

    result = minimize(
        lambda w: w @ cov_matrix.values @ w,
        x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x, bool(result.success)


def portfolio_for_target_risk(mean_returns, cov_matrix, target_vol, max_weight=1.0, allow_short=False):
    """Maximum-return portfolio with volatility at or below the target."""
    n = len(mean_returns)
    bounds, constraints = _bounds_and_constraints(n, max_weight, allow_short)
    constraints = constraints + [
        {"type": "ineq", "fun": lambda w: target_vol**2 - w @ cov_matrix.values @ w}
    ]
    x0 = np.repeat(1 / n, n)

    result = minimize(
        lambda w: -(w @ mean_returns.values),
        x0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return result.x, bool(result.success)


def effective_number_of_assets(weights):
    """1 / Herfindahl index — a concentration-aware count of 'how many stocks' a
    portfolio really holds. Equal weights across N assets gives exactly N;
    concentrating into one name pulls this toward 1 regardless of how many
    names have a nonzero weight."""
    w = np.asarray(weights, dtype=float)
    hhi = np.sum(np.square(w))
    return 1.0 / hhi if hhi > 0 else np.nan
