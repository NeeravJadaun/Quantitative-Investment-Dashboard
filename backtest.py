"""Monthly top-N equal-weight backtest with transaction costs and performance metrics."""
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


def run_backtest(monthly: pd.DataFrame, alpha: pd.DataFrame, benchmark: pd.Series,
                 top_n: int = 5, cost_bps: float = 10.0) -> dict:
    """Rebalance at each month-end t into the `top_n` stocks by alpha[t], equal weight.

    The portfolio earns the return from t to the next month-end, so the alpha known at
    t never sees the return it is used to forecast. Costs are `cost_bps` per unit of
    traded notional, charged at the rebalance.
    """
    rets = monthly.pct_change()
    bench_rets = benchmark.reindex(monthly.index).pct_change()
    dates = alpha.dropna(how="all").index
    prev_w = pd.Series(0.0, index=monthly.columns)  # start in cash
    out, weights_hist = [], {}
    for t, nxt in zip(dates[:-1], dates[1:]):
        picks = alpha.loc[t].dropna().nlargest(top_n).index
        w = pd.Series(0.0, index=monthly.columns)
        w[picks] = 1.0 / len(picks)
        traded = (w - prev_w).abs().sum()
        gross = float((w * rets.loc[nxt].fillna(0.0)).sum())
        net = gross - traded * cost_bps / 1e4
        out.append({"date": nxt, "gross": gross, "net": net,
                    "turnover": traded / 2.0, "benchmark": float(bench_rets.loc[nxt])})
        weights_hist[t] = w[w > 0]
        # weights drift with returns before the next rebalance
        drifted = w * (1.0 + rets.loc[nxt].fillna(0.0))
        prev_w = drifted / drifted.sum()
    df = pd.DataFrame(out).set_index("date")
    return {"returns": df, "holdings": weights_hist, "metrics": performance_metrics(df)}


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def rolling_returns(returns: pd.Series, window: int = 12) -> pd.Series:
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def _stats(r: pd.Series, rf: float) -> dict:
    n = len(r)
    cum = float((1 + r).prod() - 1)
    ann = (1 + cum) ** (PERIODS_PER_YEAR / n) - 1 if n else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if n > 1 else np.nan
    sharpe = (r.mean() * PERIODS_PER_YEAR - rf) / vol if vol and vol > 0 else np.nan
    return {"Cumulative return": cum, "Annualized return": ann, "Annualized volatility": vol,
            "Sharpe ratio": float(sharpe), "Max drawdown": float(drawdown(r).min())}


def performance_metrics(df: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """Strategy (net of costs) vs benchmark. Turnover is average annual one-way turnover."""
    strat = _stats(df["net"], rf)
    strat["Annual turnover"] = float(df["turnover"].mean() * PERIODS_PER_YEAR)
    bench = _stats(df["benchmark"], rf)
    bench["Annual turnover"] = np.nan
    table = pd.DataFrame({"Strategy": strat, "S&P 500": bench})
    table.loc["Excess return (annualized)"] = [
        strat["Annualized return"] - bench["Annualized return"], np.nan]
    return table
