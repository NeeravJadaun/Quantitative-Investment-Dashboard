"""Signal construction: momentum, volatility, value, z-scores and the combined alpha.

All functions take price data indexed by date and return values observed *at* each
month-end using only data up to and including that date.
"""
import numpy as np
import pandas as pd

MOMENTUM_MONTHS = 6
VOL_WINDOW_DAYS = 20
TRADING_DAYS = 252


def month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Last available price in each calendar month (indexed by the actual last trading day)."""
    daily = daily.sort_index()
    return daily.groupby(daily.index.to_period("M")).tail(1)


def momentum(monthly: pd.DataFrame, months: int = MOMENTUM_MONTHS) -> pd.DataFrame:
    """Trailing total return over `months` month-ends."""
    return monthly / monthly.shift(months) - 1.0


def volatility(daily: pd.DataFrame, month_ends: pd.Index, window: int = VOL_WINDOW_DAYS) -> pd.DataFrame:
    """Annualised stdev of the last `window` daily returns, sampled at month-ends."""
    rets = daily.sort_index().pct_change()
    vol = rets.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)
    return vol.reindex(month_ends)


def value_score(monthly: pd.DataFrame, book_value: pd.DataFrame) -> pd.DataFrame:
    """Book-to-price (inverse P/B). `book_value` is book value per share, already
    lagged so each row only holds figures that were public on that date."""
    bv = book_value.reindex(monthly.index.union(book_value.index)).sort_index().ffill()
    bv = bv.reindex(monthly.index)
    bv = bv.where(bv > 0)  # negative book value makes P/B meaningless
    return bv / monthly


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: standardise each row (date) across stocks."""
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def compute_signals(daily: pd.DataFrame, book_value: pd.DataFrame,
                    weights: dict | None = None) -> dict:
    """Return raw signals, z-scores and the combined alpha (higher = more attractive).

    Direction: high momentum is good, LOW volatility is good, high book-to-price is good.
    Missing z-scores (e.g. no fundamentals) are treated as neutral (0).
    """
    weights = weights or {"momentum": 1.0, "volatility": 1.0, "value": 1.0}
    monthly = month_end_prices(daily)
    raw = {
        "momentum": momentum(monthly),
        "volatility": volatility(daily, monthly.index),
        "value": value_score(monthly, book_value),
    }
    z = {"momentum": zscore(raw["momentum"]),
         "volatility": -zscore(raw["volatility"]),
         "value": zscore(raw["value"])}
    total_w = sum(weights.values()) or 1.0
    alpha = sum(weights[k] * z[k].fillna(0.0) for k in z) / total_w
    # No alpha until the slowest signal (momentum) has enough history.
    alpha = alpha.where(raw["momentum"].notna() & raw["volatility"].notna())
    return {"monthly": monthly, "raw": raw, "z": z, "alpha": alpha}
