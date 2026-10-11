"""Price/fundamental loading: yfinance with an on-disk cache and a deterministic sample fallback."""
import time
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "XOM", "PG",
           "V", "UNH", "HD", "KO", "PEP", "WMT", "BAC", "CVX", "MRK", "ABBV",
           "COST", "CSCO", "ORCL", "MCD", "TXN"]
BENCHMARK = "^GSPC"
CACHE_DIR = Path(".cache")
CACHE_TTL_SECONDS = 24 * 3600
START = "2014-01-01"


def sample_data(seed: int = 1, years: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic but deterministic prices (market factor + idiosyncratic noise) and
    quarterly book value per share. Not real market data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2025-12-31", periods=252 * years)
    n = len(TICKERS)
    mkt = rng.normal(0.08 / 252, 0.008, len(idx))
    beta = rng.uniform(0.6, 1.4, n)
    drift = rng.uniform(0.0, 0.10, n) / 252
    idio = rng.normal(0, 0.012, (len(idx), n))
    rets = mkt[:, None] * beta + drift + idio
    start_px = rng.uniform(40, 300, n)
    prices = pd.DataFrame(start_px * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=TICKERS)
    prices[BENCHMARK] = 2000 * np.exp(np.cumsum(mkt))
    # Book value/share: random-walk growth, reported quarterly with a 60-day publication lag.
    q = pd.date_range(idx[0], idx[-1], freq="QE")
    growth = np.cumprod(1 + rng.normal(0.015, 0.02, (len(q), n)), axis=0)
    bv = pd.DataFrame(start_px * rng.uniform(0.05, 0.4, n) * growth, index=q, columns=TICKERS)
    bv.index = bv.index + pd.Timedelta(days=60)
    return prices, bv


def _download(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    px = yf.download(tickers + [BENCHMARK], start=START, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all").dropna(axis=1, thresh=int(len(px) * 0.8))
    bv = {}
    for t in tickers:
        try:
            bv[t] = yf.Ticker(t).info.get("bookValue")
        except Exception:
            bv[t] = None
    # yfinance only exposes the latest book value, so it is held constant (see README limitations).
    bv_df = pd.DataFrame({t: [v] for t, v in bv.items()}, index=[px.index[0]], dtype=float)
    return px, bv_df


def load_data(use_sample: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (daily prices incl. benchmark, book value per share, source label)."""
    if use_sample:
        return (*sample_data(), "sample")
    CACHE_DIR.mkdir(exist_ok=True)
    px_f, bv_f = CACHE_DIR / "prices.csv", CACHE_DIR / "book_value.csv"
    if px_f.exists() and bv_f.exists() and time.time() - px_f.stat().st_mtime < CACHE_TTL_SECONDS:
        return (pd.read_csv(px_f, index_col=0, parse_dates=True),
                pd.read_csv(bv_f, index_col=0, parse_dates=True), "live (cached)")
    try:
        px, bv = _download(TICKERS)
        if px.empty or BENCHMARK not in px:
            raise ValueError("empty download")
        px.to_csv(px_f)
        bv.to_csv(bv_f)
        return px, bv, "live"
    except Exception:
        if px_f.exists() and bv_f.exists():  # stale cache beats nothing
            return (pd.read_csv(px_f, index_col=0, parse_dates=True),
                    pd.read_csv(bv_f, index_col=0, parse_dates=True), "live (stale cache)")
        return (*sample_data(), "sample (download failed)")
