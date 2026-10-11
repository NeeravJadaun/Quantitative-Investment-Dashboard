import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import run_backtest  # noqa: E402
from data import sample_data  # noqa: E402
from signals import compute_signals, momentum, zscore  # noqa: E402


def test_momentum_and_zscore():
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    prices = pd.DataFrame({"A": [100.0] * 6 + [110.0, 120.0], "B": [100.0] * 8,
                           "C": [100.0] * 6 + [90.0, 80.0]}, index=idx)
    mom = momentum(prices, months=6)
    assert mom.iloc[-1]["A"] == pytest.approx(0.20)
    assert mom.iloc[-1]["C"] == pytest.approx(-0.20)
    z = zscore(mom.dropna())
    assert z.mean(axis=1).abs().max() < 1e-12
    assert z.iloc[-1]["A"] > 0 > z.iloc[-1]["C"]


def test_no_lookahead():
    """Signals at a date must not change if later prices are altered."""
    prices, bv = sample_data()
    cutoff = prices.index[1500]
    base = compute_signals(prices, bv)["alpha"]
    tampered = prices.copy()
    tampered.loc[tampered.index > cutoff] *= 3.0
    alt = compute_signals(tampered, bv)["alpha"]
    early = base.index[base.index <= cutoff][:-1]  # last month-end may include cutoff day only
    pd.testing.assert_frame_equal(base.loc[early], alt.loc[early])


def test_backtest_returns():
    idx = pd.date_range("2021-01-31", periods=4, freq="ME")
    monthly = pd.DataFrame({"A": [100, 110, 121, 121.0], "B": [100, 100, 100, 110.0]}, index=idx)
    bench = pd.Series([100, 101, 102, 103.0], index=idx)
    # Picks A, A, then switches to B. Each held month returns +10%.
    alpha = pd.DataFrame({"A": [2.0, 2.0, 0.0, 0.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    r = run_backtest(monthly, alpha, bench, top_n=1, cost_bps=0)["returns"]
    assert r["net"].tolist() == pytest.approx([0.10, 0.10, 0.10])
    # Costs: 100% traded entering A (10 bps), nothing traded next month, full switch A->B (20 bps).
    c = run_backtest(monthly, alpha, bench, top_n=1, cost_bps=10)["returns"]
    assert c["net"].tolist() == pytest.approx([0.099, 0.10, 0.098])
    assert c["turnover"].tolist() == pytest.approx([0.5 * 1, 0.0, 1.0])
