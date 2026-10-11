"""Alpha Signal Research Dashboard (educational use only)."""
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import drawdown, rolling_returns, run_backtest
from data import BENCHMARK, load_data
from signals import compute_signals

st.set_page_config(page_title="Alpha Signal Research Dashboard", layout="wide")

STRAT_COLOR, BENCH_COLOR = "#1f77b4", "#7f7f7f"
PCT_ROWS = ["Cumulative return", "Annualized return", "Annualized volatility",
            "Max drawdown", "Annual turnover", "Excess return (annualized)"]


@st.cache_data(show_spinner="Loading data...", ttl=24 * 3600)
def get_data(use_sample: bool):
    return load_data(use_sample)


def line_fig(series: dict, title: str, yfmt: str = ".0%") -> go.Figure:
    fig = go.Figure()
    for name, (s, color) in series.items():
        fig.add_scatter(x=s.index, y=s.values, name=name, line=dict(color=color, width=2))
    fig.update_layout(title=title, yaxis_tickformat=yfmt, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    return fig


def fmt_metrics(table: pd.DataFrame) -> pd.DataFrame:
    out = table.astype(object).copy()
    for row in out.index:
        for col in out.columns:
            v = table.loc[row, col]
            out.loc[row, col] = "–" if pd.isna(v) else (f"{v:.2f}" if row == "Sharpe ratio" else f"{v:.1%}")
    return out


st.title("Alpha Signal Research Dashboard")
st.warning("Educational project only. Not investment advice. Backtested results are hypothetical "
           "and do not predict future performance.")

with st.sidebar:
    st.header("Settings")
    use_sample = st.toggle("Use sample data (offline)", value=False,
                           help="Deterministic synthetic prices; no network access needed.")
    top_n = st.slider("Stocks held", 3, 10, 5)
    cost_bps = st.slider("Transaction cost (bps per unit traded)", 0, 50, 10)
    st.subheader("Signal weights")
    w = {"momentum": st.slider("6-month momentum", 0.0, 2.0, 1.0, 0.1),
         "volatility": st.slider("Low 20-day volatility", 0.0, 2.0, 1.0, 0.1),
         "value": st.slider("Value (book-to-price)", 0.0, 2.0, 1.0, 0.1)}
    window = st.slider("Rolling window (months)", 3, 36, 12)

if sum(w.values()) == 0:
    st.error("Set at least one signal weight above zero.")
    st.stop()

daily, book_value, source = get_data(use_sample)
if source.startswith("sample"):
    st.info(f"Data source: **{source}** – synthetic prices, not real market data.")
else:
    st.caption(f"Data source: {source} (yfinance). Book value is the latest snapshot held constant, "
               "so the value signal contains look-ahead bias in live mode; see README.")

stocks = [c for c in daily.columns if c != BENCHMARK]
sig = compute_signals(daily[stocks], book_value.reindex(columns=stocks), w)
monthly = sig["monthly"]
bench = daily[BENCHMARK].reindex(monthly.index)
res = run_backtest(monthly, sig["alpha"], bench, top_n=top_n, cost_bps=cost_bps)
ret, metrics = res["returns"], res["metrics"]

# Latest rankings
last = sig["alpha"].dropna(how="all").index[-1]
rank = pd.DataFrame({
    "Alpha": sig["alpha"].loc[last],
    "6M momentum": sig["raw"]["momentum"].loc[last],
    "20D volatility": sig["raw"]["volatility"].loc[last],
    "Book-to-price": sig["raw"]["value"].loc[last],
    "Z momentum": sig["z"]["momentum"].loc[last],
    "Z low-vol": sig["z"]["volatility"].loc[last],
    "Z value": sig["z"]["value"].loc[last],
}).sort_values("Alpha", ascending=False)
rank.insert(0, "Rank", range(1, len(rank) + 1))
rank["Selected"] = rank["Rank"] <= top_n
rank.index.name = "Ticker"

m = metrics["Strategy"]
c = st.columns(5)
c[0].metric("Annualized return", f"{m['Annualized return']:.1%}",
            f"{metrics.loc['Excess return (annualized)', 'Strategy']:+.1%} vs S&P 500")
c[1].metric("Sharpe ratio", f"{m['Sharpe ratio']:.2f}")
c[2].metric("Volatility", f"{m['Annualized volatility']:.1%}")
c[3].metric("Max drawdown", f"{m['Max drawdown']:.1%}")
c[4].metric("Annual turnover", f"{m['Annual turnover']:.0%}")

tab_rank, tab_perf, tab_dd, tab_roll = st.tabs(["Rankings", "Performance", "Drawdown", "Rolling returns"])

with tab_rank:
    st.subheader(f"Latest rankings (as of {last:%Y-%m-%d})")
    fig = go.Figure(go.Bar(x=rank.index, y=rank["Alpha"],
                           marker_color=[STRAT_COLOR if s else BENCH_COLOR for s in rank["Selected"]]))
    fig.update_layout(title="Alpha score (blue = selected)", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(rank.style.format({"Alpha": "{:.2f}", "6M momentum": "{:.1%}", "20D volatility": "{:.1%}",
                                    "Book-to-price": "{:.3f}", "Z momentum": "{:.2f}",
                                    "Z low-vol": "{:.2f}", "Z value": "{:.2f}"}),
                 width="stretch")

with tab_perf:
    growth = {"Strategy (net of costs)": ((1 + ret["net"]).cumprod() - 1, STRAT_COLOR),
              "S&P 500": ((1 + ret["benchmark"]).cumprod() - 1, BENCH_COLOR)}
    st.plotly_chart(line_fig(growth, "Cumulative return"), width="stretch")
    st.dataframe(fmt_metrics(metrics), width="stretch")

with tab_dd:
    dd = {"Strategy": (drawdown(ret["net"]), STRAT_COLOR), "S&P 500": (drawdown(ret["benchmark"]), BENCH_COLOR)}
    st.plotly_chart(line_fig(dd, "Drawdown"), width="stretch")

with tab_roll:
    rr = {"Strategy": (rolling_returns(ret["net"], window), STRAT_COLOR),
          "S&P 500": (rolling_returns(ret["benchmark"], window), BENCH_COLOR)}
    st.plotly_chart(line_fig(rr, f"Rolling {window}-month return"), width="stretch")

# CSV export: latest rankings + performance summary in one file
buf = io.StringIO()
buf.write("# Educational use only. Not investment advice.\n")
buf.write(f"# Latest rankings as of {last:%Y-%m-%d} (data source: {source})\n")
rank.to_csv(buf)
buf.write("\n# Performance summary (net of costs)\n")
metrics.to_csv(buf, index_label="Metric")
st.download_button("Export rankings + performance summary (CSV)", buf.getvalue(),
                   file_name="alpha_signal_export.csv", mime="text/csv")
