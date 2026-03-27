from pathlib import Path
import math

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="VN Stock Dashboard",
    page_icon="📈",
    layout="wide",
)

# =========================
# CONFIG
# =========================
SYMBOLS = ["ACB", "FPT", "HPG"]
DAILY_DIR = Path("data/daily")
INTRADAY_DIR = Path("data/intraday")

# =========================
# UI STYLE
# =========================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .subtle {
        color: #9aa4b2;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.7rem 0.8rem;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HELPERS
# =========================
def latest_intraday_file(symbol: str):
    if not INTRADAY_DIR.exists():
        return None, None

    folders = sorted([p for p in INTRADAY_DIR.iterdir() if p.is_dir()], reverse=True)
    for folder in folders:
        file_path = folder / f"{symbol.lower()}_intraday.csv"
        if file_path.exists():
            return file_path, folder.name
    return None, None


def normalize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def enrich_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "date" in df.columns:
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    if "close" in df.columns:
        if "return_1d" not in df.columns:
            df["return_1d"] = df["close"].pct_change()

        if "ma_5" not in df.columns:
            df["ma_5"] = df["close"].rolling(5).mean()
        if "ma_20" not in df.columns:
            df["ma_20"] = df["close"].rolling(20).mean()
        if "ma_50" not in df.columns:
            df["ma_50"] = df["close"].rolling(50).mean()

        df["rolling_volatility_20d"] = df["return_1d"].rolling(20).std() * math.sqrt(252) * 100
        df["cum_max_close"] = df["close"].cummax()
        df["drawdown"] = df["close"] / df["cum_max_close"] - 1
        df["distance_to_52w_high"] = df["close"] / df["close"].rolling(252, min_periods=1).max() - 1

        if "month" in df.columns:
            monthly_return_map = (
                df.groupby("month")["close"]
                .agg(first_close="first", last_close="last")
                .assign(monthly_return=lambda x: x["last_close"] / x["first_close"] - 1)["monthly_return"]
            )
            df["monthly_return"] = df["month"].map(monthly_return_map)
        else:
            df["monthly_return"] = pd.NA

    if "volume" in df.columns:
        df["avg_volume_20d"] = df["volume"].rolling(20).mean()
        df["volume_ratio_20d"] = df["volume"] / df["avg_volume_20d"]

    return df


@st.cache_data(ttl=300)
def load_daily(symbol: str) -> pd.DataFrame:
    file_path = DAILY_DIR / f"{symbol.lower()}_daily.csv"
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns:
        date_col = next((c for c in ["time", "datetime"] if c in df.columns), None)
        if date_col is None:
            raise ValueError(f"Không tìm thấy cột ngày cho {symbol}. Columns: {df.columns.tolist()}")
        df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    df = normalize_numeric(
        df,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ma_5",
            "ma_20",
            "ma_50",
            "return_1d",
        ],
    )

    return enrich_daily(df)


@st.cache_data(ttl=300)
def load_intraday(symbol: str):
    file_path, folder_name = latest_intraday_file(symbol)
    if file_path is None:
        return pd.DataFrame(), None

    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "snapshot_time" in df.columns:
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")

    df = normalize_numeric(
        df,
        [
            "match_price",
            "last_price",
            "price",
            "close",
            "change",
            "price_change",
            "pct_change",
            "change_percent",
            "price_change_percent",
            "volume",
            "match_volume",
            "total_volume",
        ],
    )
    return df, folder_name


def choose_intraday_price_col(df: pd.DataFrame):
    for c in ["match_price", "last_price", "price", "close"]:
        if c in df.columns:
            return c
    return None


def choose_intraday_pct_col(df: pd.DataFrame):
    for c in ["pct_change", "change_percent", "price_change_percent"]:
        if c in df.columns:
            return c
    return None


def choose_intraday_vol_col(df: pd.DataFrame):
    for c in ["match_volume", "volume", "total_volume"]:
        if c in df.columns:
            return c
    return None


def apply_date_filter(df: pd.DataFrame, preset: str, custom_range):
    max_date = df["date"].max().date()

    if preset == "1M":
        start_date = (df["date"].max() - pd.Timedelta(days=30)).date()
    elif preset == "3M":
        start_date = (df["date"].max() - pd.Timedelta(days=90)).date()
    elif preset == "6M":
        start_date = (df["date"].max() - pd.Timedelta(days=180)).date()
    elif preset == "YTD":
        start_date = pd.Timestamp(year=df["date"].max().year, month=1, day=1).date()
    elif preset == "1Y":
        start_date = (df["date"].max() - pd.Timedelta(days=365)).date()
    elif preset == "Custom":
        start_date, max_date = custom_range
    else:
        start_date = df["date"].min().date()

    out = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= max_date)].copy()
    return out, start_date, max_date


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:.2f}%"


def fmt_pct_value(x):
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.2f}%"


def fmt_num(x, digits=2):
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"


def calc_return(df: pd.DataFrame, periods: int):
    if len(df) <= periods:
        return None
    last = df["close"].iloc[-1]
    prev = df["close"].iloc[-(periods + 1)]
    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return None
    return last / prev - 1


def base_layout(fig: go.Figure, height: int):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(automargin=True, tickformat="%b\n%Y", nticks=8, showgrid=False)
    fig.update_yaxes(automargin=True)
    return fig


def build_price_chart(df: pd.DataFrame, chart_type: str, show_ma5: bool, show_ma20: bool, show_ma50: bool):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
    )

    if chart_type == "Candlestick" and all(c in df.columns for c in ["open", "high", "low", "close"]):
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )

    if show_ma5 and "ma_5" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_5"], mode="lines", name="MA 5"), row=1, col=1)
    if show_ma20 and "ma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_20"], mode="lines", name="MA 20"), row=1, col=1)
    if show_ma50 and "ma_50" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_50"], mode="lines", name="MA 50"), row=1, col=1)

    if "volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Volume",
                opacity=0.65,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return base_layout(fig, 720)


def build_return_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["return_1d"] * 100,
            name="Daily Return (%)",
        )
    )
    fig.add_hline(y=0)
    fig.update_yaxes(title_text="Return (%)")
    return base_layout(fig, 320)


def build_volatility_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rolling_volatility_20d"],
            mode="lines",
            name="20D Rolling Volatility",
            line=dict(width=2),
        )
    )
    fig.update_yaxes(title_text="Volatility (%)")
    return base_layout(fig, 320)


def build_drawdown_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["drawdown"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
        )
    )
    fig.add_hline(y=0)
    fig.update_yaxes(title_text="Drawdown (%)")
    return base_layout(fig, 320)


def build_monthly_return_chart(df: pd.DataFrame):
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["monthly_return"]
        .last()
        .dropna()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["monthly_return"] * 100,
            name="Monthly Return",
        )
    )
    fig.add_hline(y=0)
    fig.update_yaxes(title_text="Return (%)")
    fig = base_layout(fig, 320)
    fig.update_xaxes(tickangle=-35, nticks=8)
    return fig


def build_intraday_chart(df: pd.DataFrame):
    price_col = choose_intraday_price_col(df)
    vol_col = choose_intraday_vol_col(df)

    fig = make_subplots(
        rows=2 if vol_col else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28] if vol_col else [1.0],
    )

    fig.add_trace(
        go.Scatter(
            x=df["snapshot_time"],
            y=df[price_col],
            mode="lines+markers",
            name="Price",
        ),
        row=1,
        col=1,
    )

    if vol_col:
        fig.add_trace(
            go.Bar(
                x=df["snapshot_time"],
                y=df[vol_col],
                name="Volume",
                opacity=0.55,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(tickformat="%H:%M", nticks=10, automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        height=540 if vol_col else 400,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    )
    return fig


def build_compare_chart(symbols: list[str], preset: str, custom_range):
    fig = go.Figure()

    for symbol in symbols:
        df = load_daily(symbol)
        filtered, _, _ = apply_date_filter(df, preset, custom_range)
        filtered = filtered.dropna(subset=["close"]).copy()
        if filtered.empty:
            continue

        base = filtered["close"].iloc[0]
        if pd.isna(base) or base == 0:
            continue

        filtered["indexed"] = filtered["close"] / base * 100
        fig.add_trace(
            go.Scatter(
                x=filtered["date"],
                y=filtered["indexed"],
                mode="lines",
                name=symbol,
            )
        )

    fig.update_yaxes(title_text="Indexed Price")
    return base_layout(fig, 430)


def build_correlation_heatmap(symbols: list[str], preset: str, custom_range):
    merged = None
    for symbol in symbols:
        df = load_daily(symbol)
        filtered, _, _ = apply_date_filter(df, preset, custom_range)
        temp = filtered[["date", "return_1d"]].rename(columns={"return_1d": symbol})
        merged = temp if merged is None else merged.merge(temp, on="date", how="outer")

    if merged is None or merged.empty:
        return go.Figure()

    corr = merged.drop(columns=["date"]).corr()
    labels = list(corr.columns)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 12},
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(side="bottom", tickangle=0, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def technical_snapshot(df: pd.DataFrame):
    last = df.iloc[-1]
    rows = []

    if "close" in df.columns and "ma_20" in df.columns:
        rows.append(
            {
                "Metric": "Close vs MA20",
                "Value": fmt_pct(last["close"] / last["ma_20"] - 1)
                if pd.notna(last["ma_20"]) and last["ma_20"] != 0
                else "-",
            }
        )

    if "close" in df.columns and "ma_50" in df.columns:
        rows.append(
            {
                "Metric": "Close vs MA50",
                "Value": fmt_pct(last["close"] / last["ma_50"] - 1)
                if pd.notna(last["ma_50"]) and last["ma_50"] != 0
                else "-",
            }
        )

    if "rolling_volatility_20d" in df.columns:
        rows.append({"Metric": "20D Volatility", "Value": fmt_pct_value(last["rolling_volatility_20d"])})

    high_52w = df["close"].tail(252).max() if "close" in df.columns else None
    low_52w = df["close"].tail(252).min() if "close" in df.columns else None
    max_drawdown = df["drawdown"].min() if "drawdown" in df.columns else None

    rows.append({"Metric": "52W High", "Value": fmt_num(high_52w)})
    rows.append({"Metric": "52W Low", "Value": fmt_num(low_52w)})
    rows.append({"Metric": "Max Drawdown", "Value": fmt_pct(max_drawdown)})

    return pd.DataFrame(rows)


# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## Bộ lọc")

selected_symbol = st.sidebar.selectbox("Chọn mã", SYMBOLS, index=0)

st.sidebar.markdown("### Thời gian")
preset = st.sidebar.radio(
    "Khoảng thời gian",
    ["1M", "3M", "6M", "YTD", "1Y", "All", "Custom"],
    index=2,
)

selected_daily = load_daily(selected_symbol)
min_date = selected_daily["date"].min().date()
max_date = selected_daily["date"].max().date()

default_start = max(min_date, (selected_daily["date"].max() - pd.Timedelta(days=180)).date())
custom_range = st.sidebar.date_input(
    "Custom range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
    disabled=(preset != "Custom"),
)

st.sidebar.markdown("### Tùy chọn chart")
chart_type = st.sidebar.radio("Kiểu chart giá", ["Candlestick", "Line"], index=0)
show_ma5 = st.sidebar.checkbox("Hiện MA 5", value=False)
show_ma20 = st.sidebar.checkbox("Hiện MA 20", value=True)
show_ma50 = st.sidebar.checkbox("Hiện MA 50", value=True)

st.sidebar.markdown("### So sánh")
compare_symbols = st.sidebar.multiselect(
    "Chọn mã để compare",
    SYMBOLS,
    default=SYMBOLS,
)

# =========================
# DATA FILTER
# =========================
daily_df = load_daily(selected_symbol)
daily_filtered, filter_start, filter_end = apply_date_filter(daily_df, preset, custom_range)

intraday_df, intraday_folder = load_intraday(selected_symbol)

# =========================
# HEADER
# =========================
st.title("📈 VN Stock Dashboard")
st.caption(f"Mã đang xem: {selected_symbol} | Daily range: {filter_start} → {filter_end}")


# =========================
# LIVE PANEL
# =========================
@st.fragment(run_every="5m")
def render_dashboard():
    latest_close = daily_filtered["close"].iloc[-1] if "close" in daily_filtered.columns and not daily_filtered.empty else None
    ret_1d = calc_return(daily_filtered, 1)
    ret_5d = calc_return(daily_filtered, 5)
    ret_20d = calc_return(daily_filtered, 20)
    period_ret = (daily_filtered["close"].iloc[-1] / daily_filtered["close"].iloc[0] - 1) if len(daily_filtered) >= 2 else None
    avg_volume = daily_filtered["volume"].tail(20).mean() if "volume" in daily_filtered.columns else None
    rolling_vol = daily_filtered["rolling_volatility_20d"].iloc[-1] if "rolling_volatility_20d" in daily_filtered.columns else None
    max_dd = daily_filtered["drawdown"].min() if "drawdown" in daily_filtered.columns else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Latest Close", fmt_num(latest_close))
    c2.metric("1D Return", fmt_pct(ret_1d))
    c3.metric("5D Return", fmt_pct(ret_5d))
    c4.metric("20D Return", fmt_pct(ret_20d))
    c5.metric("Period Return", fmt_pct(period_ret))
    c6.metric("Avg Vol (20D)", fmt_num(avg_volume, 0))

    c7, c8 = st.columns(2)
    c7.metric("20D Volatility", fmt_pct_value(rolling_vol))
    c8.metric("Max Drawdown", fmt_pct(max_dd))

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Intraday", "Compare", "Data"])

    with tab1:
        st.markdown("### Price & Volume")
        st.plotly_chart(
            build_price_chart(daily_filtered, chart_type, show_ma5, show_ma20, show_ma50),
            use_container_width=True,
        )

        row1_left, row1_right = st.columns(2)
        with row1_left:
            st.markdown("### Daily Return")
            st.plotly_chart(build_return_chart(daily_filtered), use_container_width=True)
        with row1_right:
            st.markdown("### Rolling Volatility (20D)")
            st.plotly_chart(build_volatility_chart(daily_filtered), use_container_width=True)

        row2_left, row2_right = st.columns(2)
        with row2_left:
            st.markdown("### Drawdown")
            st.plotly_chart(build_drawdown_chart(daily_filtered), use_container_width=True)
        with row2_right:
            st.markdown("### Monthly Return")
            st.plotly_chart(build_monthly_return_chart(daily_filtered), use_container_width=True)

        row3_left, row3_right = st.columns(2)
        with row3_left:
            st.markdown("### Snapshot")
            snapshot_df = technical_snapshot(daily_filtered)
            st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

        with row3_right:
            st.markdown("### Period Summary")
            summary_df = pd.DataFrame(
                [
                    {"Metric": "Period Return", "Value": fmt_pct(period_ret)},
                    {
                        "Metric": "Period High",
                        "Value": fmt_num(daily_filtered["high"].max()) if "high" in daily_filtered.columns else "-",
                    },
                    {
                        "Metric": "Period Low",
                        "Value": fmt_num(daily_filtered["low"].min()) if "low" in daily_filtered.columns else "-",
                    },
                    {"Metric": "Trading Days", "Value": len(daily_filtered)},
                    {
                        "Metric": "Latest Vol Ratio vs 20D",
                        "Value": fmt_num(daily_filtered["volume_ratio_20d"].iloc[-1], 2)
                        if "volume_ratio_20d" in daily_filtered.columns
                        else "-",
                    },
                ]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Intraday View")
        if intraday_df.empty:
            st.info("Chưa có dữ liệu intraday.")
        else:
            st.caption(f"File intraday mới nhất: {intraday_folder}")

            col_a, col_b, col_c = st.columns(3)
            intraday_price_col = choose_intraday_price_col(intraday_df)
            intraday_pct_col = choose_intraday_pct_col(intraday_df)
            intraday_vol_col = choose_intraday_vol_col(intraday_df)

            last_intraday_price = intraday_df[intraday_price_col].iloc[-1] if intraday_price_col else None
            last_intraday_pct = (
                intraday_df[intraday_pct_col].iloc[-1] / 100 if intraday_pct_col and intraday_pct_col in intraday_df.columns else None
            )
            last_intraday_time = intraday_df["snapshot_time"].iloc[-1] if "snapshot_time" in intraday_df.columns else None
            last_intraday_volume = intraday_df[intraday_vol_col].iloc[-1] if intraday_vol_col else None

            col_a.metric("Latest Intraday Price", fmt_num(last_intraday_price))
            col_b.metric("Intraday Change %", fmt_pct(last_intraday_pct))
            col_c.metric("Last Snapshot", str(last_intraday_time))

            if last_intraday_volume is not None:
                st.metric("Latest Intraday Volume", fmt_num(last_intraday_volume, 0))

            st.plotly_chart(build_intraday_chart(intraday_df), use_container_width=True)
            st.dataframe(intraday_df.tail(30), use_container_width=True)

    with tab3:
        st.markdown("### Relative Performance")
        if compare_symbols:
            st.plotly_chart(build_compare_chart(compare_symbols, preset, custom_range), use_container_width=True)
        else:
            st.warning("Hãy chọn ít nhất 1 mã để compare.")

        compare_col_left, compare_col_right = st.columns([1.3, 1])

        with compare_col_left:
            compare_rows = []
            for symbol in compare_symbols:
                temp = load_daily(symbol)
                temp_filtered, _, _ = apply_date_filter(temp, preset, custom_range)
                if temp_filtered.empty:
                    continue

                compare_rows.append(
                    {
                        "Symbol": symbol,
                        "Latest Close": temp_filtered["close"].iloc[-1] if "close" in temp_filtered.columns else None,
                        "1D Return": calc_return(temp_filtered, 1),
                        "5D Return": calc_return(temp_filtered, 5),
                        "20D Return": calc_return(temp_filtered, 20),
                        "Period Return": (temp_filtered["close"].iloc[-1] / temp_filtered["close"].iloc[0] - 1)
                        if len(temp_filtered) >= 2
                        else None,
                        "20D Vol": temp_filtered["rolling_volatility_20d"].iloc[-1]
                        if "rolling_volatility_20d" in temp_filtered.columns
                        else None,
                    }
                )

            compare_df = pd.DataFrame(compare_rows)
            if not compare_df.empty:
                for c in ["1D Return", "5D Return", "20D Return", "Period Return"]:
                    if c in compare_df.columns:
                        compare_df[c] = compare_df[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
                if "20D Vol" in compare_df.columns:
                    compare_df["20D Vol"] = compare_df["20D Vol"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
                if "Latest Close" in compare_df.columns:
                    compare_df["Latest Close"] = compare_df["Latest Close"].map(
                        lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
                    )
                st.markdown("### Compare Summary")
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

        with compare_col_right:
            if len(compare_symbols) >= 2:
                st.markdown("### Return Correlation")
                st.plotly_chart(build_correlation_heatmap(compare_symbols, preset, custom_range), use_container_width=True)

    with tab4:
        st.markdown("### Daily Data")
        st.dataframe(daily_filtered, use_container_width=True)

        csv_daily = daily_filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download daily CSV",
            csv_daily,
            file_name=f"{selected_symbol.lower()}_daily_filtered.csv",
            mime="text/csv",
        )

        st.markdown("### Intraday Data")
        if intraday_df.empty:
            st.info("Chưa có dữ liệu intraday để tải.")
        else:
            st.dataframe(intraday_df, use_container_width=True)
            csv_intraday = intraday_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download intraday CSV",
                csv_intraday,
                file_name=f"{selected_symbol.lower()}_intraday_latest.csv",
                mime="text/csv",
            )


render_dashboard()
