from pathlib import Path
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
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    .metric-card {
        background: rgba(255,255,255,0.03);
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
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
        ["open", "high", "low", "close", "volume", "ma_5", "ma_20", "ma_50", "return_1d"]
    )

    if "return_1d" not in df.columns and "close" in df.columns:
        df["return_1d"] = df["close"].pct_change()

    if "ma_5" not in df.columns and "close" in df.columns:
        df["ma_5"] = df["close"].rolling(5).mean()

    if "ma_20" not in df.columns and "close" in df.columns:
        df["ma_20"] = df["close"].rolling(20).mean()

    if "ma_50" not in df.columns and "close" in df.columns:
        df["ma_50"] = df["close"].rolling(50).mean()

    return df


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
        ["match_price", "last_price", "price", "close", "change", "price_change",
         "pct_change", "change_percent", "price_change_percent", "volume", "match_volume"]
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


def build_price_chart(df: pd.DataFrame, chart_type: str, show_ma5: bool, show_ma20: bool, show_ma50: bool):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28]
    )

    if chart_type == "Candlestick" and all(c in df.columns for c in ["open", "high", "low", "close"]):
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC"
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(width=2)
            ),
            row=1, col=1
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
                opacity=0.6
            ),
            row=2, col=1
        )

    fig.update_layout(
        title="Price & Volume",
        height=680,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def build_return_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["return_1d"] * 100,
            name="Daily Return (%)"
        )
    )
    fig.update_layout(
        title="Daily Return (%)",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Return (%)")
    return fig


def build_intraday_chart(df: pd.DataFrame):
    price_col = choose_intraday_price_col(df)
    vol_col = choose_intraday_vol_col(df)

    fig = make_subplots(
        rows=2 if vol_col else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28] if vol_col else [1.0]
    )

    fig.add_trace(
        go.Scatter(
            x=df["snapshot_time"],
            y=df[price_col],
            mode="lines+markers",
            name="Price"
        ),
        row=1, col=1
    )

    if vol_col:
        fig.add_trace(
            go.Bar(
                x=df["snapshot_time"],
                y=df[vol_col],
                name="Volume",
                opacity=0.5
            ),
            row=2, col=1
        )

    fig.update_layout(
        title="Intraday Snapshot",
        height=520 if vol_col else 380,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
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
                name=symbol
            )
        )

    fig.update_layout(
        title="Relative Performance (Base = 100)",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Indexed Price")
    return fig


def technical_snapshot(df: pd.DataFrame):
    last = df.iloc[-1]
    rows = []

    if "close" in df.columns and "ma_20" in df.columns:
        rows.append({
            "Metric": "Close vs MA20",
            "Value": fmt_pct(last["close"] / last["ma_20"] - 1) if pd.notna(last["ma_20"]) and last["ma_20"] != 0 else "-"
        })

    if "close" in df.columns and "ma_50" in df.columns:
        rows.append({
            "Metric": "Close vs MA50",
            "Value": fmt_pct(last["close"] / last["ma_50"] - 1) if pd.notna(last["ma_50"]) and last["ma_50"] != 0 else "-"
        })

    high_52w = df["close"].tail(252).max() if "close" in df.columns else None
    low_52w = df["close"].tail(252).min() if "close" in df.columns else None

    rows.append({"Metric": "52W High", "Value": fmt_num(high_52w)})
    rows.append({"Metric": "52W Low", "Value": fmt_num(low_52w)})

    return pd.DataFrame(rows)


# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## Bộ lọc")

selected_symbol = st.sidebar.selectbox("Chọn mã", SYMBOLS, index=0)

st.sidebar.markdown("### Thời gian")
preset = st.sidebar.radio(
    "Khoảng thời gian",
    ["3M", "6M", "YTD", "1Y", "All", "Custom"],
    index=1
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
    disabled=(preset != "Custom")
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
    default=SYMBOLS
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
st.caption(
    f"Mã đang xem: {selected_symbol} | Daily range: {filter_start} → {filter_end}"
)

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

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Close", fmt_num(latest_close))
    c2.metric("1D Return", fmt_pct(ret_1d))
    c3.metric("5D Return", fmt_pct(ret_5d))
    c4.metric("20D Return", fmt_pct(ret_20d))
    c5.metric("Avg Vol (20D)", fmt_num(avg_volume, 0))

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Intraday", "Compare", "Data"])

    with tab1:
        left, right = st.columns([2.2, 1])

        with left:
            st.plotly_chart(
                build_price_chart(daily_filtered, chart_type, show_ma5, show_ma20, show_ma50),
                use_container_width=True
            )
            st.plotly_chart(build_return_chart(daily_filtered), use_container_width=True)

        with right:
            st.markdown("### Snapshot")
            snapshot_df = technical_snapshot(daily_filtered)
            st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

            st.markdown("### Period Summary")
            summary_df = pd.DataFrame([
                {"Metric": "Period Return", "Value": fmt_pct(period_ret)},
                {"Metric": "Period High", "Value": fmt_num(daily_filtered["high"].max()) if "high" in daily_filtered.columns else "-"},
                {"Metric": "Period Low", "Value": fmt_num(daily_filtered["low"].min()) if "low" in daily_filtered.columns else "-"},
                {"Metric": "Trading Days", "Value": len(daily_filtered)},
            ])
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

            last_intraday_price = intraday_df[intraday_price_col].iloc[-1] if intraday_price_col else None
            last_intraday_pct = intraday_df[intraday_pct_col].iloc[-1] / 100 if intraday_pct_col and intraday_pct_col in intraday_df.columns else None
            last_intraday_time = intraday_df["snapshot_time"].iloc[-1] if "snapshot_time" in intraday_df.columns else None

            col_a.metric("Latest Intraday Price", fmt_num(last_intraday_price))
            col_b.metric("Intraday Change %", fmt_pct(last_intraday_pct))
            col_c.metric("Last Snapshot", str(last_intraday_time))

            st.plotly_chart(build_intraday_chart(intraday_df), use_container_width=True)
            st.dataframe(intraday_df.tail(30), use_container_width=True)

    with tab3:
        st.markdown("### Compare 3 mã")
        st.plotly_chart(
            build_compare_chart(compare_symbols, preset, custom_range),
            use_container_width=True
        )

        compare_rows = []
        for symbol in compare_symbols:
            temp = load_daily(symbol)
            temp_filtered, _, _ = apply_date_filter(temp, preset, custom_range)
            if temp_filtered.empty:
                continue

            compare_rows.append({
                "Symbol": symbol,
                "Latest Close": temp_filtered["close"].iloc[-1] if "close" in temp_filtered.columns else None,
                "1D Return": calc_return(temp_filtered, 1),
                "5D Return": calc_return(temp_filtered, 5),
                "20D Return": calc_return(temp_filtered, 20),
                "Period Return": (temp_filtered["close"].iloc[-1] / temp_filtered["close"].iloc[0] - 1) if len(temp_filtered) >= 2 else None,
            })

        compare_df = pd.DataFrame(compare_rows)
        if not compare_df.empty:
            for c in ["1D Return", "5D Return", "20D Return", "Period Return"]:
                if c in compare_df.columns:
                    compare_df[c] = compare_df[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
            if "Latest Close" in compare_df.columns:
                compare_df["Latest Close"] = compare_df["Latest Close"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")

            st.dataframe(compare_df, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### Daily Data")
        st.dataframe(daily_filtered, use_container_width=True)

        csv_daily = daily_filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download daily CSV",
            csv_daily,
            file_name=f"{selected_symbol.lower()}_daily_filtered.csv",
            mime="text/csv"
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
                mime="text/csv"
            )

render_dashboard()
