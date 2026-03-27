from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =========================
# CONFIG
# =========================
SYMBOLS = ["ACB", "FPT", "HPG"]   # Muon scale len 30 ma thi thay list nay
DAILY_DIR = Path("data/daily")
INTRADAY_DIR = Path("data/intraday")
TIMEZONE = "Asia/Ho_Chi_Minh"

st.set_page_config(
    page_title="VN30 Mini Dashboard",
    page_icon="📈",
    layout="wide"
)


# =========================
# HELPERS
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def resolve_symbol_file(base_dir: Path, symbol: str, suffix: str) -> Path | None:
    candidates = [
        base_dir / f"{symbol.lower()}_{suffix}.csv",
        base_dir / f"{symbol.upper()}_{suffix}.csv",
        base_dir / f"{symbol.capitalize()}_{suffix}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def format_delta(latest, previous):
    if previous is None or pd.isna(previous) or pd.isna(latest):
        return None

    delta = latest - previous
    if previous == 0:
        return f"{delta:,.2f}"

    delta_pct = (delta / previous) * 100
    return f"{delta:,.2f} ({delta_pct:.2f}%)"


def get_intraday_price_col(df: pd.DataFrame):
    return next(
        (c for c in ["match_price", "last_price", "price", "close", "reference_price"] if c in df.columns),
        None
    )


def get_volume_col(df: pd.DataFrame):
    return next(
        (c for c in ["volume", "matched_volume", "total_volume", "match_volume"] if c in df.columns),
        None
    )


# =========================
# LOADERS
# =========================
@st.cache_data(ttl=300)
def load_daily(symbol: str) -> pd.DataFrame:
    file_path = resolve_symbol_file(DAILY_DIR, symbol, "daily")
    if file_path is None:
        raise FileNotFoundError(
            f"Khong tim thay file daily cho {symbol}. "
            f"Ky vong: data/daily/{symbol.lower()}_daily.csv"
        )

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    date_col = next((c for c in ["date", "time", "datetime", "trading_date"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"Khong tim thay cot ngay trong file daily cua {symbol}. Columns hien co: {df.columns.tolist()}"
        )

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "ma_5", "ma_20", "ma_50", "return_1d"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Neu chua co MA thi tu tinh
    if "close" in df.columns:
        if "ma_5" not in df.columns:
            df["ma_5"] = df["close"].rolling(5).mean()
        if "ma_20" not in df.columns:
            df["ma_20"] = df["close"].rolling(20).mean()
        if "ma_50" not in df.columns:
            df["ma_50"] = df["close"].rolling(50).mean()
        if "return_1d" not in df.columns:
            df["return_1d"] = df["close"].pct_change() * 100

    return df


@st.cache_data(ttl=300)
def load_intraday(symbol: str) -> pd.DataFrame:
    today_str = pd.Timestamp.now(tz=TIMEZONE).strftime("%Y-%m-%d")
    intraday_day_dir = INTRADAY_DIR / today_str

    if not intraday_day_dir.exists():
        return pd.DataFrame()

    file_path = resolve_symbol_file(intraday_day_dir, symbol, "intraday")
    if file_path is None:
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df = normalize_columns(df)

    time_col = next(
        (c for c in ["snapshot_time", "time", "datetime", "date", "timestamp"] if c in df.columns),
        None
    )
    if time_col is None:
        return pd.DataFrame()

    df = df.rename(columns={time_col: "snapshot_time"})
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")

    numeric_cols = [
        "match_price", "last_price", "price", "close", "reference_price",
        "volume", "matched_volume", "total_volume", "match_volume"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["snapshot_time"]).sort_values("snapshot_time").reset_index(drop=True)
    return df


# =========================
# CHART BUILDERS
# =========================
def build_daily_price_chart(df: pd.DataFrame, symbol: str, show_ma5: bool, show_ma20: bool, show_ma50: bool):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            name="Close"
        )
    )

    if show_ma5 and "ma_5" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["ma_5"],
                mode="lines",
                name="MA 5"
            )
        )

    if show_ma20 and "ma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["ma_20"],
                mode="lines",
                name="MA 20"
            )
        )

    if show_ma50 and "ma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["ma_50"],
                mode="lines",
                name="MA 50"
            )
        )

    fig.update_layout(
        title=f"{symbol} - Daily Price",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    return fig


def build_daily_volume_chart(df: pd.DataFrame, symbol: str):
    if "volume" not in df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="Volume"
        )
    )

    fig.update_layout(
        title=f"{symbol} - Daily Volume",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified"
    )
    return fig


def build_return_chart(df: pd.DataFrame, symbol: str):
    if "return_1d" not in df.columns:
        return None

    return_df = df[["date", "return_1d"]].dropna().copy()
    if return_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=return_df["date"],
            y=return_df["return_1d"],
            name="Return 1D"
        )
    )
    fig.update_layout(
        title=f"{symbol} - Daily Return (%)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified"
    )
    return fig


def build_intraday_chart(df: pd.DataFrame, symbol: str):
    price_col = get_intraday_price_col(df)
    if price_col is None or df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["snapshot_time"],
            y=df[price_col],
            mode="lines+markers",
            name="Price"
        )
    )

    fig.update_layout(
        title=f"{symbol} - Intraday cập nhật mỗi 5 phút",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Snapshot Time",
        yaxis_title="Price",
        hovermode="x unified"
    )
    return fig


# =========================
# RENDER
# =========================
@st.fragment(run_every="5m")
def render_dashboard(symbol: str, start_date, end_date, show_ma5: bool, show_ma20: bool, show_ma50: bool):
    daily = load_daily(symbol)
    intraday = load_intraday(symbol)

    filtered_daily = daily[
        (daily["date"].dt.date >= start_date) &
        (daily["date"].dt.date <= end_date)
    ].copy()

    if filtered_daily.empty:
        st.warning("Khong co du lieu daily trong khoang ngay da chon.")
        return

    latest_row = filtered_daily.iloc[-1]
    prev_row = filtered_daily.iloc[-2] if len(filtered_daily) >= 2 else None

    latest_close = latest_row["close"] if "close" in filtered_daily.columns else None
    prev_close = prev_row["close"] if prev_row is not None and "close" in filtered_daily.columns else None

    latest_volume = latest_row["volume"] if "volume" in filtered_daily.columns else None
    prev_volume = prev_row["volume"] if prev_row is not None and "volume" in filtered_daily.columns else None

    period_high = filtered_daily["high"].max() if "high" in filtered_daily.columns else None
    period_low = filtered_daily["low"].min() if "low" in filtered_daily.columns else None

    intraday_price_col = get_intraday_price_col(intraday)
    intraday_latest = None
    intraday_prev = None
    if not intraday.empty and intraday_price_col is not None:
        intraday_latest = intraday.iloc[-1][intraday_price_col]
        if len(intraday) >= 2:
            intraday_prev = intraday.iloc[-2][intraday_price_col]

    st.markdown(f"## {symbol}")
    st.caption(
        f"Daily: từ {start_date} đến {end_date} | "
        f"Rows daily: {len(filtered_daily):,} | "
        f"Intraday auto refresh: mỗi 5 phút"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if latest_close is not None and not pd.isna(latest_close):
            st.metric(
                "Latest Close",
                f"{latest_close:,.2f}",
                format_delta(latest_close, prev_close)
            )

    with col2:
        if latest_volume is not None and not pd.isna(latest_volume):
            st.metric(
                "Latest Volume",
                f"{latest_volume:,.0f}",
                format_delta(latest_volume, prev_volume)
            )

    with col3:
        if period_high is not None and not pd.isna(period_high):
            st.metric("Period High", f"{period_high:,.2f}")

    with col4:
        if period_low is not None and not pd.isna(period_low):
            st.metric("Period Low", f"{period_low:,.2f}")

    with col5:
        if intraday_latest is not None and not pd.isna(intraday_latest):
            st.metric(
                "Intraday Last",
                f"{intraday_latest:,.2f}",
                format_delta(intraday_latest, intraday_prev)
            )

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Intraday", "Data", "Stats"])

    with tab1:
        st.plotly_chart(
            build_daily_price_chart(filtered_daily, symbol, show_ma5, show_ma20, show_ma50),
            use_container_width=True
        )

        volume_fig = build_daily_volume_chart(filtered_daily, symbol)
        if volume_fig is not None:
            st.plotly_chart(volume_fig, use_container_width=True)

        return_fig = build_return_chart(filtered_daily, symbol)
        if return_fig is not None:
            st.plotly_chart(return_fig, use_container_width=True)

    with tab2:
        st.subheader(f"{symbol} - Intraday")
        if intraday.empty:
            st.info("Hôm nay chưa có dữ liệu intraday.")
        else:
            intraday_fig = build_intraday_chart(intraday, symbol)
            if intraday_fig is not None:
                st.plotly_chart(intraday_fig, use_container_width=True)
            else:
                st.warning("Co du lieu intraday nhung khong tim thay cot gia phu hop.")

            st.dataframe(intraday.tail(20), use_container_width=True)

    with tab3:
        st.subheader("Daily Data")
        st.dataframe(filtered_daily, use_container_width=True)

        daily_csv = filtered_daily.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download daily CSV",
            data=daily_csv,
            file_name=f"{symbol.lower()}_daily_filtered.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("Intraday Data")
        if intraday.empty:
            st.info("Khong co du lieu intraday de download.")
        else:
            st.dataframe(intraday, use_container_width=True)

            intraday_csv = intraday.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download intraday CSV",
                data=intraday_csv,
                file_name=f"{symbol.lower()}_intraday_today.csv",
                mime="text/csv"
            )

    with tab4:
        st.subheader("Quick Stats")

        stats_cols = [c for c in ["open", "high", "low", "close", "volume", "return_1d"] if c in filtered_daily.columns]
        if stats_cols:
            st.dataframe(filtered_daily[stats_cols].describe().T, use_container_width=True)

        if "close" in filtered_daily.columns:
            st.write("Last 10 daily closes")
            st.dataframe(
                filtered_daily[["date", "close"]].tail(10),
                use_container_width=True
            )

        if not intraday.empty and intraday_price_col is not None:
            st.write("Last 10 intraday points")
            st.dataframe(
                intraday[["snapshot_time", intraday_price_col]].tail(10),
                use_container_width=True
            )


# =========================
# MAIN
# =========================
def main():
    st.title("📈 VN30 Mini Dashboard")
    st.caption("Daily historical từ 2024 đến nay + Intraday update mỗi 5 phút")

    symbol = st.sidebar.selectbox("Chọn mã", SYMBOLS)

    try:
        daily_preview = load_daily(symbol)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.sidebar.markdown("### Bộ lọc")
    min_date = daily_preview["date"].min().date()
    max_date = daily_preview["date"].max().date()

    default_start = max(min_date, (daily_preview["date"].max() - pd.Timedelta(days=180)).date())

    date_range = st.sidebar.date_input(
        "Khoảng thời gian",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    st.sidebar.markdown("### Moving Average")
    show_ma5 = st.sidebar.checkbox("Hiện MA 5", value=False)
    show_ma20 = st.sidebar.checkbox("Hiện MA 20", value=True)
    show_ma50 = st.sidebar.checkbox("Hiện MA 50", value=True)

    st.sidebar.markdown("### Gợi ý cấu trúc file")
    st.sidebar.code(
        f"""data/
  daily/
    {symbol.lower()}_daily.csv
  intraday/
    YYYY-MM-DD/
      {symbol.lower()}_intraday.csv""",
        language="text"
    )

    render_dashboard(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        show_ma5=show_ma5,
        show_ma20=show_ma20,
        show_ma50=show_ma50
    )


if __name__ == "__main__":
    main()
