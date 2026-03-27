from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide"
)

PROCESSED_DIR = Path("data/processed")


def list_processed_files():
    if not PROCESSED_DIR.exists():
        return []
    return sorted(PROCESSED_DIR.glob("*.csv"))


@st.cache_data(ttl=300)
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next((c for c in ["time", "date", "datetime"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"Khong tim thay cot ngay. Columns hien co: {df.columns.tolist()}")

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "return_1d", "ma_5", "ma_20", "ma_50", "volume_ma_20"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_price_chart(df: pd.DataFrame, show_ma5: bool, show_ma20: bool, show_ma50: bool):
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
        title="Price Trend",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )
    return fig


def build_volume_chart(df: pd.DataFrame):
    fig = go.Figure()

    if "volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Volume"
            )
        )

    if "volume_ma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["volume_ma_20"],
                mode="lines",
                name="Volume MA 20"
            )
        )

    fig.update_layout(
        title="Volume",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified"
    )
    return fig


def build_return_chart(df: pd.DataFrame):
    return_df = df[["date", "return_1d"]].dropna().copy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=return_df["date"],
            y=return_df["return_1d"],
            name="Return 1D"
        )
    )
    fig.update_layout(
        title="Daily Return",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Return",
        hovermode="x unified"
    )
    return fig


def format_delta(latest, previous):
    if previous is None or pd.isna(previous):
        return None

    delta = latest - previous
    if previous == 0 or pd.isna(latest):
        return f"{delta:,.2f}"

    delta_pct = (delta / previous) * 100
    return f"{delta:,.2f} ({delta_pct:.2f}%)"


@st.fragment(run_every="5m")
def render_dashboard(
    selected_file: str,
    selected_path: Path,
    start_date,
    end_date,
    show_ma5: bool,
    show_ma20: bool,
    show_ma50: bool
):
    df = load_data(str(selected_path))

    filtered = df[
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date)
    ].copy()

    if filtered.empty:
        st.warning("Khong co du lieu trong khoang ngay da chon.")
        return

    symbol_name = selected_file.replace("_price_processed.csv", "").upper()

    latest_row = filtered.iloc[-1]
    prev_row = filtered.iloc[-2] if len(filtered) >= 2 else None

    latest_close = latest_row["close"] if "close" in filtered.columns else None
    prev_close = prev_row["close"] if prev_row is not None and "close" in filtered.columns else None

    latest_volume = latest_row["volume"] if "volume" in filtered.columns else None
    prev_volume = prev_row["volume"] if prev_row is not None and "volume" in filtered.columns else None

    period_high = filtered["high"].max() if "high" in filtered.columns else None
    period_low = filtered["low"].min() if "low" in filtered.columns else None

    col1, col2, col3, col4 = st.columns(4)

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

    st.markdown(f"## {symbol_name}")
    st.caption(
        f"From {start_date} to {end_date} | Rows: {len(filtered):,} | Auto refresh every 5 minutes"
    )

    tab1, tab2, tab3 = st.tabs(["Overview", "Data", "Stats"])

    with tab1:
        st.plotly_chart(
            build_price_chart(filtered, show_ma5, show_ma20, show_ma50),
            use_container_width=True
        )

        if "volume" in filtered.columns:
            st.plotly_chart(
                build_volume_chart(filtered),
                use_container_width=True
            )

        if "return_1d" in filtered.columns:
            st.plotly_chart(
                build_return_chart(filtered),
                use_container_width=True
            )

    with tab2:
        st.subheader("Raw Filtered Data")
        st.dataframe(filtered, use_container_width=True)

        csv_data = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="Download filtered CSV",
            data=csv_data,
            file_name=f"{symbol_name.lower()}_filtered.csv",
            mime="text/csv"
        )

    with tab3:
        st.subheader("Quick Stats")

        stats_cols = [c for c in ["open", "high", "low", "close", "volume", "return_1d"] if c in filtered.columns]
        if stats_cols:
            st.dataframe(filtered[stats_cols].describe().T, use_container_width=True)

        if "close" in filtered.columns:
            st.write("Last 10 closes")
            st.dataframe(
                filtered[["date", "close"]].tail(10),
                use_container_width=True
            )


def main():
    st.title("📈 Local Stock Dashboard")
    st.caption("Dashboard doc du lieu CSV da xu ly tu vnstock")

    files = list_processed_files()
    if not files:
        st.error("Khong tim thay file CSV trong folder data/processed")
        st.stop()

    file_map = {f.name: f for f in files}
    selected_file = st.sidebar.selectbox("Chon file du lieu", list(file_map.keys()))
    selected_path = file_map[selected_file]

    # Load 1 lần để lấy min/max date cho sidebar
    df_preview = load_data(str(selected_path))

    st.sidebar.markdown("### Bo loc")
    min_date = df_preview["date"].min().date()
    max_date = df_preview["date"].max().date()

    default_start = max(min_date, (df_preview["date"].max() - pd.Timedelta(days=180)).date())
    date_range = st.sidebar.date_input(
        "Khoang thoi gian",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    show_ma5 = st.sidebar.checkbox("Hien MA 5", value=False)
    show_ma20 = st.sidebar.checkbox("Hien MA 20", value=True)
    show_ma50 = st.sidebar.checkbox("Hien MA 50", value=True)

    render_dashboard(
        selected_file=selected_file,
        selected_path=selected_path,
        start_date=start_date,
        end_date=end_date,
        show_ma5=show_ma5,
        show_ma20=show_ma20,
        show_ma50=show_ma50
    )


if __name__ == "__main__":
    main()