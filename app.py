from __future__ import annotations

from pathlib import Path
import math
import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="VN Stock Dashboard Pro",
    page_icon="📈",
    layout="wide",
)

# =========================
# CONFIG
# =========================
DAILY_DIR = Path("data/daily")
INTRADAY_DIR = Path("data/intraday")
FALLBACK_SYMBOLS = ["ACB", "FPT", "HPG"]
DEFAULT_COMPARE_COUNT = 5
PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "edits": {"shapePosition": True},
}

# =========================
# UI STYLE
# =========================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }
    .small-note {
        color: #8b95a7;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def discover_symbols() -> list[str]:
    if DAILY_DIR.exists():
        files = sorted(DAILY_DIR.glob("*_daily.csv"))
        symbols = sorted({p.name.replace("_daily.csv", "").upper() for p in files})
        if symbols:
            return symbols
    return FALLBACK_SYMBOLS


SYMBOLS = discover_symbols()


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


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.astype(float)


def calc_true_range(df: pd.DataFrame) -> pd.Series:
    if not all(c in df.columns for c in ["high", "low", "close"]):
        return pd.Series(index=df.index, dtype="float64")
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)


def enrich_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%b")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "return_1d",
        "ma_5",
        "ma_20",
        "ma_50",
        "ma_200",
    ]
    df = normalize_numeric(df, numeric_cols)

    if "close" in df.columns:
        df["return_1d"] = safe_pct_change(df["close"], 1)
        df["ret_5d"] = safe_pct_change(df["close"], 5)
        df["ret_20d"] = safe_pct_change(df["close"], 20)
        df["ret_60d"] = safe_pct_change(df["close"], 60)
        df["ret_ytd"] = df["close"] / df.groupby(df["date"].dt.year)["close"].transform("first") - 1

        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()
        df["ma_200"] = df["close"].rolling(200).mean()
        df["ema_12"] = ema(df["close"], 12)
        df["ema_26"] = ema(df["close"], 26)

        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = ema(df["macd"], 9)
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["rsi_14"] = calc_rsi(df["close"], 14)

        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std

        df["cum_max_close"] = df["close"].cummax()
        df["drawdown"] = df["close"] / df["cum_max_close"] - 1
        df["rolling_volatility_20d"] = df["return_1d"].rolling(20).std() * math.sqrt(252) * 100
        df["52w_high"] = df["close"].rolling(252, min_periods=1).max()
        df["52w_low"] = df["close"].rolling(252, min_periods=1).min()
        df["distance_to_52w_high"] = df["close"] / df["52w_high"] - 1
        df["distance_to_52w_low"] = df["close"] / df["52w_low"] - 1

        monthly_return_map = (
            df.groupby("month")["close"]
            .agg(first_close="first", last_close="last")
            .assign(monthly_return=lambda x: x["last_close"] / x["first_close"] - 1)["monthly_return"]
        )
        df["monthly_return"] = df["month"].map(monthly_return_map)

    if "volume" in df.columns:
        df["avg_volume_20d"] = df["volume"].rolling(20).mean()
        df["volume_ratio_20d"] = df["volume"] / df["avg_volume_20d"]

    tr = calc_true_range(df)
    if not tr.empty:
        df["atr_14"] = tr.rolling(14).mean()

    return df


@st.cache_data(ttl=300)
def load_daily(symbol: str) -> pd.DataFrame:
    file_path = DAILY_DIR / f"{symbol.lower()}_daily.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns:
        date_col = next((c for c in ["time", "datetime"] if c in df.columns), None)
        if date_col is None:
            raise ValueError(f"Không tìm thấy cột ngày cho {symbol}. Columns: {df.columns.tolist()}")
        df = df.rename(columns={date_col: "date"})

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


@st.cache_data(ttl=300)
def build_watchlist(symbols: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol in symbols:
        df = load_daily(symbol)
        if df.empty or "close" not in df.columns:
            continue
        last = df.iloc[-1]
        rows.append(
            {
                "Symbol": symbol,
                "Date": last["date"].date() if "date" in last else None,
                "Close": last.get("close"),
                "1D": last.get("return_1d"),
                "5D": last.get("ret_5d"),
                "20D": last.get("ret_20d"),
                "YTD": last.get("ret_ytd"),
                "RSI14": last.get("rsi_14"),
                "Vol Ratio": last.get("volume_ratio_20d"),
                "Volatility 20D": last.get("rolling_volatility_20d"),
                "Dist 52W High": last.get("distance_to_52w_high"),
                "Above MA20": "Yes"
                if pd.notna(last.get("close")) and pd.notna(last.get("ma_20")) and last.get("close") > last.get("ma_20")
                else "No",
                "Above MA50": "Yes"
                if pd.notna(last.get("close")) and pd.notna(last.get("ma_50")) and last.get("close") > last.get("ma_50")
                else "No",
            }
        )
    return pd.DataFrame(rows)


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


# =========================
# FORMATTERS & TABLES
# =========================
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
    if df.empty or len(df) <= periods or "close" not in df.columns:
        return None
    last = df["close"].iloc[-1]
    prev = df["close"].iloc[-(periods + 1)]
    if pd.isna(last) or pd.isna(prev) or prev == 0:
        return None
    return last / prev - 1


def safe_last(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns:
        return None
    val = df[col].iloc[-1]
    return None if pd.isna(val) else val


def monthly_pivot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "monthly_return" not in df.columns:
        return pd.DataFrame()

    monthly = (
        df[["year", "month_num", "month_name", "monthly_return"]]
        .drop_duplicates(subset=["year", "month_num"])
        .sort_values(["year", "month_num"])
    )
    if monthly.empty:
        return pd.DataFrame()

    pivot = monthly.pivot(index="year", columns="month_name", values="monthly_return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    existing = [m for m in month_order if m in pivot.columns]
    return pivot.reindex(columns=existing)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    period_ret = calc_return(df, len(df) - 1) if len(df) >= 2 else None
    high = df["high"].max() if "high" in df.columns else None
    low = df["low"].min() if "low" in df.columns else None
    avg_volume = df["volume"].mean() if "volume" in df.columns else None
    return pd.DataFrame(
        [
            {"Metric": "Latest Close", "Value": fmt_num(safe_last(df, 'close'))},
            {"Metric": "1D Return", "Value": fmt_pct(calc_return(df, 1))},
            {"Metric": "20D Return", "Value": fmt_pct(calc_return(df, 20))},
            {"Metric": "YTD Return", "Value": fmt_pct(safe_last(df, 'ret_ytd'))},
            {"Metric": "Period Return", "Value": fmt_pct(period_ret)},
            {"Metric": "Period High", "Value": fmt_num(high)},
            {"Metric": "Period Low", "Value": fmt_num(low)},
            {"Metric": "Trading Days", "Value": len(df)},
            {"Metric": "Average Volume", "Value": fmt_num(avg_volume, 0)},
            {"Metric": "RSI 14", "Value": fmt_num(safe_last(df, 'rsi_14'), 2)},
            {"Metric": "20D Volatility", "Value": fmt_pct_value(safe_last(df, 'rolling_volatility_20d'))},
            {"Metric": "Max Drawdown", "Value": fmt_pct(df['drawdown'].min()) if 'drawdown' in df.columns else '-'},
        ]
    )


def technical_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    last = df.iloc[-1]
    rows = [
        {
            "Metric": "Close vs MA20",
            "Value": fmt_pct(last["close"] / last["ma_20"] - 1)
            if pd.notna(last.get("close")) and pd.notna(last.get("ma_20")) and last.get("ma_20") != 0
            else "-",
        },
        {
            "Metric": "Close vs MA50",
            "Value": fmt_pct(last["close"] / last["ma_50"] - 1)
            if pd.notna(last.get("close")) and pd.notna(last.get("ma_50")) and last.get("ma_50") != 0
            else "-",
        },
        {"Metric": "Close vs MA200", "Value": fmt_pct(last["close"] / last["ma_200"] - 1) if pd.notna(last.get("close")) and pd.notna(last.get("ma_200")) and last.get("ma_200") != 0 else "-"},
        {"Metric": "RSI 14", "Value": fmt_num(last.get("rsi_14"), 2)},
        {"Metric": "MACD", "Value": fmt_num(last.get("macd"), 3)},
        {"Metric": "MACD Signal", "Value": fmt_num(last.get("macd_signal"), 3)},
        {"Metric": "ATR 14", "Value": fmt_num(last.get("atr_14"), 2)},
        {"Metric": "Vol Ratio 20D", "Value": fmt_num(last.get("volume_ratio_20d"), 2)},
        {"Metric": "20D Volatility", "Value": fmt_pct_value(last.get("rolling_volatility_20d"))},
        {"Metric": "52W High", "Value": fmt_num(last.get("52w_high"))},
        {"Metric": "52W Low", "Value": fmt_num(last.get("52w_low"))},
        {"Metric": "Distance to 52W High", "Value": fmt_pct(last.get("distance_to_52w_high"))},
    ]
    return pd.DataFrame(rows)


# =========================
# CHARTS
# =========================


def enable_drawing_tools(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        dragmode="zoom",
        newshape=dict(
            line=dict(color="#f59e0b", width=2),
            fillcolor="rgba(245, 158, 11, 0.08)",
            opacity=0.9,
        ),
    )
    return fig
def base_layout(fig: go.Figure, height: int):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=70, b=20),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    )
    fig.update_xaxes(automargin=True, tickformat="%b\n%Y", nticks=8, showgrid=False)
    fig.update_yaxes(automargin=True)
    return fig


def plot_chart(fig: go.Figure, key: str | None = None):
    plot_height = int(fig.layout.height) if fig.layout.height else 500
    chart_id = f"chart_{(key or 'plot').replace(' ', '_').replace('-', '_')}"
    fig_json = fig.to_json()
    config_json = json.dumps(PLOT_CONFIG)

    html = f"""
    <div id="{chart_id}" style="width:100%;height:{plot_height}px;"></div>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <script>
    (function() {{
        const gd = document.getElementById({json.dumps(chart_id)});
        const fig = {fig_json};
        const config = {config_json};
        let selectedShapeIndex = null;
        let history = [];
        let historyIndex = -1;
        let restoreDragMode = (fig.layout && fig.layout.dragmode) || 'zoom';
        let rightDragActive = false;
        let panTarget = null;

        gd.setAttribute('tabindex', '0');
        gd.style.outline = 'none';

        function cloneShapes() {{
            return JSON.parse(JSON.stringify((gd.layout && gd.layout.shapes) || []));
        }}

        function sameShapes(a, b) {{
            return JSON.stringify(a || []) === JSON.stringify(b || []);
        }}

        function pushHistory(force) {{
            const shapes = cloneShapes();
            if (force || historyIndex < 0 || !sameShapes(history[historyIndex], shapes)) {{
                history = history.slice(0, historyIndex + 1);
                history.push(shapes);
                historyIndex = history.length - 1;
            }}
        }}

        function applyShapes(shapes) {{
            const nextShapes = JSON.parse(JSON.stringify(shapes || []));
            Plotly.relayout(gd, {{ shapes: nextShapes }});
            selectedShapeIndex = null;
            highlightSelected();
        }}

        function undoShapes() {{
            if (historyIndex > 0) {{
                historyIndex -= 1;
                applyShapes(history[historyIndex]);
            }}
        }}

        function deleteSelectedShape() {{
            if (selectedShapeIndex === null) return;
            const shapes = cloneShapes();
            if (selectedShapeIndex < 0 || selectedShapeIndex >= shapes.length) return;
            shapes.splice(selectedShapeIndex, 1);
            history = history.slice(0, historyIndex + 1);
            history.push(JSON.parse(JSON.stringify(shapes)));
            historyIndex = history.length - 1;
            applyShapes(shapes);
        }}

        function getShapeGroups() {{
            return Array.from(gd.querySelectorAll('.shapelayer .shape-group'));
        }}

        function highlightSelected() {{
            const groups = getShapeGroups();
            groups.forEach((group, idx) => {{
                group.style.filter = idx === selectedShapeIndex ? 'drop-shadow(0 0 2px #ffffff)' : '';
                group.style.opacity = idx === selectedShapeIndex ? '1' : '';
            }});
        }}

        function selectShapeFromEvent(target) {{
            const group = target.closest('.shape-group');
            const groups = getShapeGroups();
            if (!group) return null;
            const idx = groups.indexOf(group);
            return idx >= 0 ? idx : null;
        }}

        function onKey(e) {{
            const key = (e.key || '').toLowerCase();
            if ((e.key === 'Delete' || e.key === 'Backspace') && selectedShapeIndex !== null) {{
                e.preventDefault();
                deleteSelectedShape();
                return;
            }}
            if ((e.ctrlKey || e.metaKey) && key === 'z') {{
                e.preventDefault();
                undoShapes();
            }}
        }}

        Plotly.newPlot(gd, fig.data, fig.layout, config).then(function() {{
            pushHistory(true);

            gd.on('plotly_relayout', function(eventData) {{
                if (!eventData) return;
                const changedKeys = Object.keys(eventData);
                const hasShapeChange = changedKeys.some((k) => k === 'shapes' || k.startsWith('shapes['));
                if (hasShapeChange) {{
                    setTimeout(function() {{
                        pushHistory(false);
                        highlightSelected();
                    }}, 30);
                }}
            }});

            gd.addEventListener('click', function(e) {{
                gd.focus();
                const idx = selectShapeFromEvent(e.target);
                selectedShapeIndex = idx;
                highlightSelected();
            }}, true);

            gd.addEventListener('mouseenter', function() {{ gd.focus(); }});
            gd.addEventListener('keydown', onKey);
            window.addEventListener('keydown', onKey);

            gd.addEventListener('contextmenu', function(e) {{
                e.preventDefault();
            }});

            gd.addEventListener('mousedown', function(e) {{
                if (e.button !== 2) return;
                e.preventDefault();
                gd.focus();
                restoreDragMode = (gd.layout && gd.layout.dragmode) || 'zoom';
                panTarget = gd.querySelector('.nsewdrag') || e.target;
                rightDragActive = true;
                Plotly.relayout(gd, {{ dragmode: 'pan' }}).then(function() {{
                    const syntheticDown = new MouseEvent('mousedown', {{
                        bubbles: true,
                        cancelable: true,
                        clientX: e.clientX,
                        clientY: e.clientY,
                        button: 0,
                        buttons: 1
                    }});
                    panTarget.dispatchEvent(syntheticDown);
                }});
            }}, true);

            document.addEventListener('mouseup', function(e) {{
                if (!rightDragActive) return;
                rightDragActive = false;
                if (panTarget) {{
                    const syntheticUp = new MouseEvent('mouseup', {{
                        bubbles: true,
                        cancelable: true,
                        clientX: e.clientX,
                        clientY: e.clientY,
                        button: 0,
                        buttons: 0
                    }});
                    panTarget.dispatchEvent(syntheticUp);
                }}
                setTimeout(function() {{
                    Plotly.relayout(gd, {{ dragmode: restoreDragMode }});
                }}, 0);
            }}, true);
        }});
    }})();
    </script>
    """

    components.html(html, height=plot_height + 10)


def build_price_chart(
    df: pd.DataFrame,
    chart_type: str,
    show_ma5: bool,
    show_ma20: bool,
    show_ma50: bool,
    show_ma200: bool,
    show_bollinger: bool,
):
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

    overlays = [
        (show_ma5, "ma_5", "MA 5"),
        (show_ma20, "ma_20", "MA 20"),
        (show_ma50, "ma_50", "MA 50"),
        (show_ma200, "ma_200", "MA 200"),
    ]
    for show, col, label in overlays:
        if show and col in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=label), row=1, col=1)

    if show_bollinger and all(c in df.columns for c in ["bb_upper", "bb_lower"]):
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], mode="lines", name="BB Upper", opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], mode="lines", name="BB Lower", opacity=0.6), row=1, col=1)

    if "volume" in df.columns:
        colors = ["#2ca02c" if ret >= 0 else "#d62728" for ret in df["return_1d"].fillna(0)]
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.75,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return enable_drawing_tools(base_layout(fig, 760))


def build_return_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["return_1d"] * 100, name="Daily Return (%)"))
    fig.add_hline(y=0)
    fig.update_yaxes(title_text="Return (%)")
    return enable_drawing_tools(base_layout(fig, 320))


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
    return enable_drawing_tools(base_layout(fig, 320))


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
    return enable_drawing_tools(base_layout(fig, 320))


def build_monthly_return_chart(df: pd.DataFrame):
    monthly = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["monthly_return"]
        .last()
        .dropna()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["monthly_return"] * 100, name="Monthly Return"))
    fig.add_hline(y=0)
    fig.update_yaxes(title_text="Return (%)")
    fig = base_layout(fig, 320)
    fig.update_xaxes(tickangle=-35, nticks=8)
    return fig


def build_rsi_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], mode="lines", name="RSI 14"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_yaxes(title_text="RSI", range=[0, 100])
    return enable_drawing_tools(base_layout(fig, 300))


def build_macd_chart(df: pd.DataFrame):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], mode="lines", name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], mode="lines", name="Signal"), row=1, col=1)
    colors = ["#2ca02c" if x >= 0 else "#d62728" for x in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="Histogram", marker_color=colors), row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=1, col=1)
    fig.update_yaxes(title_text="Hist", row=2, col=1)
    return enable_drawing_tools(base_layout(fig, 420))


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
        height=560 if vol_col else 420,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    )
    return enable_drawing_tools(fig)


def build_compare_chart(symbols: list[str]):
    fig = go.Figure()
    for symbol in symbols:
        df = load_daily(symbol)
        filtered = df.dropna(subset=["close"]).copy()
        if filtered.empty:
            continue
        base = filtered["close"].iloc[0]
        if pd.isna(base) or base == 0:
            continue
        filtered["indexed"] = filtered["close"] / base * 100
        fig.add_trace(go.Scatter(x=filtered["date"], y=filtered["indexed"], mode="lines", name=symbol))
    fig.update_yaxes(title_text="Indexed Price (Base=100)")
    return enable_drawing_tools(base_layout(fig, 430))


def build_correlation_heatmap(symbols: list[str]):
    merged = None
    for symbol in symbols:
        df = load_daily(symbol)
        temp = df[["date", "return_1d"]].rename(columns={"return_1d": symbol})
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
    return enable_drawing_tools(fig)


def build_return_distribution(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["return_1d"] * 100, nbinsx=40, name="Return Distribution"))
    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Frequency")
    return enable_drawing_tools(base_layout(fig, 320))


# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## Bộ lọc")

selected_symbol = st.sidebar.selectbox("Chọn mã", SYMBOLS, index=0)

selected_daily_for_check = load_daily(selected_symbol)
if selected_daily_for_check.empty:
    st.error(f"Không tìm thấy dữ liệu daily cho mã {selected_symbol}.")
    st.stop()

st.sidebar.markdown("### Tùy chọn chart")
chart_type = st.sidebar.radio("Kiểu chart giá", ["Candlestick", "Line"], index=0)
show_ma5 = st.sidebar.checkbox("Hiện MA 5", value=False)
show_ma20 = st.sidebar.checkbox("Hiện MA 20", value=True)
show_ma50 = st.sidebar.checkbox("Hiện MA 50", value=True)
show_ma200 = st.sidebar.checkbox("Hiện MA 200", value=False)
show_bollinger = st.sidebar.checkbox("Hiện Bollinger Bands", value=False)

st.sidebar.markdown("### So sánh")
default_compare = [selected_symbol] + [s for s in SYMBOLS if s != selected_symbol][: max(DEFAULT_COMPARE_COUNT - 1, 0)]
compare_symbols = st.sidebar.multiselect(
    "Chọn mã để compare",
    SYMBOLS,
    default=default_compare,
)

if st.sidebar.button("🔄 Refresh data ngay"):
    st.cache_data.clear()
    st.rerun()



# =========================
# HEADER
# =========================
st.title("📈 VN Stock Dashboard Pro")


# =========================
# LIVE PANEL
# =========================
@st.fragment(run_every="5m")
def render_dashboard():
    daily_df = load_daily(selected_symbol)
    intraday_df, intraday_folder = load_intraday(selected_symbol)

    if daily_df.empty:
        st.warning("Không có dữ liệu daily cho mã đã chọn.")
        return

    first_date = daily_df["date"].min().strftime("%Y-%m-%d")
    last_date = daily_df["date"].max().strftime("%Y-%m-%d")
    latest_close = safe_last(daily_df, "close")
    st.markdown(
        f"**Mã đang xem:** {selected_symbol} &nbsp;&nbsp;|&nbsp;&nbsp; **Data range:** {first_date} → {last_date} &nbsp;&nbsp;|&nbsp;&nbsp; **Latest close:** {fmt_num(latest_close)}"
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Technical", "Intraday", "Compare", "Watchlist", "Data"])

    with tab1:
        st.markdown("### Price & Volume")
        plot_chart(
            build_price_chart(daily_df, chart_type, show_ma5, show_ma20, show_ma50, show_ma200, show_bollinger),
            key="price_chart",
        )

        row1_left, row1_right = st.columns(2)
        with row1_left:
            st.markdown("### Daily Return")
            plot_chart(build_return_chart(daily_df), key="return_chart")
        with row1_right:
            st.markdown("### Monthly Return")
            plot_chart(build_monthly_return_chart(daily_df), key="monthly_return_chart")

        row2_left, row2_right = st.columns(2)
        with row2_left:
            st.markdown("### Drawdown")
            plot_chart(build_drawdown_chart(daily_df), key="drawdown_chart")
        with row2_right:
            st.markdown("### Volatility (20D)")
            plot_chart(build_volatility_chart(daily_df), key="volatility_chart")

        row3_left, row3_right = st.columns(2)
        with row3_left:
            st.markdown("### Snapshot")
            st.dataframe(technical_snapshot(daily_df), use_container_width=True, hide_index=True)
        with row3_right:
            st.markdown("### Summary")
            st.dataframe(summary_table(daily_df), use_container_width=True, hide_index=True)

    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### RSI")
            plot_chart(build_rsi_chart(daily_df), key="rsi_chart")
        with col_right:
            st.markdown("### Return Distribution")
            dist_df = daily_df.dropna(subset=["return_1d"]) if "return_1d" in daily_df.columns else daily_df
            plot_chart(build_return_distribution(dist_df), key="return_distribution_chart")

        st.markdown("### MACD")
        plot_chart(build_macd_chart(daily_df), key="macd_chart")

        monthly_tbl = monthly_pivot(daily_df)
        if not monthly_tbl.empty:
            display_monthly = monthly_tbl.applymap(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
            st.markdown("### Monthly Heatmap Table")
            st.dataframe(display_monthly, use_container_width=True)

    with tab3:
        st.markdown("### Intraday View")
        if intraday_df.empty:
            st.info("Chưa có dữ liệu intraday.")
        else:
            st.caption(f"File intraday mới nhất: {intraday_folder}")
            plot_chart(build_intraday_chart(intraday_df), key="intraday_chart")
            st.dataframe(intraday_df.tail(50), use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### Relative Performance")
        if compare_symbols:
            plot_chart(build_compare_chart(compare_symbols), key="compare_chart")
        else:
            st.warning("Hãy chọn ít nhất 1 mã để compare.")

        compare_col_left, compare_col_right = st.columns([1.35, 1])

        with compare_col_left:
            compare_rows = []
            for symbol in compare_symbols:
                temp = load_daily(symbol)
                if temp.empty or "close" not in temp.columns:
                    continue
                last = temp.iloc[-1]
                compare_rows.append(
                    {
                        "Symbol": symbol,
                        "Latest Close": last.get("close"),
                        "1D Return": calc_return(temp, 1),
                        "5D Return": calc_return(temp, 5),
                        "20D Return": calc_return(temp, 20),
                        "YTD Return": last.get("ret_ytd"),
                        "Period Return": (temp["close"].iloc[-1] / temp["close"].iloc[0] - 1) if len(temp) >= 2 else None,
                        "RSI 14": last.get("rsi_14"),
                        "20D Vol": last.get("rolling_volatility_20d"),
                        "Vol Ratio": last.get("volume_ratio_20d"),
                        "Dist 52W High": last.get("distance_to_52w_high"),
                    }
                )

            compare_df = pd.DataFrame(compare_rows)
            if not compare_df.empty:
                sortable = compare_df.copy()
                vol_fallback = sortable["20D Vol"].median() if "20D Vol" in sortable.columns else 0
                sortable["Rank Score"] = sortable["20D Return"].fillna(0) * 100 - sortable["20D Vol"].fillna(vol_fallback) / 5
                sortable = sortable.sort_values("Rank Score", ascending=False)

                display_df = sortable.drop(columns=["Rank Score"])
                for c in ["1D Return", "5D Return", "20D Return", "YTD Return", "Period Return", "Dist 52W High"]:
                    if c in display_df.columns:
                        display_df[c] = display_df[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
                if "20D Vol" in display_df.columns:
                    display_df["20D Vol"] = display_df["20D Vol"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
                if "Latest Close" in display_df.columns:
                    display_df["Latest Close"] = display_df["Latest Close"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
                if "RSI 14" in display_df.columns:
                    display_df["RSI 14"] = display_df["RSI 14"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                if "Vol Ratio" in display_df.columns:
                    display_df["Vol Ratio"] = display_df["Vol Ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                st.markdown("### Compare Summary")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        with compare_col_right:
            if len(compare_symbols) >= 2:
                st.markdown("### Return Correlation")
                plot_chart(build_correlation_heatmap(compare_symbols), key="correlation_heatmap")

    with tab5:
        st.markdown("### Watchlist Overview")
        watchlist_df = build_watchlist(tuple(SYMBOLS))
        if watchlist_df.empty:
            st.info("Chưa có dữ liệu để tạo watchlist.")
        else:
            sort_col = st.selectbox(
                "Sắp xếp theo",
                ["20D", "YTD", "1D", "RSI14", "Vol Ratio", "Dist 52W High", "Volatility 20D"],
                index=0,
                key="watchlist_sort_col",
            )
            sort_ascending = st.checkbox("Tăng dần", value=False, key="watchlist_sort_asc")
            sorted_watchlist = watchlist_df.sort_values(sort_col, ascending=sort_ascending, na_position="last").copy()

            for c in ["1D", "5D", "20D", "YTD", "Dist 52W High"]:
                sorted_watchlist[c] = sorted_watchlist[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
            sorted_watchlist["Close"] = sorted_watchlist["Close"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
            sorted_watchlist["RSI14"] = sorted_watchlist["RSI14"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            sorted_watchlist["Vol Ratio"] = sorted_watchlist["Vol Ratio"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            sorted_watchlist["Volatility 20D"] = sorted_watchlist["Volatility 20D"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
            st.dataframe(sorted_watchlist, use_container_width=True, hide_index=True)

    with tab6:
        st.markdown("### Daily Data")
        st.dataframe(daily_df, use_container_width=True)
        csv_daily = daily_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download daily CSV",
            csv_daily,
            file_name=f"{selected_symbol.lower()}_daily_full.csv",
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
