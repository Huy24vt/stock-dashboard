from datetime import date
from pathlib import Path
import pandas as pd
from vnstock import Quote
from config import SYMBOLS, START_DATE

OUT_DIR = Path("data/daily")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next((c for c in ["time", "date", "datetime"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"Không tìm thấy cột ngày. Columns: {df.columns.tolist()}")

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" in df.columns:
        df["return_1d"] = df["close"].pct_change()
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()

    return df

def main():
    end_date = date.today().strftime("%Y-%m-%d")

    for symbol in SYMBOLS:
        print(f"Backfill daily: {symbol}")
        quote = Quote(symbol=symbol, source="KBS")
        df = quote.history(start=START_DATE, end=end_date, interval="1D")
        df = build_features(df)

        out_file = OUT_DIR / f"{symbol.lower()}_daily.csv"
        df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"Saved {out_file} | rows={len(df)}")

if __name__ == "__main__":
    main()
