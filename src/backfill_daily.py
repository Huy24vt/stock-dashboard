from datetime import date, timedelta
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
    today = date.today()

    for symbol in SYMBOLS:
        print(f"Backfill daily: {symbol}")
        out_file = OUT_DIR / f"{symbol.lower()}_daily.csv"

        # Determine start date for fetching: continue from last available date
        if out_file.exists():
            try:
                old = pd.read_csv(out_file)
                old.columns = [c.strip().lower() for c in old.columns]
                if "date" in old.columns:
                    old["date"] = pd.to_datetime(old["date"], errors="coerce")
                    last_date = old["date"].max().date()
                    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    start_date = START_DATE
            except Exception:
                start_date = START_DATE
        else:
            start_date = START_DATE

        end_date = today.strftime("%Y-%m-%d")

        # If start_date is after end_date, nothing to do
        if pd.to_datetime(start_date).date() > today:
            print(f"No new data for {symbol} (up-to-date: {start_date})")
            continue

        quote = Quote(symbol=symbol, source="KBS")
        try:
            df = quote.history(start=start_date, end=end_date, interval="1D")
        except Exception as e:
            print(f"[ERROR] Failed to fetch {symbol}: {e}")
            continue

        if df is None or len(df) == 0:
            print(f"[INFO] No new rows for {symbol} between {start_date} and {end_date}")
            continue

        df = build_features(df)

        if out_file.exists():
            try:
                old = pd.read_csv(out_file)
                new = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
            except Exception:
                new = df
        else:
            new = df

        new.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"Saved {out_file} | rows={len(new)}")

if __name__ == "__main__":
    main()
