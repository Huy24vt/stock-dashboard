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

        # If file exists, append only new rows (avoid rewriting historical data)
        if out_file.exists():
            try:
                old = pd.read_csv(out_file)
                old.columns = [c.strip().lower() for c in old.columns]
                if "date" in old.columns:
                    old["date"] = pd.to_datetime(old["date"], errors="coerce")
                    last_date = old["date"].max().date()
                    # keep only strictly newer rows
                    df_new = df[df["date"].dt.date > last_date].copy()
                else:
                    df_new = df
            except Exception:
                df_new = df

            if df_new is None or len(df_new) == 0:
                print(f"[INFO] No new rows to append for {symbol}")
                continue

            # Ensure same columns/order as existing file
            for c in old.columns:
                if c not in df_new.columns:
                    df_new[c] = pd.NA

            # Add any new columns from df_new to old column list (keep df_new order)
            df_new = df_new.reindex(columns=old.columns.tolist())

            # Append to CSV without header
            df_new.to_csv(out_file, mode="a", header=False, index=False, encoding="utf-8-sig")
            print(f"Appended {len(df_new)} rows to {out_file}")
        else:
            # Write full file
            df.to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"Saved {out_file} | rows={len(df)}")

if __name__ == "__main__":
    main()
