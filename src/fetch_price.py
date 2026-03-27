from pathlib import Path
from datetime import date
import pandas as pd
from vnstock import Quote

from config import VN30_SYMBOLS, START_DATE


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_one_symbol(symbol: str, start_date: str, end_date: str):
    print(f"Fetching {symbol} | {start_date} -> {end_date}")

    quote = Quote(symbol=symbol, source="VCI")
    df = quote.history(start=start_date, end=end_date, interval="1D")

    if df is None or len(df) == 0:
        print(f"[WARN] No data for {symbol}")
        return

    df.columns = [c.strip().lower() for c in df.columns]

    out_file = RAW_DIR / f"{symbol.lower()}_price.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_file} | rows={len(df)}")


def main():
    end_date = date.today().strftime("%Y-%m-%d")

    for symbol in VN30_SYMBOLS:
        try:
            fetch_one_symbol(symbol, START_DATE, end_date)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")


if __name__ == "__main__":
    main()
