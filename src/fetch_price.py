from vnstock import Quote
import pandas as pd
from pathlib import Path

def main():
    symbol = "ACB"
    quote = Quote(symbol=symbol, source="VCI")
    df = quote.history(start="2024-01-01", end="2024-12-31", interval="1D")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / f"{symbol.lower()}_price.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"Saved to: {out_file}")
    print(df.head())

if __name__ == "__main__":
    main()