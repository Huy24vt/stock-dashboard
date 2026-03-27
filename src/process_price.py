from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_one_file(input_file: Path):
    df = pd.read_csv(input_file)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = None
    for c in ["time", "date", "datetime"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"Không tìm thấy cột ngày trong file {input_file.name}")

    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "close" in df.columns:
        df["return_1d"] = df["close"].pct_change()
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()
        df["ath_close"] = df["close"].cummax()
        df["drawdown"] = df["close"] / df["ath_close"] - 1

    if "volume" in df.columns:
        df["volume_ma_20"] = df["volume"].rolling(20).mean()

    symbol_name = input_file.name.replace("_price.csv", "")
    output_file = PROCESSED_DIR / f"{symbol_name}_price_processed.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Processed: {output_file} | rows={len(df)}")


def main():
    files = sorted(RAW_DIR.glob("*_price.csv"))

    if not files:
        raise FileNotFoundError("Không có file nào trong data/raw")

    for file in files:
        try:
            process_one_file(file)
        except Exception as e:
            print(f"[ERROR] {file.name}: {e}")


if __name__ == "__main__":
    main()
