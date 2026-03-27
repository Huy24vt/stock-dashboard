import pandas as pd
from pathlib import Path


def main():
    input_file = Path("data/acb_price.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    # Chuẩn hóa tên cột
    df.columns = [c.strip().lower() for c in df.columns]

    # Tìm cột ngày
    date_col = None
    for c in ["time", "date", "datetime"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"Không tìm thấy cột ngày. Columns hiện có: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Convert numeric an toàn
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Feature engineering cơ bản
    if "close" in df.columns:
        df["return_1d"] = df["close"].pct_change()
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()

    if "volume" in df.columns:
        df["volume_ma_20"] = df["volume"].rolling(20).mean()

    output_file = output_dir / "acb_price_processed.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Saved processed file to: {output_file}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()