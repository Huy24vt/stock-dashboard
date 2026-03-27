import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    input_file = Path("data/processed/acb_price_processed.csv")
    output_dir = Path("data/charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = None
    for c in ["time", "date", "datetime"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        raise ValueError(f"Không tìm thấy cột ngày. Columns hiện có: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col])

    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df["close"], label="Close")
    if "ma_20" in df.columns:
        plt.plot(df[date_col], df["ma_20"], label="MA20")
    if "ma_50" in df.columns:
        plt.plot(df[date_col], df["ma_50"], label="MA50")

    plt.title("ACB Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    output_file = output_dir / "acb_price_trend.png"
    plt.savefig(output_file, dpi=150)
    plt.show()

    print(f"Saved chart to: {output_file}")


if __name__ == "__main__":
    main()