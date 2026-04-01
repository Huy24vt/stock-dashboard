from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

# `vnstock_data` may be provided by a separate package. Try a safe import
try:
    from vnstock_data import Trading
except Exception:
    Trading = None

from config import SYMBOLS

if Trading is None:
    print("[WARN] vnstock_data not available — skipping intraday snapshot step.")
    sys.exit(0)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    today_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

    out_dir = Path("data/intraday") / today_str
    ensure_dir(out_dir)

    trading = Trading(symbol="ACB", source="vci")
    board = trading.price_board(SYMBOLS, flatten_columns=True, drop_levels=[0])
    board.columns = [c.strip().lower() for c in board.columns]

    # Bạn có thể in 1 lần để xác nhận đúng tên cột thực tế
    # print(board.columns.tolist())

    # Ưu tiên khớp các cột thường gặp
    price_col = next((c for c in ["match_price", "last_price", "price", "close"] if c in board.columns), None)
    change_col = next((c for c in ["change", "price_change"] if c in board.columns), None)
    pct_col = next((c for c in ["pct_change", "change_percent", "price_change_percent"] if c in board.columns), None)
    vol_col = next((c for c in ["match_volume", "volume", "total_volume"] if c in board.columns), None)
    ref_col = next((c for c in ["ref_price", "reference_price"] if c in board.columns), None)

    if price_col is None:
        raise ValueError(f"Không tìm thấy cột giá hiện tại. Columns: {board.columns.tolist()}")

    board["snapshot_time"] = ts_str

    for symbol in SYMBOLS:
        row = board[board["symbol"].str.upper() == symbol].copy()
        if row.empty:
            print(f"[WARN] No row for {symbol}")
            continue

        keep_cols = ["snapshot_time", "symbol", price_col]
        for c in [change_col, pct_col, vol_col, ref_col]:
            if c:
                keep_cols.append(c)

        row = row[keep_cols].copy()

        out_file = out_dir / f"{symbol.lower()}_intraday.csv"

        if out_file.exists():
            old = pd.read_csv(out_file)
            row = pd.concat([old, row], ignore_index=True)
            row = row.drop_duplicates(subset=["snapshot_time"]).sort_values("snapshot_time")

        row.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"Updated {out_file} | rows={len(row)}")

if __name__ == "__main__":
    main()
