from pathlib import Path
import pandas as pd

from vnstock import Trading
from config import SYMBOLS


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    today_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")

    out_dir = Path("data/intraday") / today_str
    ensure_dir(out_dir)

    # Theo docs hiện tại của vnstock:
    # from vnstock import Trading
    # Trading(source='VCI').price_board([...])
    trading = Trading(source="VCI")
    board = trading.price_board(SYMBOLS, flatten_columns=True, drop_levels=[0])

    if board is None or len(board) == 0:
        raise ValueError("price_board trả về rỗng.")

    board.columns = [c.strip().lower() for c in board.columns]

    if "symbol" not in board.columns:
        raise ValueError(f"Không tìm thấy cột 'symbol'. Columns: {board.columns.tolist()}")

    # Ưu tiên khớp các cột thường gặp
    price_col = next((c for c in ["match_price", "last_price", "price", "close"] if c in board.columns), None)
    change_col = next((c for c in ["change", "price_change"] if c in board.columns), None)
    pct_col = next((c for c in ["pct_change", "change_percent", "price_change_percent"] if c in board.columns), None)
    vol_col = next((c for c in ["match_volume", "volume", "total_volume"] if c in board.columns), None)
    ref_col = next((c for c in ["ref_price", "reference_price"] if c in board.columns), None)

    if price_col is None:
        raise ValueError(f"Không tìm thấy cột giá hiện tại. Columns: {board.columns.tolist()}")

    board["snapshot_time"] = ts_str
    board["symbol"] = board["symbol"].astype(str).str.upper()

    for symbol in SYMBOLS:
        symbol = symbol.upper()
        row = board[board["symbol"] == symbol].copy()

        if row.empty:
            print(f"[WARN] No row for {symbol}")
            continue

        keep_cols = ["snapshot_time", "symbol", price_col]
        for c in [change_col, pct_col, vol_col, ref_col]:
            if c and c not in keep_cols:
                keep_cols.append(c)

        row = row[keep_cols].copy()

        out_file = out_dir / f"{symbol.lower()}_intraday.csv"

        if out_file.exists():
            old = pd.read_csv(out_file)
            old.columns = [c.strip().lower() for c in old.columns]
            row.columns = [c.strip().lower() for c in row.columns]

            merged = pd.concat([old, row], ignore_index=True)
            merged = merged.drop_duplicates(subset=["snapshot_time"]).sort_values("snapshot_time")
            merged.to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"Updated {out_file} | rows={len(merged)}")
        else:
            row.to_csv(out_file, index=False, encoding="utf-8-sig")
            print(f"Created {out_file} | rows={len(row)}")


if __name__ == "__main__":
    main()
