import subprocess
import sys

def run_step(script_path: str):
    subprocess.run([sys.executable, script_path], check=True)

def main():
    # Run intraday snapshot update (safe to skip if dependency missing)
    try:
        run_step("src/update_intraday_snapshots.py")
        print("Intraday update completed.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Intraday step failed: {e}")

    # Update daily history (incremental)
    try:
        run_step("src/backfill_daily.py")
        print("Daily backfill completed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Daily backfill failed: {e}")

if __name__ == "__main__":
    main()
