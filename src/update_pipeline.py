import subprocess
import sys

def run_step(script_path: str):
    subprocess.run([sys.executable, script_path], check=True)

def main():
    run_step("src/update_intraday_snapshots.py")
    print("Intraday update completed.")

if __name__ == "__main__":
    main()
