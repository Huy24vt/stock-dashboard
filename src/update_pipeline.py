import subprocess
import sys

def run_step(script_path: str):
    result = subprocess.run([sys.executable, script_path], check=True, text=True)
    return result

def main():
    run_step("src/fetch_price.py")
    run_step("src/process_price.py")
    print("Update completed.")

if __name__ == "__main__":
    main()