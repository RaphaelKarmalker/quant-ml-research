"""
Run all pre-processing steps (1-3) automatically for Bitget data

Executes:
1. Pre-processing 1: Merge major coin features (BTC, ETH, DOGE, SOL) + FNG
2. Pre-processing 2: Cleanup timestamps and create readable format
3. Pre-processing 3: Merge all symbols into one final dataset
"""

import subprocess
import sys
from pathlib import Path

# Get the Python executable path
PYTHON_EXE = sys.executable

# Pre-processing scripts in order
SCRIPTS = [
    "1_pre_processing.py",
    "2_pre_processing.py",
    "3_pre_processing.py"
]

def run_script(script_name: str) -> bool:
    """Run a single pre-processing script
    
    Args:
        script_name: Name of the script to run
        
    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        return False
    
    print("\n" + "=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [PYTHON_EXE, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"[SUCCESS] {script_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] {script_name} failed: {e}")
        return False


def main():
    """Run all pre-processing steps sequentially"""
    print("=" * 80)
    print("Starting Pre-Processing Pipeline for Bitget Data")
    print("=" * 80)
    
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[Step {i}/{len(SCRIPTS)}] Executing {script}...")
        
        if not run_script(script):
            print(f"\n[ABORT] Pipeline stopped at step {i}")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ ALL PRE-PROCESSING STEPS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nFinal output: data_storage_bitget/final/all_matched_data.csv")


if __name__ == "__main__":
    main()
