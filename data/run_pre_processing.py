"""
Run all pre-processing steps (0-6) automatically for Bitget data

Executes:
0. Pre-processing 0: Create multi-metric CSV from OHLCV + DEPTH + LUNAR
1. Pre-processing 1: Merge major coin features (BTC, ETH, DOGE, SOL) + FNG + Major Coin LUNAR
2. Pre-processing 2: Cleanup timestamps and create readable format
3. Pre-processing 3: Merge all symbols into one final dataset
4. Pre-processing 4: Data quality check & fix (NaN, empty values -> 0)
5. Pre-processing 5: Behavioral cluster encoding (market categories)
6. Pre-processing 6: Multi-horizon target generation (15min, 1h, 4h, 12h)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Get the Python executable path
PYTHON_EXE = sys.executable

# Pre-processing scripts in order
SCRIPTS = [
    "0_pre_processing.py",
    "1_pre_processing.py",
    "2_pre_processing.py",
    "3_pre_processing.py",
    "4_pre_processing.py",
    "5_pre_processing.py",
    "6_pre_processing.py"
]

# Paths
BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FINAL_DIR = OUTPUT_DIR / "final"

# Copy step 6 output to final directory
INPUT_FILE = INTERMEDIATE_DIR / "step_6" / "all_matched_data_with_targets.csv"
FINAL_FILE = FINAL_DIR / "dataset.csv"

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


def copy_to_final() -> bool:
    """Copy step 5 output to final directory
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Copying final dataset to output folder")
    print("=" * 80)
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return False
    
    try:
        # Load data
        print(f"Loading: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        
        # Create final directory and save
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(FINAL_FILE, index=False)
        print(f"\n[SUCCESS] Final dataset saved to: {FINAL_FILE}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Copy failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all pre-processing steps sequentially"""
    print("=" * 80)
    print("Starting Pre-Processing Pipeline for Bitget Data")
    print("=" * 80)
    
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[Step {i}/{len(SCRIPTS) + 1}] Executing {script}...")
        
        if not run_script(script):
            print(f"\n[ABORT] Pipeline stopped at step {i}")
            sys.exit(1)
    
    # Copy to final directory as last step
    print(f"\n[Step {len(SCRIPTS) + 1}/{len(SCRIPTS) + 1}] Copying to final directory...")
    if not copy_to_final():
        print(f"\n[ABORT] Pipeline stopped at copy step")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("ALL PRE-PROCESSING STEPS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal dataset: {FINAL_FILE}")


if __name__ == "__main__":
    main()
