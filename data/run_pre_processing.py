"""
Run all pre-processing steps (0-4) automatically for Bitget data

Executes:
0. Pre-processing 0: Create multi-metric CSV from OHLCV + DEPTH + LUNAR
1. Pre-processing 1: Merge major coin features (BTC, ETH, DOGE, SOL) + FNG + Major Coin LUNAR
2. Pre-processing 2: Cleanup timestamps and create readable format
3. Pre-processing 3: Merge all symbols into one final dataset
4. Pre-processing 4: Data quality check & fix (NaN, empty values -> 0)
5. Time filter: Filter dataset to specified date range
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
    "4_pre_processing.py"
]

# ============================================================================
# TIME FILTER CONFIGURATION
# ============================================================================
# Format: "DD.MM.YY" or "DD.MM.YYYY"
DATE_FROM = "08.01.24"  # Start date (inclusive)
DATE_TO = "20.01.26"    # End date (inclusive)
# ============================================================================

# Paths
BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
FINAL_DIR = OUTPUT_DIR / "final"

# Input from step 4, output to final
INPUT_FILE = INTERMEDIATE_DIR / "step_3" / "all_matched_data_clean.csv"
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


def parse_date(date_str: str) -> datetime:
    """Parse date string in DD.MM.YY or DD.MM.YYYY format"""
    for fmt in ["%d.%m.%y", "%d.%m.%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}. Use format DD.MM.YY or DD.MM.YYYY")


def apply_time_filter() -> bool:
    """Apply time filter to the final dataset
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Applying Time Filter: {DATE_FROM} to {DATE_TO}")
    print("=" * 80)
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return False
    
    try:
        # Parse filter dates
        start_date = parse_date(DATE_FROM)
        end_date = parse_date(DATE_TO)
        
        # Add end of day to end_date for inclusive filtering
        end_date = end_date.replace(hour=23, minute=59, second=59)
        
        print(f"Filter range: {start_date} to {end_date}")
        
        # Load data
        print(f"Loading: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        original_rows = len(df)
        print(f"Original rows: {original_rows:,}")
        
        # Parse timestamp column
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        
        # Apply filter
        mask = (df['timestamp_dt'] >= start_date) & (df['timestamp_dt'] <= end_date)
        df_filtered = df[mask].copy()
        
        # Remove helper column
        df_filtered = df_filtered.drop(columns=['timestamp_dt'])
        
        filtered_rows = len(df_filtered)
        removed_rows = original_rows - filtered_rows
        
        print(f"Filtered rows: {filtered_rows:,}")
        print(f"Removed rows: {removed_rows:,} ({100*removed_rows/original_rows:.2f}%)")
        
        # Get actual date range in filtered data
        df_filtered_check = df_filtered.copy()
        df_filtered_check['ts'] = pd.to_datetime(df_filtered_check['timestamp'])
        actual_start = df_filtered_check['ts'].min()
        actual_end = df_filtered_check['ts'].max()
        print(f"Actual date range in filtered data: {actual_start} to {actual_end}")
        
        # Create final directory and save filtered data
        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(FINAL_FILE, index=False)
        print(f"\n[SUCCESS] Final dataset saved to: {FINAL_FILE}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Time filter failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all pre-processing steps sequentially"""
    print("=" * 80)
    print("Starting Pre-Processing Pipeline for Bitget Data")
    print(f"Time Filter: {DATE_FROM} to {DATE_TO}")
    print("=" * 80)
    
    for i, script in enumerate(SCRIPTS, 1):
        print(f"\n[Step {i}/{len(SCRIPTS) + 1}] Executing {script}...")
        
        if not run_script(script):
            print(f"\n[ABORT] Pipeline stopped at step {i}")
            sys.exit(1)
    
    # Apply time filter as final step
    print(f"\n[Step {len(SCRIPTS) + 1}/{len(SCRIPTS) + 1}] Applying time filter...")
    if not apply_time_filter():
        print(f"\n[ABORT] Pipeline stopped at time filter step")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("ALL PRE-PROCESSING STEPS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal dataset: {FINAL_FILE}")
    print(f"Date range: {DATE_FROM} to {DATE_TO}")


if __name__ == "__main__":
    main()
