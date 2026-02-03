"""
Pre-processing Step 6 for Bitget data: Multi-Horizon Target Generation

This script generates multiple prediction targets for different time horizons:
- 15min (1 candle)
- 1h (4 candles)
- 4h (16 candles)
- 12h (48 candles)

For each horizon, it creates:
1. Log Returns (Regression)
2. Direction (Classification: -1, 0, +1)
3. Strong Pump/Dump (Binary: 0/1)

Input:
  data_storage_bitget/output/intermediate/step_5/all_matched_data_encoded.csv

Output:
  data_storage_bitget/output/intermediate/step_6/all_matched_data_with_targets.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
LOOKBACK_WINDOW = 96  # 24h in 15min candles (for context)

PREDICTION_HORIZONS = {
    '15min': 1,   # 15min * 1 = 15min
    '1h': 4,      # 15min * 4 = 1h
    '4h': 16,     # 15min * 16 = 4h
    '12h': 48,    # 15min * 48 = 12h
}

# Strong move thresholds (percentage)
STRONG_MOVE_THRESHOLD = 0.03  # 3%
# ---------------------

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"

INPUT_FILE = INTERMEDIATE_DIR / "step_5" / "all_matched_data_encoded.csv"
OUTPUT_FOLDER = INTERMEDIATE_DIR / "step_6"
OUTPUT_FILE = OUTPUT_FOLDER / "all_matched_data_with_targets.csv"


def generate_targets_for_instrument(df_inst: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all target variables for a single instrument.
    
    Args:
        df_inst: DataFrame for one instrument, sorted by timestamp
        
    Returns:
        DataFrame with added target columns
    """
    df = df_inst.copy()
    
    # Ensure we have close prices
    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column in data")
    
    close = df['close'].values
    
    # Generate targets for each horizon
    for horizon_name, horizon_steps in PREDICTION_HORIZONS.items():
        
        # --- 1. LOG RETURNS (Regression) ---
        # log(close[t+h] / close[t])
        future_close = np.roll(close, -horizon_steps)
        log_returns = np.log(future_close / (close + 1e-10))  # Avoid log(0)
        
        # Set last N values to NaN (no future data)
        log_returns[-horizon_steps:] = np.nan
        
        df[f'target_log_return_{horizon_name}'] = log_returns
        
        # --- 2. DIRECTION (Classification: -1, 0, +1) ---
        # sign(close[t+h] - close[t])
        price_diff = future_close - close
        direction = np.sign(price_diff)
        direction[-horizon_steps:] = np.nan
        
        df[f'target_direction_{horizon_name}'] = direction
        
        # --- 3. SIMPLE RETURNS (for threshold calculation) ---
        simple_returns = (future_close - close) / (close + 1e-10)
        simple_returns[-horizon_steps:] = np.nan
        
        # --- 4. STRONG PUMP (Binary: 1 if return > threshold) ---
        strong_pump = (simple_returns > STRONG_MOVE_THRESHOLD).astype(float)
        strong_pump[-horizon_steps:] = np.nan
        
        df[f'target_strong_pump_{horizon_name}'] = strong_pump
        
        # --- 5. STRONG DUMP (Binary: 1 if return < -threshold) ---
        strong_dump = (simple_returns < -STRONG_MOVE_THRESHOLD).astype(float)
        strong_dump[-horizon_steps:] = np.nan
        
        df[f'target_strong_dump_{horizon_name}'] = strong_dump
    
    return df


def generate_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate targets for all instruments in the dataset.
    
    Args:
        df: Full dataset with multiple instruments
        
    Returns:
        DataFrame with target columns added
    """
    print("-" * 100)
    print(" GENERATING MULTI-HORIZON TARGETS")
    print("-" * 100)
    
    # Check if we have instrument_id
    if 'instrument_id' not in df.columns:
        raise ValueError("Missing 'instrument_id' column - cannot group by instrument")
    
    print(f"\nHorizons: {', '.join([f'{k} ({v} candles)' for k, v in PREDICTION_HORIZONS.items()])}")
    print(f"Strong move threshold: ±{STRONG_MOVE_THRESHOLD*100:.1f}%")
    
    # Sort by instrument and timestamp
    df = df.sort_values(['instrument_id', 'timestamp']).reset_index(drop=True)
    
    # Process each instrument separately
    print(f"\nProcessing {df['instrument_id'].nunique()} instruments...")
    
    result_dfs = []
    processed = 0
    
    for instrument_id, group in df.groupby('instrument_id'):
        # Generate targets for this instrument
        df_with_targets = generate_targets_for_instrument(group)
        result_dfs.append(df_with_targets)
        
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed} instruments...")
    
    print(f"✓ Processed all {processed} instruments")
    
    # Concatenate all results
    df_final = pd.concat(result_dfs, ignore_index=True)
    
    # Statistics
    print("\n" + "=" * 100)
    print(" TARGET STATISTICS")
    print("=" * 100)
    
    target_cols = [c for c in df_final.columns if c.startswith('target_')]
    
    print(f"\nTotal target columns created: {len(target_cols)}")
    print(f"\nTarget columns per horizon:")
    for horizon_name in PREDICTION_HORIZONS.keys():
        horizon_targets = [c for c in target_cols if horizon_name in c]
        print(f"  {horizon_name}: {len(horizon_targets)} targets")
        for col in sorted(horizon_targets):
            print(f"    - {col}")
    
    # Count valid samples (non-NaN targets)
    print(f"\n" + "-" * 100)
    print(f"Valid samples (with future data):")
    print("-" * 100)
    
    total_rows = len(df_final)
    
    for horizon_name in PREDICTION_HORIZONS.keys():
        col = f'target_log_return_{horizon_name}'
        valid_count = df_final[col].notna().sum()
        valid_pct = (valid_count / total_rows) * 100
        print(f"  {horizon_name}: {valid_count:,} / {total_rows:,} ({valid_pct:.1f}%)")
    
    # Class distribution for direction targets
    print(f"\n" + "-" * 100)
    print(f"Direction Target Distribution:")
    print("-" * 100)
    
    for horizon_name in PREDICTION_HORIZONS.keys():
        col = f'target_direction_{horizon_name}'
        value_counts = df_final[col].value_counts().sort_index()
        
        print(f"\n  {horizon_name}:")
        total_valid = value_counts.sum()
        for direction, count in value_counts.items():
            if not np.isnan(direction):
                pct = (count / total_valid) * 100
                label = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}.get(int(direction), "UNKNOWN")
                print(f"    {label:8} ({int(direction):+2}): {count:8,} ({pct:5.2f}%)")
    
    # Strong move statistics
    print(f"\n" + "-" * 100)
    print(f"Strong Move Distribution (>{STRONG_MOVE_THRESHOLD*100:.1f}%):")
    print("-" * 100)
    
    for horizon_name in PREDICTION_HORIZONS.keys():
        pump_col = f'target_strong_pump_{horizon_name}'
        dump_col = f'target_strong_dump_{horizon_name}'
        
        pump_count = (df_final[pump_col] == 1).sum()
        dump_count = (df_final[dump_col] == 1).sum()
        total_valid = df_final[pump_col].notna().sum()
        
        pump_pct = (pump_count / total_valid) * 100 if total_valid > 0 else 0
        dump_pct = (dump_count / total_valid) * 100 if total_valid > 0 else 0
        
        print(f"\n  {horizon_name}:")
        print(f"    Strong PUMP: {pump_count:8,} ({pump_pct:5.2f}%)")
        print(f"    Strong DUMP: {dump_count:8,} ({dump_pct:5.2f}%)")
    
    return df_final


def run():
    """Main entry point"""
    print("=" * 100)
    print(" Pre-processing Step 6: Multi-Horizon Target Generation")
    print("=" * 100)
    
    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    print(f"\nLoading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Generate targets
    df_with_targets = generate_all_targets(df)
    
    # Remove rows with NaN targets (last N rows per instrument where we don't have future data)
    print("\n" + "=" * 100)
    print(" REMOVING INCOMPLETE ROWS")
    print("=" * 100)
    
    original_rows = len(df_with_targets)
    
    # Drop rows where ANY target is NaN (conservative approach)
    target_cols = [c for c in df_with_targets.columns if c.startswith('target_')]
    df_clean = df_with_targets.dropna(subset=target_cols)
    
    removed_rows = original_rows - len(df_clean)
    
    print(f"\nOriginal rows: {original_rows:,}")
    print(f"Removed rows (no future data): {removed_rows:,} ({removed_rows/original_rows*100:.2f}%)")
    print(f"Final rows: {len(df_clean):,}")
    
    # Save
    print("\n" + "=" * 100)
    print(" SAVING DATA")
    print("=" * 100)
    
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Saved to: {OUTPUT_FILE}")
    print(f"  Rows: {len(df_clean):,}")
    print(f"  Columns: {len(df_clean.columns)}")
    print(f"  Target columns: {len(target_cols)}")
    
    # Summary of final dataset
    print("\n" + "-" * 100)
    print("Final Dataset Summary:")
    print("-" * 100)
    print(f"  Instruments: {df_clean['instrument_id'].nunique()}")
    print(f"  Time range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
    print(f"  Features: {len(df_clean.columns) - len(target_cols)}")
    print(f"  Targets: {len(target_cols)}")
    
    print("\n[DONE]")


if __name__ == "__main__":
    run()
