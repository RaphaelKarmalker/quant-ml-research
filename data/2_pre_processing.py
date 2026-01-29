"""
Pre-processing step 2 for Bitget data: Cleanup and create readable timestamps

Reads step_1/*/matched_data.csv
Removes all *_timestamp_nano, timestamp_iso, symbol columns
Creates timestamp (readable) and ts_since_listing (row number starting from 1)

Input:
  data_storage_bitget/step_1/{SYMBOL}/matched_data.csv

Output:
  data_storage_bitget/step_2/{SYMBOL}/matched_data_filtered.csv
"""

from pathlib import Path
import pandas as pd

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
PROCESSED_ROOT = OUTPUT_DIR / "step_1"
FILTERED_ROOT = OUTPUT_DIR / "step_2"

INPUT_FILENAME = "matched_data.csv"
OUTPUT_FILENAME = "matched_data_filtered.csv"

# Columns to remove (timestamps that are not needed after readable timestamp is created)
DROP_COLS = {
    "timestamp_nano",
    "timestamp_iso",
    "btc_timestamp_nano",
    "eth_timestamp_nano",
    "doge_timestamp_nano",
    "sol_timestamp_nano",
    "fng_timestamp_nano",
    "symbol",
}

# Pattern-based columns to drop (any column matching these patterns)
DROP_PATTERNS = ["_timestamp_nano", "_timestamp_iso"]


def _load_csv(path: Path) -> pd.DataFrame | None:
    """Safely load CSV file"""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def _build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Build readable timestamp from timestamp_iso or timestamp_nano"""
    if "timestamp_iso" in df.columns:
        ts = pd.to_datetime(df["timestamp_iso"], utc=True, errors="coerce")
    elif "timestamp_nano" in df.columns:
        # nanoseconds -> datetime
        ts = pd.to_datetime(pd.to_numeric(df["timestamp_nano"], errors="coerce"), utc=True, unit="ns")
    else:
        raise ValueError("Neither timestamp_iso nor timestamp_nano present")
    ts = ts.dropna()
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S")


def process_directory(sym_dir: Path, dest_root: Path):
    """Process single symbol directory"""
    in_file = sym_dir / INPUT_FILENAME
    if not in_file.exists():
        return
    
    df = _load_csv(in_file)
    if df is None or df.empty:
        return
    
    # Create new timestamp column
    try:
        new_timestamp = _build_timestamp(df)
    except Exception as e:
        print(f"[SKIP] {sym_dir.name}: Timestamp creation failed: {e}")
        return
    
    # Sort by existing time column (use timestamp_nano if possible, else timestamp_iso)
    sort_key = None
    if "timestamp_nano" in df.columns:
        sort_key = ("timestamp_nano", True)  # numeric
    elif "timestamp_iso" in df.columns:
        sort_key = ("timestamp_iso", False)
    
    if sort_key:
        key, is_numeric = sort_key
        if is_numeric:
            df[key] = pd.to_numeric(df[key], errors="coerce")
        else:
            df[key] = pd.to_datetime(df[key], utc=True, errors="coerce")
        df = df.sort_values(key).reset_index(drop=True)
    
    # ts_since_listing (starting from 1)
    df["ts_since_listing"] = range(1, len(df) + 1)
    df["timestamp"] = new_timestamp
    
    # Remove explicit columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    
    # Remove pattern-based columns (e.g. any column ending with _timestamp_nano or _timestamp_iso)
    for pattern in DROP_PATTERNS:
        cols_to_drop.extend([c for c in df.columns if pattern in c])
    
    # Also remove any duplicate symbol columns that might have slipped through
    cols_to_drop.extend([c for c in df.columns if c.endswith("_x") or c.endswith("_y")])
    
    # Dedupe the drop list
    cols_to_drop = list(set(cols_to_drop))
    df = df.drop(columns=cols_to_drop, errors="ignore")
    
    # Ensure no duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Column order: timestamp, ts_since_listing, instrument_id followed by rest
    front = ["timestamp", "ts_since_listing"]
    if "instrument_id" in df.columns:
        front.append("instrument_id")
    remaining = [c for c in df.columns if c not in front]
    df = df[front + remaining]
    
    # Write output
    out_dir = dest_root / sym_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILENAME
    df.to_csv(out_path, index=False)
    print(f"[OK] {sym_dir.name}: {len(df)} rows -> {out_path}")


def run():
    if not PROCESSED_ROOT.exists():
        raise FileNotFoundError(f"Directory not found: {PROCESSED_ROOT}")
    FILTERED_ROOT.mkdir(parents=True, exist_ok=True)
    
    symbol_count = 0
    for sym_dir in sorted(PROCESSED_ROOT.iterdir()):
        if not sym_dir.is_dir():
            continue
        process_directory(sym_dir, FILTERED_ROOT)
        symbol_count += 1
    
    print(f"\n[DONE] Processed {symbol_count} symbols")


if __name__ == "__main__":
    run()
