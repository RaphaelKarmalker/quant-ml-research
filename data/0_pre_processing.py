"""
Pre-processing step 0 for Bitget data: Create multi-metric CSV from raw OHLCV + DEPTH + LUNAR data

This script processes raw OHLCV data and enriches it by mapping:
1. DEPTH data (order book snapshots) - uses closest timestamp (backward direction)
2. LUNAR data (social/market metrics) - uses closest timestamp (hourly data mapped to each OHLCV row)

FILTER: Only processes coins where BOTH Lunar AND Depth data are available!

Input structure:
  data_storage_bitget/csv_data_all_bitget/{SYMBOL}/OHLCV.csv
  data_storage_bitget/csv_data_all_bitget/{SYMBOL}/LUNAR.csv
  data_storage_bitget/Depth_Bitget/{BASE_SYMBOL}/DEPTH.csv  (e.g. 2ZUSDT for 2ZUSDT-LINEAR)
  
  data_storage_bitget/large_coins_bitget/{SYMBOL}/OHLCV.csv
  data_storage_bitget/large_coins_bitget/{SYMBOL}/LUNAR.csv
  data_storage_bitget/Depth_Bitget/{BASE_SYMBOL}/DEPTH.csv

Output:
  data_storage_bitget/step_0/{SYMBOL}/multi_metric.csv
  data_storage_bitget/step_0_large_coins/{SYMBOL}/multi_metric.csv
"""

from pathlib import Path
import pandas as pd
import shutil

# ============================================================================
# CONFIGURATION: Require both Lunar and Depth data
# ============================================================================
REQUIRE_BOTH_LUNAR_AND_DEPTH = True  # Set to False to process all coins
# ============================================================================

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"

# Input directories
CSV_DATA_ALL_DIR = BASE_DATA_DIR / "csv_data_all_bitget"
LARGE_COINS_DIR = BASE_DATA_DIR / "large_coins_bitget"
DEPTH_DIR = BASE_DATA_DIR / "Depth_Bitget"

# Output directories
OUTPUT_DIR = BASE_DATA_DIR / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
OUTPUT_ALL_DIR = INTERMEDIATE_DIR / "step_0"
OUTPUT_LARGE_COINS_DIR = INTERMEDIATE_DIR / "step_0_large_coins"


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Safely read CSV file, return None if not exists or error"""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Reading {path}: {e}")
        return None


def _prepare_time(df: pd.DataFrame, col: str = "timestamp_nano") -> pd.DataFrame:
    """Convert timestamp column to int64 and sort"""
    if col not in df.columns:
        raise ValueError(f"Missing required timestamp column '{col}'")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[col])
    df[col] = df[col].astype("int64")
    df = df.sort_values(col)
    return df


def get_base_symbol(symbol: str) -> str:
    """Extract base symbol for DEPTH lookup (e.g. '2ZUSDT-LINEAR' -> '2ZUSDT')"""
    # Remove suffix like -LINEAR, -SWAP etc.
    if "-" in symbol:
        return symbol.split("-")[0]
    return symbol


def load_depth_data(symbol: str) -> pd.DataFrame | None:
    """Load DEPTH data for a symbol
    
    Returns DataFrame with depth columns prefixed with 'depth_'
    """
    base_symbol = get_base_symbol(symbol)
    depth_path = DEPTH_DIR / base_symbol / "DEPTH.csv"
    
    if not depth_path.exists():
        return None
    
    try:
        df = pd.read_csv(depth_path)
        if "timestamp_nano" not in df.columns:
            print(f"[WARN] DEPTH for {symbol} missing timestamp_nano column")
            return None
        
        df = _prepare_time(df, "timestamp_nano")
        
        # Select relevant columns
        depth_cols = ["ask_price", "bid_price", "ask_volume", "bid_volume", "spread", "mid_price"]
        available_cols = [c for c in depth_cols if c in df.columns]
        
        if not available_cols:
            print(f"[WARN] DEPTH for {symbol} has no relevant columns")
            return None
        
        result = df[["timestamp_nano"] + available_cols].copy()
        
        # Rename columns with depth_ prefix
        rename_dict = {col: f"depth_{col}" for col in available_cols}
        result = result.rename(columns=rename_dict)
        
        # Remove duplicates (keep last)
        result = result.drop_duplicates(subset=["timestamp_nano"], keep="last")
        
        return result.sort_values("timestamp_nano")
    except Exception as e:
        print(f"[ERROR] Loading DEPTH for {symbol}: {e}")
        return None


def load_lunar_data(lunar_path: Path) -> pd.DataFrame | None:
    """Load LUNAR data from given path
    
    Returns DataFrame with lunar columns prefixed with 'lunar_'
    """
    if not lunar_path.exists():
        return None
    
    try:
        df = pd.read_csv(lunar_path)
        if "timestamp_nano" not in df.columns:
            print(f"[WARN] LUNAR at {lunar_path} missing timestamp_nano column")
            return None
        
        df = _prepare_time(df, "timestamp_nano")
        
        # Lunar columns to keep (exclude duplicates with OHLCV like open, high, low, close)
        # Keep these social/market metrics
        lunar_cols = [
            "contributors_active", "contributors_created", "interactions",
            "posts_active", "posts_created", "sentiment", "spam",
            "alt_rank", "circulating_supply", "galaxy_score",
            "market_cap", "market_dominance", "social_dominance", "volume_24h",
            "market_categories"
        ]
        available_cols = [c for c in lunar_cols if c in df.columns]
        
        if not available_cols:
            print(f"[WARN] LUNAR at {lunar_path} has no relevant columns")
            return None
        
        result = df[["timestamp_nano"] + available_cols].copy()
        
        # Rename columns with lunar_ prefix
        rename_dict = {col: f"lunar_{col}" for col in available_cols}
        result = result.rename(columns=rename_dict)
        
        # Remove duplicates (keep last)
        result = result.drop_duplicates(subset=["timestamp_nano"], keep="last")
        
        return result.sort_values("timestamp_nano")
    except Exception as e:
        print(f"[ERROR] Loading LUNAR from {lunar_path}: {e}")
        return None


def process_symbol(sym_dir: Path, output_root: Path, is_large_coin: bool = False) -> bool:
    """Process single symbol directory and create multi-metric CSV
    
    Args:
        sym_dir: Path to symbol directory containing OHLCV.csv and optionally LUNAR.csv
        output_root: Output directory root
        is_large_coin: Whether this is a large coin (for logging)
    
    Returns:
        True if processed successfully, False otherwise
    """
    symbol = sym_dir.name
    ohlcv_path = sym_dir / "OHLCV.csv"
    lunar_path = sym_dir / "LUNAR.csv"
    
    # Check if directory is empty or OHLCV doesn't exist
    if not ohlcv_path.exists():
        print(f"[SKIP] {symbol}: OHLCV.csv missing")
        return False
    
    # PRE-CHECK: If REQUIRE_BOTH_LUNAR_AND_DEPTH is True, check availability before processing
    if REQUIRE_BOTH_LUNAR_AND_DEPTH:
        # Check if Lunar exists
        if not lunar_path.exists():
            print(f"[SKIP] {symbol}: LUNAR.csv missing (required)")
            return False
        
        # Check if Depth exists
        base_symbol = get_base_symbol(symbol)
        depth_path = DEPTH_DIR / base_symbol / "DEPTH.csv"
        if not depth_path.exists():
            print(f"[SKIP] {symbol}: DEPTH.csv missing for {base_symbol} (required)")
            return False
    
    # Load OHLCV
    ohlcv = _read_csv_safe(ohlcv_path)
    if ohlcv is None:
        print(f"[SKIP] {symbol}: Cannot read OHLCV.csv")
        return False
    
    required_ohlcv = {"timestamp_nano", "timestamp_iso", "symbol", "open", "high", "low", "close", "volume"}
    missing = required_ohlcv.difference(ohlcv.columns)
    if missing:
        print(f"[WARN] {symbol}: OHLCV missing columns {missing}, skipping")
        return False
    
    ohlcv = _prepare_time(ohlcv, "timestamp_nano")
    ohlcv = ohlcv.sort_values("timestamp_nano").reset_index(drop=True)
    
    # Start with OHLCV as base
    merged = ohlcv[["timestamp_nano", "timestamp_iso", "symbol", "open", "high", "low", "close", "volume"]].copy()
    
    # Track what was merged
    has_depth = False
    has_lunar = False
    
    # Load and merge DEPTH data using merge_asof (backward fill - closest timestamp <= OHLCV timestamp)
    depth_df = load_depth_data(symbol)
    if depth_df is not None and not depth_df.empty:
        merged = pd.merge_asof(
            merged.sort_values("timestamp_nano"),
            depth_df,
            on="timestamp_nano",
            direction="backward"
        )
        has_depth = True
    
    # Load and merge LUNAR data using merge_asof (backward fill - hourly data mapped to minute data)
    lunar_df = load_lunar_data(lunar_path)
    if lunar_df is not None and not lunar_df.empty:
        merged = pd.merge_asof(
            merged.sort_values("timestamp_nano"),
            lunar_df,
            on="timestamp_nano",
            direction="backward"
        )
        has_lunar = True
    
    # Remove any duplicate/unwanted columns that might have been added during merge
    # Only keep: base OHLCV cols + depth_ prefixed + lunar_ prefixed
    unwanted_cols = [c for c in merged.columns if c.endswith("_x") or c.endswith("_y")]
    if unwanted_cols:
        merged = merged.drop(columns=unwanted_cols)
    
    # Column ordering - strict column selection to avoid duplicates
    base_cols = ["timestamp_nano", "timestamp_iso", "symbol", "open", "high", "low", "close", "volume"]
    depth_cols = sorted([c for c in merged.columns if c.startswith("depth_")])
    lunar_cols = sorted([c for c in merged.columns if c.startswith("lunar_")])
    
    # Final columns: only base + prefixed columns (no duplicates possible)
    final_cols = base_cols + depth_cols + lunar_cols
    merged = merged[[c for c in final_cols if c in merged.columns]]
    
    # Ensure no duplicate columns exist
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    # Write output
    out_dir = output_root / symbol
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "multi_metric.csv"
    merged.to_csv(out_file, index=False)
    
    coin_type = "large_coin" if is_large_coin else "all"
    depth_status = "✓" if has_depth else "✗"
    lunar_status = "✓" if has_lunar else "✗"
    print(f"[OK] {symbol} ({coin_type}): rows={len(merged)}, depth={depth_status}, lunar={lunar_status} -> {out_file}")
    
    return True


def run():
    """Main entry point"""
    print("=" * 60)
    print("Step 0: Creating multi-metric CSVs from OHLCV + DEPTH + LUNAR")
    if REQUIRE_BOTH_LUNAR_AND_DEPTH:
        print("FILTER: Only processing coins with BOTH Lunar AND Depth data")
    print("=" * 60)
    
    # Create output directories
    OUTPUT_ALL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LARGE_COINS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check input directories
    if not CSV_DATA_ALL_DIR.exists():
        print(f"[ERROR] csv_data_all_bitget not found at {CSV_DATA_ALL_DIR}")
    if not LARGE_COINS_DIR.exists():
        print(f"[ERROR] large_coins_bitget not found at {LARGE_COINS_DIR}")
    if not DEPTH_DIR.exists():
        print(f"[WARN] Depth_Bitget not found at {DEPTH_DIR} - depth data will not be included")
    
    # Process csv_data_all_bitget
    print("\n" + "-" * 40)
    print("Processing csv_data_all_bitget...")
    print("-" * 40)
    
    all_count = 0
    all_success = 0
    if CSV_DATA_ALL_DIR.exists():
        for sym_dir in sorted(CSV_DATA_ALL_DIR.iterdir()):
            if not sym_dir.is_dir():
                continue
            all_count += 1
            if process_symbol(sym_dir, OUTPUT_ALL_DIR, is_large_coin=False):
                all_success += 1
    
    # Process large_coins_bitget
    print("\n" + "-" * 40)
    print("Processing large_coins_bitget...")
    print("-" * 40)
    
    large_count = 0
    large_success = 0
    if LARGE_COINS_DIR.exists():
        for sym_dir in sorted(LARGE_COINS_DIR.iterdir()):
            if not sym_dir.is_dir():
                continue
            large_count += 1
            if process_symbol(sym_dir, OUTPUT_LARGE_COINS_DIR, is_large_coin=True):
                large_success += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if REQUIRE_BOTH_LUNAR_AND_DEPTH:
        print("Filter: Only coins with BOTH Lunar AND Depth data")
    print(f"csv_data_all_bitget: {all_success}/{all_count} symbols processed")
    print(f"large_coins_bitget:  {large_success}/{large_count} symbols processed")
    print(f"Total processed: {all_success + large_success}/{all_count + large_count}")
    print(f"Output directories:")
    print(f"  - {OUTPUT_ALL_DIR}")
    print(f"  - {OUTPUT_LARGE_COINS_DIR}")
    print("[DONE]")


if __name__ == "__main__":
    run()


