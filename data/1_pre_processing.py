"""
Pre-processing step 1 for Bitget data: Merge major coin features to all symbols

This script merges close and volume from major coins (BTC, ETH, DOGE, SOL) to all other symbols using merge_asof.

Input structure:
  data_storage_bitget/step_0/{SYMBOL}/multi_metric.csv (output from step 0)
  data_storage_bitget/step_0_large_coins/{COIN}USDT-LINEAR/multi_metric.csv (BTC, ETH, DOGE, SOL reference data)

Output:
  data_storage_bitget/step_1/{SYMBOL}/matched_data.csv
"""

from pathlib import Path
import pandas as pd
import shutil

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
OUTPUT_DIR = BASE_DATA_DIR / "output"
INPUT_ROOT = OUTPUT_DIR / "step_0"
OUTPUT_ROOT = OUTPUT_DIR / "step_1"

# Major coins for close price and volume features
MAJOR_COINS = ["BTC", "ETH", "DOGE", "SOL"]
MAJOR_COIN_PATHS = {
    coin: OUTPUT_DIR / "step_0_large_coins" / f"{coin}USDT-LINEAR" / "multi_metric.csv"
    for coin in MAJOR_COINS
}

# FNG Index
FNG_FILE = BASE_DATA_DIR / "fng" / "FNG_2024-01-01_to_2026-01-20.csv"


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Safely read CSV file, return None if not exists or error"""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
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


def load_major_coin_data(coin: str) -> pd.DataFrame | None:
    """Load close and volume from major coin (BTC, ETH, DOGE, SOL)
    
    Returns DataFrame with columns:
    - timestamp_nano
    - {coin}_close, {coin}_volume (prefixed with lowercase coin name)
    """
    path = MAJOR_COIN_PATHS.get(coin)
    if not path or not path.exists():
        print(f"[WARN] {coin} data not found at {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        if "timestamp_nano" not in df.columns:
            print(f"[WARN] {coin} missing timestamp_nano column")
            return None
        
        required_cols = ["close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[WARN] {coin} missing columns: {missing}")
            return None
        
        df = _prepare_time(df, "timestamp_nano")
        
        # Select timestamp + close + volume
        result = df[["timestamp_nano", "close", "volume"]].copy()
        
        # Rename with coin prefix (lowercase)
        result = result.rename(columns={
            "close": f"{coin.lower()}_close",
            "volume": f"{coin.lower()}_volume"
        })
        
        # Remove duplicates (keep last)
        result = result.drop_duplicates(subset=["timestamp_nano"], keep="last")
        
        return result.sort_values("timestamp_nano")
    except Exception as e:
        print(f"[ERROR] Loading {coin}: {e}")
        return None


def load_major_coin_lunar(coin: str) -> pd.DataFrame | None:
    """Load LUNAR data from major coin (BTC, ETH, DOGE, SOL)
    
    Returns DataFrame with columns prefixed with {coin}_lunar_
    """
    path = MAJOR_COIN_PATHS.get(coin)
    if not path or not path.exists():
        print(f"[WARN] {coin} LUNAR data not found at {path}")
        return None
    
    try:
        df = pd.read_csv(path)
        if "timestamp_nano" not in df.columns:
            print(f"[WARN] {coin} LUNAR missing timestamp_nano column")
            return None
        
        df = _prepare_time(df, "timestamp_nano")
        
        # Lunar columns from step_0 (prefixed with lunar_)
        lunar_cols = [c for c in df.columns if c.startswith("lunar_")]
        
        if not lunar_cols:
            print(f"[WARN] {coin} has no lunar_ columns")
            return None
        
        # Select timestamp + lunar columns
        result = df[["timestamp_nano"] + lunar_cols].copy()
        
        # Rename lunar_ columns to {coin}_lunar_ (e.g. lunar_sentiment -> btc_lunar_sentiment)
        rename_dict = {col: f"{coin.lower()}_{col}" for col in lunar_cols}
        result = result.rename(columns=rename_dict)
        
        # Remove duplicates (keep last)
        result = result.drop_duplicates(subset=["timestamp_nano"], keep="last")
        
        return result.sort_values("timestamp_nano")
    except Exception as e:
        print(f"[ERROR] Loading {coin} LUNAR: {e}")
        return None


def load_fng() -> pd.DataFrame | None:
    """Load Fear & Greed Index data
    
    Returns DataFrame with columns:
    - timestamp_nano
    - fng
    """
    if not FNG_FILE.exists():
        print(f"[WARN] FNG data not found at {FNG_FILE}")
        return None
    
    try:
        df = pd.read_csv(FNG_FILE)
        
        # Parse timestamp to nanoseconds
        if "timestamp" not in df.columns:
            print(f"[WARN] FNG missing timestamp column")
            return None
        
        df["timestamp_nano"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").astype("int64")
        df = df.dropna(subset=["timestamp_nano"])
        
        # Normalize fear_greed column
        if "fear_greed" in df.columns:
            df["fng"] = pd.to_numeric(df["fear_greed"], errors="coerce")
        elif "fng" not in df.columns:
            print(f"[WARN] FNG file missing fear_greed/fng column")
            return None
        
        # Select timestamp + fng
        result = df[["timestamp_nano", "fng"]].copy()
        
        # Remove duplicates (keep last)
        result = result.drop_duplicates(subset=["timestamp_nano"], keep="last")
        
        return result.sort_values("timestamp_nano")
    except Exception as e:
        print(f"[ERROR] Loading FNG: {e}")
        return None


def process_symbol_dir(sym_dir: Path, major_coins_dict: dict, major_coins_lunar_dict: dict, fng_df: pd.DataFrame):
    """Process single symbol directory and merge with major coin features"""
    symbol = sym_dir.name
    input_path = sym_dir / "multi_metric.csv"
    
    # Check if directory is empty
    if not any(sym_dir.iterdir()):
        print(f"[SKIP] {symbol}: Directory is empty")
        return
    
    base_df = _read_csv_safe(input_path)
    if base_df is None:
        print(f"[SKIP] {symbol}: multi_metric.csv missing")
        return
    
    required_cols = {"timestamp_nano", "timestamp_iso", "symbol", "open", "high", "low", "close", "volume"}
    missing = required_cols.difference(base_df.columns)
    if missing:
        print(f"[WARN] {symbol}: multi_metric.csv missing columns {missing}, skipping")
        return
    
    base_df = _prepare_time(base_df, "timestamp_nano")
    base_df = base_df.sort_values("timestamp_nano").reset_index(drop=True)
    
    # Base DataFrame + instrument_id
    merged = base_df.copy()
    merged["instrument_id"] = merged["symbol"]
    
    # Merge major coin OHLCV features using merge_asof (backward fill)
    for coin_name, coin_df in major_coins_dict.items():
        if coin_df is not None and not coin_df.empty:
            merged = pd.merge_asof(
                merged.sort_values("timestamp_nano"),
                coin_df,
                on="timestamp_nano",
                direction="backward"
            )
            
            # Forward fill coin features
            coin_cols = [c for c in coin_df.columns if c != "timestamp_nano"]
            merged[coin_cols] = merged[coin_cols].ffill()
    
    # Merge major coin LUNAR features using merge_asof (backward fill)
    for coin_name, lunar_df in major_coins_lunar_dict.items():
        if lunar_df is not None and not lunar_df.empty:
            merged = pd.merge_asof(
                merged.sort_values("timestamp_nano"),
                lunar_df,
                on="timestamp_nano",
                direction="backward"
            )
            
            # Forward fill lunar features
            lunar_cols = [c for c in lunar_df.columns if c != "timestamp_nano"]
            merged[lunar_cols] = merged[lunar_cols].ffill()
    
    # Merge FNG (global) using merge_asof (backward fill)
    if fng_df is not None and not fng_df.empty:
        merged = pd.merge_asof(
            merged.sort_values("timestamp_nano"),
            fng_df,
            on="timestamp_nano",
            direction="backward"
        )
        
        # Forward fill FNG
        merged["fng"] = merged["fng"].ffill()
    
    # Column ordering
    base_cols = ["timestamp_nano", "timestamp_iso", "instrument_id", "open", "high", "low", "close", "volume"]
    depth_cols = sorted([c for c in merged.columns if c.startswith("depth_")])
    lunar_cols = sorted([c for c in merged.columns if c.startswith("lunar_") and not any(c.startswith(f"{coin.lower()}_lunar_") for coin in MAJOR_COINS)])
    major_coin_ohlcv_cols = sorted([c for c in merged.columns if any(c.startswith(f"{coin.lower()}_") for coin in MAJOR_COINS) and not "_lunar_" in c])
    major_coin_lunar_cols = sorted([c for c in merged.columns if any(c.startswith(f"{coin.lower()}_lunar_") for coin in MAJOR_COINS)])
    fng_cols = ["fng"] if "fng" in merged.columns else []
    
    # Remove symbol column (we have instrument_id)
    if "symbol" in merged.columns:
        merged = merged.drop(columns=["symbol"])
    
    # Build final column order
    all_ordered = base_cols + depth_cols + lunar_cols + major_coin_ohlcv_cols + major_coin_lunar_cols + fng_cols
    remaining = [c for c in merged.columns if c not in all_ordered]
    final_cols = [c for c in all_ordered if c in merged.columns] + remaining
    merged = merged[final_cols]
    
    # Remove any duplicate columns (from merge_asof)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    # Fill empty entries with 0
    merged = merged.replace("", pd.NA).fillna(0)
    
    # Write output
    out_dir = OUTPUT_ROOT / symbol
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "matched_data.csv"
    merged.to_csv(out_file, index=False)
    print(f"[OK] {symbol}: merged rows={len(merged)} -> {out_file}")


def run():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load major coins OHLCV data (close + volume)
    major_coins_dict = {}
    for coin in MAJOR_COINS:
        coin_df = load_major_coin_data(coin)
        if coin_df is not None:
            major_coins_dict[coin] = coin_df
            print(f"[OK] {coin} OHLCV loaded: {len(coin_df)} rows")
        else:
            major_coins_dict[coin] = None
    
    # Load major coins LUNAR data
    major_coins_lunar_dict = {}
    for coin in MAJOR_COINS:
        lunar_df = load_major_coin_lunar(coin)
        if lunar_df is not None:
            major_coins_lunar_dict[coin] = lunar_df
            lunar_cols_count = len([c for c in lunar_df.columns if c != "timestamp_nano"])
            print(f"[OK] {coin} LUNAR loaded: {len(lunar_df)} rows, {lunar_cols_count} features")
        else:
            major_coins_lunar_dict[coin] = None
    
    # Load FNG data
    fng_df = load_fng()
    if fng_df is None:
        print("[WARN] No FNG data found – fng column will be empty")
    else:
        print(f"[OK] FNG loaded: {len(fng_df)} rows")
    
    # Process all symbols
    symbol_count = 0
    for sym_dir in sorted(INPUT_ROOT.iterdir()):
        if not sym_dir.is_dir():
            continue
        process_symbol_dir(sym_dir, major_coins_dict, major_coins_lunar_dict, fng_df)
        symbol_count += 1
    
    print(f"\n[DONE] Processed {symbol_count} symbols")


if __name__ == "__main__":
    run()
