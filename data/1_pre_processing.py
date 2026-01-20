"""
Pre-processing step 1 for Bitget data: Merge major coin features to all symbols

This script merges close and volume from major coins (BTC, ETH, DOGE, SOL) to all other symbols using merge_asof.

Input structure:
  data_storage_bitget/large_coins_bitget/{COIN}USDT-LINEAR/OHLCV.csv (BTC, ETH, DOGE, SOL reference data)
  data_storage_bitget/csv_data_all_bitget/{SYMBOL}/OHLCV.csv (all symbols)

Output:
  data_storage_bitget/step_1/{SYMBOL}/matched_data.csv
"""

from pathlib import Path
import pandas as pd
import shutil

BASE_DATA_DIR = Path(__file__).resolve().parent / "data_storage_bitget"
INPUT_ROOT = BASE_DATA_DIR / "csv_data_all_bitget"
OUTPUT_ROOT = BASE_DATA_DIR / "step_1"

# Major coins for close price and volume features
MAJOR_COINS = ["BTC", "ETH", "DOGE", "SOL"]
MAJOR_COIN_PATHS = {
    coin: BASE_DATA_DIR / "large_coins_bitget" / f"{coin}USDT-LINEAR" / "OHLCV.csv"
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


def process_symbol_dir(sym_dir: Path, major_coins_dict: dict, fng_df: pd.DataFrame):
    """Process single symbol directory and merge with major coin features"""
    symbol = sym_dir.name
    ohlcv_path = sym_dir / "OHLCV.csv"
    
    # Check if directory is empty
    if not any(sym_dir.iterdir()):
        print(f"[SKIP] {symbol}: Directory is empty")
        return
    
    ohlcv = _read_csv_safe(ohlcv_path)
    if ohlcv is None:
        print(f"[SKIP] {symbol}: OHLCV.csv missing")
        return
    
    required_ohlcv = {"timestamp_nano", "timestamp_iso", "symbol", "open", "high", "low", "close", "volume"}
    missing = required_ohlcv.difference(ohlcv.columns)
    if missing:
        print(f"[WARN] {symbol}: OHLCV missing columns {missing}, skipping")
        return
    
    ohlcv = _prepare_time(ohlcv, "timestamp_nano")
    ohlcv = ohlcv.sort_values("timestamp_nano").reset_index(drop=True)
    
    # Base DataFrame + instrument_id
    merged = ohlcv.copy()
    merged["instrument_id"] = merged["symbol"]
    
    # Merge major coin features using merge_asof (backward fill)
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
    major_coin_cols = sorted([c for c in merged.columns if any(c.startswith(f"{coin.lower()}_") for coin in MAJOR_COINS)])
    fng_cols = ["fng"] if "fng" in merged.columns else []
    others = [c for c in merged.columns if c not in base_cols + major_coin_cols + fng_cols]
    
    final_cols = base_cols + major_coin_cols + fng_cols + [c for c in others if c not in base_cols]
    merged = merged[final_cols]
    
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
    
    # Load major coins data (close + volume)
    major_coins_dict = {}
    for coin in MAJOR_COINS:
        coin_df = load_major_coin_data(coin)
        if coin_df is not None:
            major_coins_dict[coin] = coin_df
            print(f"[OK] {coin} loaded: {len(coin_df)} rows")
        else:
            major_coins_dict[coin] = None
    
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
        process_symbol_dir(sym_dir, major_coins_dict, fng_df)
        symbol_count += 1
    
    print(f"\n[DONE] Processed {symbol_count} symbols")


if __name__ == "__main__":
    run()
