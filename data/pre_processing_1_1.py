"""
Pre-processing step 1.1: Merge OHLCV and METRICS for major coins (BTC, ETH, DOGE, SOL)

This script merges OHLCV.csv and METRICS.csv for each major coin and outputs
a single merged_coin_data.csv file containing:
- timestamp_nano, timestamp_iso
- close (from OHLCV)
- open_interest, funding_rate, long_short_ratio (from METRICS)

Input structure (new):
  dataset_storage/coin_specific_data/{COIN}/{COIN}USDT-LINEAR/
    - OHLCV.csv
    - METRICS.csv

Output:
  dataset_storage/coin_specific_data/{COIN}/{COIN}USDT-LINEAR/
    - merged_coin_data.csv
"""

from pathlib import Path
import pandas as pd

BASE_DATA_DIR = Path(__file__).resolve().parent / "dataset_storage"
COIN_DATA_ROOT = BASE_DATA_DIR / "coin_specific_data"

MAJOR_COINS = ["BTC", "ETH", "DOGE", "SOL"]


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Safely read CSV file, return None if not exists or error"""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
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


def merge_coin_data(coin: str) -> bool:
    """
    Merge OHLCV and METRICS for a single major coin.
    
    Args:
        coin: Coin symbol (BTC, ETH, DOGE, SOL)
        
    Returns:
        True if successful, False otherwise
    """
    coin_dir = COIN_DATA_ROOT / coin / f"{coin}USDT-LINEAR"
    
    if not coin_dir.exists():
        print(f"[SKIP] {coin}: Directory not found at {coin_dir}")
        return False
    
    ohlcv_path = coin_dir / "OHLCV.csv"
    metrics_path = coin_dir / "METRICS.csv"
    
    # Load OHLCV
    ohlcv = _read_csv_safe(ohlcv_path)
    if ohlcv is None or ohlcv.empty:
        print(f"[SKIP] {coin}: OHLCV.csv missing or empty")
        return False
    
    # Validate OHLCV columns
    required_ohlcv = {"timestamp_nano", "timestamp_iso", "close"}
    missing = required_ohlcv.difference(ohlcv.columns)
    if missing:
        print(f"[WARN] {coin}: OHLCV missing columns {missing}, skipping")
        return False
    
    # Prepare OHLCV timestamps
    ohlcv = _prepare_time(ohlcv, "timestamp_nano")
    ohlcv = ohlcv.sort_values("timestamp_nano").reset_index(drop=True)
    
    # Keep only required columns from OHLCV
    base_df = ohlcv[["timestamp_nano", "timestamp_iso", "close"]].copy()
    
    # Load METRICS
    metrics = _read_csv_safe(metrics_path)
    if metrics is None or metrics.empty:
        print(f"[WARN] {coin}: METRICS.csv missing or empty - creating output with close only")
        # Output with just close price
        out_path = coin_dir / "merged_coin_data.csv"
        base_df.to_csv(out_path, index=False)
        print(f"[OK] {coin}: {len(base_df)} rows (OHLCV only) -> {out_path}")
        return True
    
    # Validate METRICS columns
    expected_metrics = {"timestamp_nano", "open_interest", "funding_rate", "long_short_ratio"}
    available_metrics = expected_metrics.intersection(metrics.columns)
    
    if "timestamp_nano" not in metrics.columns:
        print(f"[WARN] {coin}: METRICS missing timestamp_nano, skipping metrics merge")
        out_path = coin_dir / "merged_coin_data.csv"
        base_df.to_csv(out_path, index=False)
        print(f"[OK] {coin}: {len(base_df)} rows (OHLCV only) -> {out_path}")
        return True
    
    # Prepare METRICS timestamps
    metrics = _prepare_time(metrics, "timestamp_nano")
    
    # Select available metric columns
    metrics_cols = ["timestamp_nano"] + [c for c in ["open_interest", "funding_rate", "long_short_ratio"] 
                                          if c in metrics.columns]
    metrics = metrics[metrics_cols].copy()
    
    # Remove duplicates (keep last value per timestamp)
    metrics = metrics.drop_duplicates(subset=["timestamp_nano"], keep="last")
    metrics = metrics.sort_values("timestamp_nano")
    
    # Merge OHLCV with METRICS using merge_asof (backward fill)
    merged = pd.merge_asof(
        base_df,
        metrics,
        on="timestamp_nano",
        direction="backward"
    )
    
    # Forward fill missing metric values
    metric_feature_cols = [c for c in merged.columns if c not in ["timestamp_nano", "timestamp_iso", "close"]]
    if metric_feature_cols:
        merged[metric_feature_cols] = merged[metric_feature_cols].ffill()
    
    # Fill remaining NaN with 0
    merged = merged.fillna(0)
    
    # Output
    out_path = coin_dir / "merged_coin_data.csv"
    merged.to_csv(out_path, index=False)
    
    metrics_info = ", ".join([c for c in metrics_cols if c != "timestamp_nano"])
    print(f"[OK] {coin}: {len(merged)} rows with [{metrics_info}] -> {out_path}")
    
    return True


def run():
    """Process all major coins"""
    if not COIN_DATA_ROOT.exists():
        print(f"[WARN] Coin data root not found: {COIN_DATA_ROOT}")
        print("[INFO] Creating directory structure (will be populated by user)")
        COIN_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        return
    
    print("=" * 80)
    print("Pre-processing 1.1: Merging OHLCV + METRICS for major coins")
    print("=" * 80)
    
    success_count = 0
    for coin in MAJOR_COINS:
        if merge_coin_data(coin):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Completed: {success_count}/{len(MAJOR_COINS)} major coins processed")
    print("=" * 80)


if __name__ == "__main__":
    run()
