"""
Bitget Data Feature Engineering Pipeline for Kronos Transformer

This script transforms raw Bitget OHLCV + major coin data into return-based
features suitable for time series forecasting with the Kronos transformer.

Key transformations:
1. Replace absolute OHLCV with relative/return-based features
2. Compute returns at multiple horizons (1h, 4h, 24h)
3. Normalize major coin features to returns instead of absolute prices
4. Add volatility and relative volume features
5. Keep FNG as-is (already 0-100 range)

Author: Kronos ML Team
Date: 2026-01-21
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

class BitgetPreprocessConfig:
    """Configuration for Bitget data preprocessing."""
    
    def __init__(self):
        # Paths
        self.raw_data_path = Path(__file__).parent.parent.parent.parent / "data" / "data_storage_bitget" / "final" / "all_matched_data.csv"
        self.output_dir = Path(__file__).parent.parent / "data" / "dataset"
        self.output_filename = "bitget_engineered.csv"
        self.processed_datasets_dir = Path(__file__).parent.parent / "data" / "processed_datasets"
        
        # Major coins to include (selectable)
        # Options: ['btc', 'eth', 'sol', 'doge']
        self.major_coins = ['btc', 'eth']  # Start with BTC/ETH only
        
        # Return horizons (in bars/hours)
        self.return_horizons = [1, 4, 24]
        
        # Rolling window sizes for volatility/volume features
        self.volatility_window = 24  # 24 hours
        self.volume_ma_window = 24   # 24 hours
        
        # Date ranges for train/val split
        self.train_start = "2024-01-01"
        self.train_end = "2025-08-31"
        self.val_start = "2025-09-01"
        self.val_end = "2026-01-20"
        
        # Minimum data requirements
        self.min_rows_per_symbol = 200  # At least 200 rows (lookback + predict + buffer)
        
        # Columns in raw data
        self.datetime_col = "timestamp"
        self.symbol_col = "instrument_id"
        
        # Original OHLCV columns
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Major coin columns in raw data (from preprocessing)
        # Format: {coin}_close, {coin}_volume
        self.major_coin_cols_template = {
            'btc': {'close': 'btc_close', 'volume': 'btc_volume'},
            'eth': {'close': 'eth_close', 'volume': 'eth_volume'},
            'sol': {'close': 'sol_close', 'volume': 'sol_volume'},
            'doge': {'close': 'doge_close', 'volume': 'doge_volume'},
        }
        
        # FNG column
        self.fng_col = 'fng'
        
        # Lifecycle column
        self.ts_since_listing_col = 'ts_since_listing'


# =============================================================================
# Feature Engineering Functions
# =============================================================================

def compute_returns(df: pd.DataFrame, price_col: str, horizons: List[int]) -> pd.DataFrame:
    """
    Compute percentage returns at multiple horizons.
    
    Returns are computed as: (price_t - price_{t-h}) / price_{t-h}
    This gives the return looking BACKWARD (what happened in the last h bars).
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        horizons: List of return horizons (e.g., [1, 4, 24])
    
    Returns:
        DataFrame with return columns added
    """
    result = df.copy()
    for h in horizons:
        col_name = f'returns_{h}h'
        result[col_name] = df[price_col].pct_change(periods=h)
    return result


def compute_relative_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert absolute OHLC to relative features based on close price.
    
    Features created:
    - hl_range: (high - low) / close  -> Intrabar volatility proxy
    - oc_range: (close - open) / close  -> Bar direction/body size
    - high_rel: (high - close) / close  -> Upper wick
    - low_rel: (close - low) / close   -> Lower wick
    
    Args:
        df: DataFrame with OHLCV columns
    
    Returns:
        DataFrame with relative OHLC features
    """
    result = df.copy()
    close = df['close']
    
    # Relative ranges
    result['hl_range'] = (df['high'] - df['low']) / close
    result['oc_range'] = (df['close'] - df['open']) / close
    result['high_rel'] = (df['high'] - df['close']) / close
    result['low_rel'] = (df['close'] - df['low']) / close
    
    return result


def compute_volatility(df: pd.DataFrame, returns_col: str, window: int) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation of returns).
    
    Args:
        df: DataFrame with returns column
        returns_col: Column name for returns
        window: Rolling window size
    
    Returns:
        DataFrame with volatility column added
    """
    result = df.copy()
    result[f'volatility_{window}h'] = df[returns_col].rolling(window=window, min_periods=1).std()
    return result


def compute_relative_volume(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute volume relative to moving average.
    
    volume_rel = volume / MA(volume, window)
    
    This normalizes volume to show whether current volume is above/below average.
    
    Args:
        df: DataFrame with volume column
        window: Rolling window for MA
    
    Returns:
        DataFrame with relative volume column
    """
    result = df.copy()
    volume_ma = df['volume'].rolling(window=window, min_periods=1).mean()
    result['volume_rel'] = df['volume'] / (volume_ma + 1e-10)  # Avoid division by zero
    
    # Also compute volume percent change
    result['volume_pct_change'] = df['volume'].pct_change()
    
    return result


def compute_major_coin_returns(df: pd.DataFrame, coin: str, cols: Dict[str, str]) -> pd.DataFrame:
    """
    Convert major coin absolute prices/volumes to returns.
    
    Args:
        df: DataFrame with major coin columns
        coin: Coin name (e.g., 'btc')
        cols: Dict with 'close' and 'volume' column names
    
    Returns:
        DataFrame with major coin return features
    """
    result = df.copy()
    
    close_col = cols['close']
    volume_col = cols['volume']
    
    if close_col in df.columns:
        # Compute returns for major coin
        result[f'{coin}_returns_1h'] = df[close_col].pct_change(periods=1)
        result[f'{coin}_returns_4h'] = df[close_col].pct_change(periods=4)
        result[f'{coin}_returns_24h'] = df[close_col].pct_change(periods=24)
        
        # Volatility of major coin
        result[f'{coin}_volatility_24h'] = df[close_col].pct_change().rolling(window=24, min_periods=1).std()
    
    if volume_col in df.columns:
        # Relative volume for major coin
        vol_ma = df[volume_col].rolling(window=24, min_periods=1).mean()
        result[f'{coin}_volume_rel'] = df[volume_col] / (vol_ma + 1e-10)
    
    return result


def normalize_ts_since_listing(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Log-normalize ts_since_listing to compress the scale.
    
    log_ts = log(ts_since_listing + 1)
    
    This compresses large values while preserving relative ordering.
    
    Args:
        df: DataFrame with ts_since_listing column
        col: Column name
    
    Returns:
        DataFrame with normalized ts_since_listing
    """
    result = df.copy()
    if col in df.columns:
        result[f'{col}_log'] = np.log1p(df[col])  # log(x + 1)
    return result


def normalize_fng(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Normalize FNG from 0-100 to -1 to 1 range.
    
    fng_norm = (fng - 50) / 50
    
    This centers FNG at 0 with range [-1, 1].
    
    Args:
        df: DataFrame with FNG column
        col: Column name
    
    Returns:
        DataFrame with normalized FNG
    """
    result = df.copy()
    if col in df.columns:
        result['fng_norm'] = (df[col] - 50) / 50
    return result


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_symbol(df_symbol: pd.DataFrame, config: BitgetPreprocessConfig) -> pd.DataFrame:
    """
    Process a single symbol's data through the feature engineering pipeline.
    
    Args:
        df_symbol: DataFrame for one symbol (already indexed by datetime)
        config: Preprocessing configuration
    
    Returns:
        DataFrame with engineered features
    """
    df = df_symbol.copy()
    
    # 1. Compute returns at multiple horizons
    for h in config.return_horizons:
        col_name = f'returns_{h}h'
        df[col_name] = df['close'].pct_change(periods=h)
    
    # 2. Compute relative OHLC features
    df = compute_relative_ohlc(df)
    
    # 3. Compute volatility (from 1h returns)
    df = compute_volatility(df, 'returns_1h', config.volatility_window)
    
    # 4. Compute relative volume
    df = compute_relative_volume(df, config.volume_ma_window)
    
    # 5. Process major coin features
    for coin in config.major_coins:
        if coin in config.major_coin_cols_template:
            cols = config.major_coin_cols_template[coin]
            df = compute_major_coin_returns(df, coin, cols)
    
    # 6. Normalize ts_since_listing
    df = normalize_ts_since_listing(df, config.ts_since_listing_col)
    
    # 7. Normalize FNG
    df = normalize_fng(df, config.fng_col)
    
    return df


def get_engineered_feature_list(config: BitgetPreprocessConfig) -> List[str]:
    """
    Get the list of engineered feature columns based on config.
    
    Returns:
        List of feature column names
    """
    features = []
    
    # Return features (from close price)
    for h in config.return_horizons:
        features.append(f'returns_{h}h')
    
    # Relative OHLC
    features.extend(['hl_range', 'oc_range', 'high_rel', 'low_rel'])
    
    # Volatility
    features.append(f'volatility_{config.volatility_window}h')
    
    # Volume features
    features.extend(['volume_rel', 'volume_pct_change'])
    
    # Major coin features
    for coin in config.major_coins:
        features.extend([
            f'{coin}_returns_1h',
            f'{coin}_returns_4h',
            f'{coin}_returns_24h',
            f'{coin}_volatility_24h',
            f'{coin}_volume_rel',
        ])
    
    # Lifecycle
    features.append(f'{config.ts_since_listing_col}_log')
    
    # FNG
    features.append('fng_norm')
    
    return features


def load_raw_data(config: BitgetPreprocessConfig) -> pd.DataFrame:
    """Load raw Bitget data from CSV."""
    print(f"Loading raw data from: {config.raw_data_path}")
    
    if not config.raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {config.raw_data_path}")
    
    df = pd.read_csv(config.raw_data_path)
    print(f"Loaded {len(df):,} rows, {df[config.symbol_col].nunique()} symbols")
    
    return df


def run_feature_engineering(config: Optional[BitgetPreprocessConfig] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run the complete feature engineering pipeline.
    
    Args:
        config: Optional configuration (uses default if None)
    
    Returns:
        Tuple of (engineered DataFrame, feature list)
    """
    if config is None:
        config = BitgetPreprocessConfig()
    
    # Load raw data
    df = load_raw_data(config)
    
    # Parse datetime
    df[config.datetime_col] = pd.to_datetime(df[config.datetime_col])
    
    # Process each symbol
    processed_dfs = []
    symbols = df[config.symbol_col].unique()
    
    print(f"\nProcessing {len(symbols)} symbols...")
    for symbol in tqdm(symbols, desc="Feature engineering"):
        df_sym = df[df[config.symbol_col] == symbol].copy()
        df_sym = df_sym.sort_values(config.datetime_col)
        df_sym = df_sym.set_index(config.datetime_col)
        
        # Skip symbols with insufficient data
        if len(df_sym) < config.min_rows_per_symbol:
            continue
        
        # Process features
        df_processed = process_symbol(df_sym, config)
        df_processed[config.symbol_col] = symbol
        df_processed = df_processed.reset_index()
        
        processed_dfs.append(df_processed)
    
    # Combine all symbols
    result = pd.concat(processed_dfs, ignore_index=True)
    
    # Get feature list
    feature_list = get_engineered_feature_list(config)
    
    # Drop rows with NaN in critical features (from rolling windows)
    # Keep first 24 rows with potential NaNs but fill forward
    result = result.sort_values([config.symbol_col, config.datetime_col])
    
    print(f"\nFinal dataset: {len(result):,} rows, {result[config.symbol_col].nunique()} symbols")
    print(f"Features ({len(feature_list)}): {feature_list}")
    
    return result, feature_list


def save_engineered_data(df: pd.DataFrame, config: BitgetPreprocessConfig):
    """Save engineered data to CSV."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / config.output_filename
    
    print(f"\nSaving engineered data to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows")


def create_train_val_pickles(df: pd.DataFrame, feature_list: List[str], 
                              config: BitgetPreprocessConfig):
    """
    Create train/val pickle files for Kronos training.
    
    Args:
        df: Engineered DataFrame
        feature_list: List of feature columns
        config: Preprocessing config
    """
    config.processed_datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse dates
    df[config.datetime_col] = pd.to_datetime(df[config.datetime_col])
    
    train_start = pd.to_datetime(config.train_start)
    train_end = pd.to_datetime(config.train_end)
    val_start = pd.to_datetime(config.val_start)
    val_end = pd.to_datetime(config.val_end)
    
    train_dict = {}
    val_dict = {}
    
    symbols = df[config.symbol_col].unique()
    
    print(f"\nCreating train/val pickles...")
    print(f"Train: {config.train_start} to {config.train_end}")
    print(f"Val: {config.val_start} to {config.val_end}")
    
    for symbol in tqdm(symbols, desc="Creating pickles"):
        df_sym = df[df[config.symbol_col] == symbol].copy()
        df_sym = df_sym.set_index(config.datetime_col)
        
        # Select only feature columns that exist
        available_features = [f for f in feature_list if f in df_sym.columns]
        df_sym = df_sym[available_features]
        
        # Drop NaN rows
        df_sym = df_sym.dropna()
        
        # Split by time
        train_mask = (df_sym.index >= train_start) & (df_sym.index <= train_end)
        val_mask = (df_sym.index >= val_start) & (df_sym.index <= val_end)
        
        df_train = df_sym[train_mask]
        df_val = df_sym[val_mask]
        
        # Add to dicts if sufficient data
        min_len = config.min_rows_per_symbol
        if len(df_train) >= min_len:
            train_dict[symbol] = df_train
        if len(df_val) >= min_len:
            val_dict[symbol] = df_val
    
    # Save pickles
    train_path = config.processed_datasets_dir / "train_data.pkl"
    val_path = config.processed_datasets_dir / "val_data.pkl"
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(val_path, 'wb') as f:
        pickle.dump(val_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n[OK] Train: {len(train_dict)} symbols -> {train_path}")
    print(f"[OK] Val: {len(val_dict)} symbols -> {val_path}")
    
    # Print sample stats
    if train_dict:
        sample_sym = list(train_dict.keys())[0]
        sample_df = train_dict[sample_sym]
        print(f"\nSample ({sample_sym}):")
        print(f"  Train rows: {len(sample_df)}")
        print(f"  Features: {list(sample_df.columns)}")
        print(f"  Date range: {sample_df.index.min()} to {sample_df.index.max()}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for feature engineering pipeline."""
    print("=" * 80)
    print("BITGET DATA FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    config = BitgetPreprocessConfig()
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Major coins: {config.major_coins}")
    print(f"  Return horizons: {config.return_horizons}")
    print(f"  Volatility window: {config.volatility_window}h")
    print(f"  Train: {config.train_start} to {config.train_end}")
    print(f"  Val: {config.val_start} to {config.val_end}")
    
    # Run feature engineering
    df, feature_list = run_feature_engineering(config)
    
    # Save engineered CSV
    save_engineered_data(df, config)
    
    # Create train/val pickles
    create_train_val_pickles(df, feature_list, config)
    
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    
    return df, feature_list


if __name__ == "__main__":
    main()
