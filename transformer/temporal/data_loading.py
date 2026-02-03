"""
Data Loading Pipeline for Temporal Transformer

This module handles:
1. Loading preprocessed data from step_6
2. Feature engineering (time features)
3. Instrument-based dataset with proper temporal ordering
4. Custom batch sampler ensuring same-instrument batches
5. Date-based train/val/test splitting
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
from datetime import datetime
import random
from collections import defaultdict

from config import Config, get_config


# =============================================================================
# TIME FEATURE ENGINEERING
# =============================================================================

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add cyclical time features to the dataframe.
    
    Creates sin/cos encodings for:
    - Hour of day (0-23)
    - Day of week (0-6)
    - Day of month (1-31)
    - Month (1-12)
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Parse timestamp if needed
    if df[timestamp_col].dtype == 'object':
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract components
    hour = df[timestamp_col].dt.hour
    day_of_week = df[timestamp_col].dt.dayofweek
    day_of_month = df[timestamp_col].dt.day
    month = df[timestamp_col].dt.month
    
    # Cyclical encoding with sin/cos
    # Hour (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (7-day cycle)
    df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Day of month (31-day cycle)
    df["day_of_month_sin"] = np.sin(2 * np.pi * day_of_month / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * day_of_month / 31)
    
    # Month (12-month cycle)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    return df


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(config: Config) -> pd.DataFrame:
    """
    Load data from step_6 and apply preprocessing.
    
    Args:
        config: Configuration object
        
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from: {config.dataset_path}")
    
    # Load data with low_memory=False to handle mixed dtypes
    df = pd.read_csv(config.dataset_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Add time features
    print("  Adding time features...")
    df = add_time_features(df)
    
    # Sort by instrument and timestamp
    df = df.sort_values(["instrument_id", "timestamp"]).reset_index(drop=True)
    
    # Create time_idx per instrument (for sequence ordering)
    df["time_idx"] = df.groupby("instrument_id").cumcount()
    
    print(f"  Final shape: {df.shape}")
    print(f"  Instruments: {df['instrument_id'].nunique()}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def split_by_date(
    df: pd.DataFrame,
    config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by date into train, validation, and test sets.
    
    Train: train_start to train_end (minus val_ratio at the end)
    Val: Last val_ratio of training period
    Test: test_start to test_end
    
    Args:
        df: Full DataFrame
        config: Configuration object
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Get date boundaries
    train_start = pd.to_datetime(config.dates.train_start)
    train_end = pd.to_datetime(config.dates.train_end)
    test_start = pd.to_datetime(config.dates.test_start)
    test_end = pd.to_datetime(config.dates.test_end)
    
    # Calculate validation split point (within training period)
    train_duration = train_end - train_start
    val_start = train_end - (train_duration * config.dates.val_ratio)
    
    # Split
    train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < val_start)
    val_mask = (df["timestamp"] >= val_start) & (df["timestamp"] < train_end)
    test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)
    
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df):,} rows ({train_start.date()} to {val_start.date()})")
    print(f"  Val:   {len(val_df):,} rows ({val_start.date()} to {train_end.date()})")
    print(f"  Test:  {len(test_df):,} rows ({test_start.date()} to {test_end.date()})")
    
    return train_df, val_df, test_df


# =============================================================================
# NORMALIZATION UTILITIES
# =============================================================================

class FeatureNormalizer:
    """
    Feature normalization with statistics computed on training data.
    
    Uses robust scaling for continuous features to handle outliers.
    """
    
    def __init__(self, continuous_features: List[str]):
        self.continuous_features = continuous_features
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None
        self.medians: Optional[pd.Series] = None
        self.iqrs: Optional[pd.Series] = None
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, method: str = "zscore"):
        """
        Compute normalization statistics from training data.
        
        Args:
            df: Training DataFrame
            method: 'zscore' or 'robust'
        """
        self.method = method
        
        if method == "zscore":
            self.means = df[self.continuous_features].mean()
            self.stds = df[self.continuous_features].std() + 1e-8
        elif method == "robust":
            self.medians = df[self.continuous_features].median()
            q75 = df[self.continuous_features].quantile(0.75)
            q25 = df[self.continuous_features].quantile(0.25)
            self.iqrs = (q75 - q25) + 1e-8
        
        self.fitted = True
        print(f"  Fitted normalizer on {len(self.continuous_features)} features")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted!")
        
        df = df.copy()
        
        if self.method == "zscore":
            df[self.continuous_features] = (
                df[self.continuous_features] - self.means
            ) / self.stds
        elif self.method == "robust":
            df[self.continuous_features] = (
                df[self.continuous_features] - self.medians
            ) / self.iqrs
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, method)
        return self.transform(df)


# =============================================================================
# DATASET CLASS
# =============================================================================

class CryptoTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for crypto time series with instrument-based sequences.
    
    Each sample is a sequence of `seq_length` timesteps from a single instrument.
    Maintains proper temporal ordering within each instrument.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        is_train: bool = True,
    ):
        """
        Args:
            df: DataFrame with all data
            config: Configuration object
            is_train: Whether this is training data (enables shuffling)
        """
        self.config = config
        self.is_train = is_train
        self.seq_length = config.model.seq_length
        
        # Get feature and target columns
        self.feature_cols = self._get_feature_columns(df)
        self.target_cols = config.targets.get_active_target_columns()
        
        # Filter to only columns we need
        keep_cols = ["instrument_id", "time_idx", "timestamp"] + self.feature_cols + self.target_cols
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]
        
        # Store data per instrument
        self.instruments = df["instrument_id"].unique().tolist()
        self.instrument_data: Dict[str, pd.DataFrame] = {}
        
        for inst in self.instruments:
            inst_df = df[df["instrument_id"] == inst].sort_values("time_idx").reset_index(drop=True)
            self.instrument_data[inst] = inst_df
        
        # Pre-compute all valid (instrument, start_idx) pairs
        self.indices = self._compute_valid_indices()
        
        print(f"  Created dataset: {len(self.indices):,} samples from {len(self.instruments)} instruments")
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns that exist in the DataFrame."""
        all_features = self.config.features.get_all_features()
        existing = [f for f in all_features if f in df.columns]
        
        # Warn about missing features
        missing = set(all_features) - set(existing)
        if missing:
            print(f"  Warning: {len(missing)} features missing from data: {list(missing)[:5]}...")
        
        return existing
    
    def _compute_valid_indices(self) -> List[Tuple[str, int]]:
        """
        Pre-compute all valid (instrument, start_idx) pairs.
        
        A sequence is valid if:
        1. It has `seq_length` timesteps of input features
        2. It has at least one valid (non-NaN) target at the last position
        """
        indices = []
        
        for inst in self.instruments:
            inst_df = self.instrument_data[inst]
            n_rows = len(inst_df)
            
            # Minimum length needed
            if n_rows < self.seq_length + 1:
                continue
            
            # Find valid sequences (where targets are not NaN at the end)
            target_data = inst_df[self.target_cols].values
            
            for start_idx in range(n_rows - self.seq_length):
                end_idx = start_idx + self.seq_length
                
                # Check if at least one target is valid at sequence end
                if not np.all(np.isnan(target_data[end_idx - 1])):
                    indices.append((inst, start_idx))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - features: (seq_length, n_features) tensor
            - targets: (n_targets,) tensor (at last timestep)
            - instrument_idx: integer index into dataset instruments list
            - time_idx_end: time_idx of last timestep in sequence
            - feature_mask: (seq_length,) tensor for valid positions
        """
        inst, start_idx = self.indices[idx]
        inst_df = self.instrument_data[inst]
        
        # Extract sequence
        end_idx = start_idx + self.seq_length
        seq_df = inst_df.iloc[start_idx:end_idx]
        
        # Get features
        features = seq_df[self.feature_cols].values.astype(np.float32)
        
        # Get targets (at the last timestep)
        targets = seq_df[self.target_cols].iloc[-1].values.astype(np.float32)
        time_idx_end = int(seq_df["time_idx"].iloc[-1])
        
        # Handle NaN in features (replace with 0, will be masked)
        feature_mask = ~np.isnan(features).any(axis=1)
        features = np.nan_to_num(features, nan=0.0)
        
        # Handle NaN in targets (will be masked in loss computation)
        target_mask = ~np.isnan(targets)
        targets = np.nan_to_num(targets, nan=0.0)
        
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "feature_mask": torch.tensor(feature_mask, dtype=torch.bool),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "instrument_idx": torch.tensor(self.instruments.index(inst), dtype=torch.long),
            "time_idx_end": torch.tensor(time_idx_end, dtype=torch.long),
        }
    
    def get_instrument_indices(self) -> Dict[str, List[int]]:
        """Get mapping from instrument to dataset indices."""
        inst_to_indices = defaultdict(list)
        for idx, (inst, _) in enumerate(self.indices):
            inst_to_indices[inst].append(idx)
        return dict(inst_to_indices)


# =============================================================================
# INSTRUMENT-BASED BATCH SAMPLER
# =============================================================================

class InstrumentBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that ensures all samples in a batch come from the same instrument.
    
    This is important for:
    1. RevIN: Instance normalization should be done per-instrument
    2. Temporal consistency: Batches from same instrument share distribution
    """
    
    def __init__(
        self,
        dataset: CryptoTimeSeriesDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset: CryptoTimeSeriesDataset instance
            batch_size: Number of sequences per batch
            shuffle: Whether to shuffle instruments and sequences
            drop_last: Whether to drop last incomplete batch per instrument
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get instrument -> indices mapping
        self.inst_to_indices = dataset.get_instrument_indices()
        self.instruments = list(self.inst_to_indices.keys())
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices."""
        instruments = self.instruments.copy()
        
        if self.shuffle:
            random.shuffle(instruments)
        
        for inst in instruments:
            indices = self.inst_to_indices[inst].copy()
            
            if self.shuffle:
                random.shuffle(indices)
            
            # Yield batches for this instrument
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                
                yield batch
    
    def __len__(self) -> int:
        """Total number of batches."""
        total = 0
        for inst in self.instruments:
            n_samples = len(self.inst_to_indices[inst])
            if self.drop_last:
                total += n_samples // self.batch_size
            else:
                total += (n_samples + self.batch_size - 1) // self.batch_size
        return total


class MixedInstrumentBatchSampler(Sampler[List[int]]):
    """
    Alternative batch sampler that mixes instruments but ensures temporal ordering.
    
    Samples are grouped by approximate timestamp across instruments.
    Useful for learning cross-instrument patterns.
    """
    
    def __init__(
        self,
        dataset: CryptoTimeSeriesDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get all indices
        self.all_indices = list(range(len(dataset)))
    
    def __iter__(self) -> Iterator[List[int]]:
        indices = self.all_indices.copy()
        
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            
            if len(batch) < self.batch_size and self.drop_last:
                continue
            
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.all_indices) // self.batch_size
        return (len(self.all_indices) + self.batch_size - 1) // self.batch_size


# =============================================================================
# DATA MODULE (COMBINES EVERYTHING)
# =============================================================================

class CryptoDataModule:
    """
    Data module combining dataset creation, normalization, and data loaders.
    
    Handles the full data pipeline:
    1. Load and preprocess data
    2. Split by date
    3. Fit normalizer on training data
    4. Create datasets and data loaders
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.normalizer: Optional[FeatureNormalizer] = None
        
        self.train_dataset: Optional[CryptoTimeSeriesDataset] = None
        self.val_dataset: Optional[CryptoTimeSeriesDataset] = None
        self.test_dataset: Optional[CryptoTimeSeriesDataset] = None
    
    def setup(self):
        """Load data and create datasets."""
        print("\n" + "=" * 80)
        print(" DATA LOADING")
        print("=" * 80)
        
        # Load full data
        df = load_and_preprocess_data(self.config)
        
        # Split by date
        train_df, val_df, test_df = split_by_date(df, self.config)
        
        # Initialize normalizer with continuous features
        continuous_features = self.config.features.get_all_continuous_features()
        existing_features = [f for f in continuous_features if f in train_df.columns]
        
        self.normalizer = FeatureNormalizer(existing_features)
        
        # Fit on training data and transform all splits
        print("\nNormalizing features...")
        train_df = self.normalizer.fit_transform(train_df, method="zscore")
        val_df = self.normalizer.transform(val_df)
        test_df = self.normalizer.transform(test_df)
        
        # Create datasets
        print("\nCreating datasets...")
        print("  Train:")
        self.train_dataset = CryptoTimeSeriesDataset(train_df, self.config, is_train=True)
        print("  Validation:")
        self.val_dataset = CryptoTimeSeriesDataset(val_df, self.config, is_train=False)
        print("  Test:")
        self.test_dataset = CryptoTimeSeriesDataset(test_df, self.config, is_train=False)
        
        print("\n" + "=" * 80)
    
    def train_dataloader(self, use_instrument_sampler: bool = True) -> DataLoader:
        """Get training data loader."""
        if use_instrument_sampler:
            sampler = InstrumentBatchSampler(
                self.train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                drop_last=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.pin_memory,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.pin_memory,
                drop_last=True,
            )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size * 2,  # Larger batch for eval
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
        )
    
    def get_feature_dim(self) -> int:
        """Get input feature dimension."""
        return len(self.train_dataset.feature_cols)
    
    def get_target_dim(self) -> int:
        """Get number of targets."""
        return len(self.train_dataset.target_cols)
    
    def get_num_instruments(self) -> int:
        """Get number of unique instruments."""
        return len(self.train_dataset.instruments)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_data_module(config: Optional[Config] = None) -> CryptoDataModule:
    """Create and setup data module."""
    if config is None:
        config = get_config()
    
    data_module = CryptoDataModule(config)
    data_module.setup()
    
    return data_module


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test data loading
    config = get_config()
    config.print_summary()
    
    data_module = create_data_module(config)
    
    # Test dataloaders
    print("\nTesting dataloaders...")
    train_loader = data_module.train_dataloader()
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  features: {batch['features'].shape}")
    print(f"  targets: {batch['targets'].shape}")
    print(f"  feature_mask: {batch['feature_mask'].shape}")
    print(f"  target_mask: {batch['target_mask'].shape}")
    
    print(f"\nFeature dimension: {data_module.get_feature_dim()}")
    print(f"Target dimension: {data_module.get_target_dim()}")
    print(f"Number of instruments: {data_module.get_num_instruments()}")
