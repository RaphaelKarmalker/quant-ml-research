import pickle
import random
import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model.kronos import calc_time_stamps


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib/Bitget financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Features:
    - Per-window normalization with optional robust scaling for returns
    - Support for return-based features from Bitget preprocessing
    - Random sampling from all (symbol, start_idx) pairs

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        self.py_rng = random.Random(self.config.seed)

        # Set paths
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list
        
        # Identify return-based features for robust scaling
        self.return_features = [f for f in self.feature_list if 'returns' in f.lower()]
        self.other_features = [f for f in self.feature_list if 'returns' not in f.lower()]

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol]

            # Ensure datetime index exists
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise KeyError(f"{symbol}: could not coerce index to DatetimeIndex: {e}")

            # Check required feature columns (allow missing with warning)
            available_feats = [c for c in self.feature_list if c in df.columns]
            missing_feat = [c for c in self.feature_list if c not in df.columns]
            if missing_feat:
                print(f"[WARN] {symbol}: Missing features {missing_feat}, using available {len(available_feats)}")
            
            if len(available_feats) < 5:  # Need at least some features
                print(f"[WARN] {symbol}: Too few features ({len(available_feats)}), skipping")
                continue

            # Keep only available features
            keep_cols = list(dict.fromkeys(available_feats))
            if self.time_feature_list and ('ts_since_listing' in self.time_feature_list) and ('ts_since_listing' in df.columns):
                if 'ts_since_listing' not in keep_cols:
                    keep_cols.append('ts_since_listing')
            df = df[keep_cols].copy()
            self.data[symbol] = df

            # Build sliding windows over entire series
            series_len = len(df)
            num_samples = series_len - self.window + 1
            if num_samples > 0:
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # Restore fixed epoch sizing from config
        if self.data_type == 'train':
            self.n_samples = int(getattr(self.config, "n_train_iter", 1))
        else:
            self.n_samples = int(getattr(self.config, "n_val_iter", 1))

        if len(self.indices) == 0:
            print(f"[{self.data_type.upper()}] No sliding windows found.")
        else:
            bs = int(getattr(self.config, "batch_size", 32))
            est_steps = (self.n_samples + bs - 1) // bs
            print(f"[{self.data_type.upper()}] symbols={len(self.symbols)}, windows={len(self.indices)}, "
                  f"features={len(self.feature_list)}, batch_size={bs}, steps/epoch≈{est_steps}")

    def set_epoch_seed(self, epoch: int):
        """Sets a new seed for the random sampler for each epoch."""
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def _normalize_features(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize features per-window.
        
        For return features: use robust scaling (median + IQR)
        For other features: use standard z-score (mean + std)
        
        Args:
            x: Feature array of shape (window_len, num_features)
        
        Returns:
            Normalized array with same shape
        """
        x_norm = np.zeros_like(x)
        
        for i in range(x.shape[1]):
            col = x[:, i]
            
            # Check if this is a return feature (use robust scaling)
            # Robust scaling: (x - median) / IQR
            # This is more stable for heavy-tailed return distributions
            if i < len(self.return_features):
                median = np.median(col)
                q75, q25 = np.percentile(col, [75, 25])
                iqr = q75 - q25
                if iqr < 1e-8:
                    iqr = np.std(col) + 1e-8  # Fallback to std
                x_norm[:, i] = (col - median) / (iqr + 1e-8)
            else:
                # Standard z-score for non-return features
                mean = np.mean(col)
                std = np.std(col)
                x_norm[:, i] = (col - mean) / (std + 1e-5)
        
        # Clip to prevent extreme values
        x_norm = np.clip(x_norm, -self.config.clip, self.config.clip)
        
        return x_norm

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`.

        Args:
            idx (int): Ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
                - x_stamp_tensor (torch.Tensor): The time feature tensor.
        """
        # Randomly pick a window (ignore provided idx)
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Get available features (some may be missing per symbol)
        available_feats = [f for f in self.feature_list if f in win_df.columns]
        x = win_df[available_feats].values.astype(np.float32)
        
        # Pad missing features with zeros if needed
        if len(available_feats) < len(self.feature_list):
            full_x = np.zeros((x.shape[0], len(self.feature_list)), dtype=np.float32)
            for i, f in enumerate(self.feature_list):
                if f in available_feats:
                    feat_idx = available_feats.index(f)
                    full_x[:, i] = x[:, feat_idx]
            x = full_x

        # Time features derived from timestamps
        x_stamp_df = calc_time_stamps(win_df.index, extra_df=win_df, time_feature_list=self.time_feature_list)
        if self.time_feature_list:
            available_time = [c for c in self.time_feature_list if c in x_stamp_df.columns]
            if available_time:
                x_stamp = x_stamp_df[available_time].values.astype(np.float32)
            else:
                x_stamp = x_stamp_df.values.astype(np.float32)
        else:
            x_stamp = x_stamp_df.values.astype(np.float32)

        # Handle NaN values (fill with 0 for stability)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_stamp = np.nan_to_num(x_stamp, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize per-instance with robust scaling for returns
        x = self._normalize_features(x)

        # To tensors
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")
    print(f"Return features: {train_dataset.return_features}")
    print(f"Other features: {train_dataset.other_features}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
        print(f"Feature stats - mean: {try_x.mean():.4f}, std: {try_x.std():.4f}, min: {try_x.min():.4f}, max: {try_x.max():.4f}")
    else:
        print("Dataset is empty.")
