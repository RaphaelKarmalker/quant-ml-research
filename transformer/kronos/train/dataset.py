import pickle
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from config import Config
from model.kronos import calc_time_stamps


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

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

        # Set paths (do not size epoch here)
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

            # Check required feature columns only (time features are derived later)
            missing_feat = [c for c in self.feature_list if c not in df.columns]
            if missing_feat:
                raise KeyError(f"Missing feature columns for symbol `{symbol}`: {missing_feat}. "
                               f"Ensure config.feature_list matches your preprocess output.")

            # Keep features (+ optional ts_since_listing if requested)
            keep_cols = list(dict.fromkeys(self.feature_list))
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

        # Restore fixed epoch sizing from config instead of deriving from window counts
        if self.data_type == 'train':
            self.n_samples = int(getattr(self.config, "n_train_iter", 1))
        else:
            self.n_samples = int(getattr(self.config, "n_val_iter", 1))

        if len(self.indices) == 0:
            print(f"[{self.data_type.upper()}] No sliding windows found.")
        else:
            bs = int(getattr(self.config, "batch_size", 32))
            est_steps = (self.n_samples + bs - 1) // bs
            print(f"[{self.data_type.upper()}] windows={len(self.indices)}, batch_size={bs}, steps/epoch≈{est_steps}")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """
        Returns the number of samples per epoch.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`. This
        ensures random sampling over the entire dataset for each call.

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

        # Features
        x = win_df[self.feature_list].values.astype(np.float32)

        # Time features derived from timestamps; inject ts_since_listing if present
        x_stamp_df = calc_time_stamps(win_df.index, extra_df=win_df, time_feature_list=self.time_feature_list)
        if self.time_feature_list:
            missing = [c for c in self.time_feature_list if c not in x_stamp_df.columns]
            if missing:
                raise ValueError(f"QlibDataset: missing time columns in stamps: {missing}")
            x_stamp = x_stamp_df[self.time_feature_list].values.astype(np.float32)
        else:
            x_stamp = x_stamp_df.values.astype(np.float32)

        # Normalize per-instance
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # To tensors
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")
