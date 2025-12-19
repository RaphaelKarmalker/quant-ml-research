import os
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from tqdm import trange
from typing import Dict

from config import Config


class QlibDataPreprocessor:
    """
    A class to handle the loading, processing, and splitting of Qlib financial data.
    """

    def __init__(self):
        """Initializes the preprocessor with configuration and data fields."""
        self.config = Config()
        self.data_fields = self.config.feature_list.copy()
        self.data = {}  # A dictionary to store processed data for each symbol.

    def initialize_qlib(self):
        """Initializes the Qlib environment."""
        if self.config.use_raw_csv:
            print("Qlib init skipped (using raw CSV mode).")
            return
        print("Initializing Qlib...")
        qlib.init(provider_uri=self.config.qlib_data_path, region=REG_CN)

    def _load_from_csv_dir(self):
        """
        Load a single multi-symbol CSV (must contain 'datetime' and 'symbol').
        """
        csv_file = getattr(self.config, "raw_csv_path", None)
        if not csv_file:
            raise ValueError("raw_csv_path must be set for multi-symbol CSV mode.")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [csv_file,
                      os.path.abspath(csv_file),
                      os.path.join(script_dir, csv_file)] if not os.path.isabs(csv_file) else [csv_file]
        resolved = next((c for c in candidates if os.path.isfile(c)), None)
        if resolved is None:
            raise FileNotFoundError(f"raw_csv_path not found. Tried: {candidates}")

        print(f"Loading multi-symbol CSV file: {resolved}")

        def _post_process(df_sym: pd.DataFrame) -> pd.DataFrame:
            # FLEX MODE: keep only available configured features (no alias mapping).
            requested = list(self.config.feature_list)
            available = [c for c in requested if c in df_sym.columns]
            missing = [c for c in requested if c not in df_sym.columns]
            if not available:
                print(f"[WARN] Skipping symbol (no configured features present). Missing all: {requested}")
                return None
            if missing:
                print(f"[INFO] Symbol missing features (will ignore): {missing}")
            df_sym = df_sym[available].dropna()
            if len(df_sym) < self.config.lookback_window + self.config.predict_window + 1:
                return None
            return df_sym

        df = pd.read_csv(resolved)
        if self.config.csv_datetime_col not in df.columns:
            raise ValueError(f"Missing datetime column '{self.config.csv_datetime_col}' in CSV.")
        if self.config.csv_symbol_col not in df.columns:
            raise ValueError(f"Missing symbol column '{self.config.csv_symbol_col}' in CSV (multi-symbol mode requires it).")

        df[self.config.csv_datetime_col] = pd.to_datetime(df[self.config.csv_datetime_col])
        df = df.sort_values(self.config.csv_datetime_col)

        data = {}
        for symbol, df_sym in df.groupby(self.config.csv_symbol_col):
            df_sym = df_sym.set_index(self.config.csv_datetime_col)
            df_sym = _post_process(df_sym)
            if df_sym is not None:
                data[symbol] = df_sym

        if not data:
            raise RuntimeError("No valid symbols loaded from CSV. Check features and data length.")
        self.data = data
        print(f"Loaded {len(self.data)} symbols from multi-symbol CSV.")
        return

    def load_qlib_data(self):
        """
        Loads raw data from Qlib or CSV, processes it, and stores in self.data.
        """
        if self.config.use_raw_csv:
            self._load_from_csv_dir()
            return

        print("Loading and processing data from Qlib...")
        data_fields_qlib = ['$' + f for f in self.data_fields]
        cal: np.ndarray = D.calendar()

        # Determine the actual start and end times to load, including buffer for lookback and predict windows.
        start_index = cal.searchsorted(pd.Timestamp(self.config.dataset_begin_time))
        end_index = cal.searchsorted(pd.Timestamp(self.config.dataset_end_time))

        # Check if start_index lookbackw_window will cause negative index
        adjusted_start_index = max(start_index - self.config.lookback_window, 0)
        real_start_time = cal[adjusted_start_index]

        # Check if end_index exceeds the range of the array
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self.config.dataset_end_time):
            end_index -= 1

        # Check if end_index+predictw_window will exceed the range of the array
        adjusted_end_index = min(end_index + self.config.predict_window, len(cal) - 1)
        real_end_time = cal[adjusted_end_index]

        # Load data using Qlib's data loader.
        data_df = QlibDataLoader(config=data_fields_qlib).load(
            self.config.instrument, real_start_time, real_end_time
        )
        data_df = data_df.stack().unstack(level=1)  # Reshape for easier access.

        symbol_list = list(data_df.columns)
        for i in trange(len(symbol_list), desc="Processing Symbols"):
            symbol = symbol_list[i]
            symbol_df = data_df[symbol]

            # Pivot the table to have features as columns and datetime as index.
            symbol_df = symbol_df.reset_index().rename(columns={'level_1': 'field'})
            symbol_df = pd.pivot(symbol_df, index='datetime', columns='field', values=symbol)
            symbol_df = symbol_df.rename(columns={f'${field}': field for field in self.data_fields})

            # Calculate amount and select final features.
            symbol_df['vol'] = symbol_df['volume']
            symbol_df['amt'] = (symbol_df['open'] + symbol_df['high'] + symbol_df['low'] + symbol_df['close']) / 4 * symbol_df['vol']
            symbol_df = symbol_df[self.config.feature_list]

            # Filter out symbols with insufficient data.
            symbol_df = symbol_df.dropna()
            if len(symbol_df) < self.config.lookback_window + self.config.predict_window + 1:
                continue

            self.data[symbol] = symbol_df

    def prepare_dataset(self):
        """
        Splits the loaded data into train, validation, and test sets and saves them to disk.
        """
        print("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = {}, {}, {}

        symbol_list = list(self.data.keys())
        for i in trange(len(symbol_list), desc="Preparing Datasets"):
            symbol = symbol_list[i]
            symbol_df = self.data[symbol]

            # Define time ranges from config.
            train_start, train_end = self.config.train_time_range
            val_start, val_end = self.config.val_time_range
            test_start, test_end = self.config.test_time_range

            # Create boolean masks for each dataset split.
            train_mask = (symbol_df.index >= train_start) & (symbol_df.index <= train_end)
            val_mask = (symbol_df.index >= val_start) & (symbol_df.index <= val_end)
            test_mask = (symbol_df.index >= test_start) & (symbol_df.index <= test_end)

            # Apply masks to create the final datasets.
            train_data[symbol] = symbol_df[train_mask]
            val_data[symbol] = symbol_df[val_mask]
            test_data[symbol] = symbol_df[test_mask]

        # Save the datasets using pickle.
        os.makedirs(self.config.dataset_path, exist_ok=True)
        with open(f"{self.config.dataset_path}/train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{self.config.dataset_path}/val_data.pkl", 'wb') as f:
            pickle.dump(val_data, f)
        with open(f"{self.config.dataset_path}/test_data.pkl", 'wb') as f:
            pickle.dump(test_data, f)

        print("Datasets prepared and saved successfully.")


def time_split_df(df: pd.DataFrame, split_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert isinstance(df.index, pd.DatetimeIndex), "DataFrame index must be DatetimeIndex."
    n = len(df)
    cut = max(1, min(n - 1, int(n * split_ratio)))
    train_df = df.iloc[:cut].copy()
    val_df = df.iloc[cut:].copy()
    return train_df, val_df

def build_train_val_from_full():
    cfg = Config()
    ds_path = cfg.dataset_path
    # full pickle containing all symbols and full history
    full_name = getattr(cfg, "processed_full_pickle_name", "full_data.pkl")
    full_path = os.path.join(ds_path, full_name)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Full dataset pickle not found: {full_path}")

    with open(full_path, "rb") as f:
        data: Dict[str, pd.DataFrame] = pickle.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("full_data.pkl must be a dict[symbol -> DataFrame].")

    split_ratio = float(getattr(cfg, "train_val_split_ratio", 0.8))
    keep_features = list(cfg.feature_list)
    # keep ts_since_listing if requested in time features and exists
    time_feats = list(getattr(cfg, "time_feature_list", []))
    if "ts_since_listing" in time_feats and "ts_since_listing" not in keep_features:
        keep_features.append("ts_since_listing")

    train_dict, val_dict = {}, {}
    for sym, df in data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.copy()
        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"[WARN] {sym}: could not coerce index to datetime: {e}")
                continue
        missing = [c for c in cfg.feature_list if c not in df.columns]
        if missing:
            print(f"[WARN] {sym}: missing features {missing}, skipping.")
            continue
        # select columns (features + optional ts_since_listing)
        cols = [c for c in keep_features if c in df.columns]
        df = df[cols].dropna()
        if len(df) < (cfg.lookback_window + cfg.predict_window + 1):
            print(f"[WARN] {sym}: not enough rows after dropna, skipping.")
            continue

        tr, va = time_split_df(df, split_ratio)
        if len(tr) >= (cfg.lookback_window + cfg.predict_window + 1):
            train_dict[sym] = tr
        if len(va) >= (cfg.lookback_window + cfg.predict_window + 1):
            val_dict[sym] = va

    if not train_dict:
        raise RuntimeError("No symbols for training split.")
    if not val_dict:
        raise RuntimeError("No symbols for validation split.")

    train_path = os.path.join(ds_path, "train_data.pkl")
    val_path = os.path.join(ds_path, "val_data.pkl")
    with open(train_path, "wb") as f:
        pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(val_path, "wb") as f:
        pickle.dump(val_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[OK] Wrote {len(train_dict)} symbols to {train_path}")
    print(f"[OK] Wrote {len(val_dict)} symbols to {val_path}")


if __name__ == '__main__':
    # This block allows the script to be run directly to perform data preprocessing.
    preprocessor = QlibDataPreprocessor()
    preprocessor.initialize_qlib()
    preprocessor.load_qlib_data()
    preprocessor.prepare_dataset()
    build_train_val_from_full()

