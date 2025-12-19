import os
import sys
from typing import List, Tuple, Dict
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from time import gmtime, strftime
try:
    import wandb
except Exception:
    wandb = None

# Ensure project root is importable
sys.path.append("../")

from model.kronos import (
    Kronos,
    KronosTokenizer,
    calc_time_stamps,
    sample_from_logits,  # reuse sampling
)
from config import Config
from utils.analyser import SlidingWindowAnalyzer
import torch
import torch.nn as nn
import json


def plot_prediction(kline_df, pred_df, symbol: str = None, show: bool = True, save_path: str = None):
    # ...existing code structure; handle missing non-target columns gracefully...
    # Align index
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    # Close series
    sr_close = kline_df['close']
    sr_close.name = 'Ground Truth'
    if 'close' not in pred_df.columns:
        raise KeyError("pred_df must contain 'close' column for target inference plotting.")
    sr_pred_close = pred_df['close']
    sr_pred_close.name = "Prediction"

    # Optional volume plot only if present
    has_volume = 'volume' in kline_df.columns and 'volume' in pred_df.columns

    if has_volume:
        sr_volume = kline_df['volume']; sr_volume.name = 'Ground Truth'
        sr_pred_volume = pred_df['volume']; sr_pred_volume.name = 'Prediction'
        close_df = pd.concat([sr_close, sr_pred_close], axis=1)
        volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
        ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
        ax1.set_ylabel('Close Price', fontsize=14)
        ax1.legend(loc='lower left', fontsize=12)
        ax1.grid(True)

        ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
        ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
        ax2.set_ylabel('Volume', fontsize=14)
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True)
    else:
        close_df = pd.concat([sr_close, sr_pred_close], axis=1)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
        ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
        ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
        ax1.set_ylabel('Close Price', fontsize=14)
        ax1.legend(loc='lower left', fontsize=12)
        ax1.grid(True)

    plt.tight_layout()
    if symbol:
        fig.suptitle(f"Symbol: {symbol}", fontsize=14)
        plt.subplots_adjust(top=0.90)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    elif show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


class CloseHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, d_model] -> [B, T]
        return self.proj(h).squeeze(-1)


def auto_regressive_close_inference(
    tokenizer: KronosTokenizer,
    base_model: Kronos,
    close_head: CloseHead,
    x: torch.Tensor,
    x_stamp: torch.Tensor,
    y_stamp: torch.Tensor,
    max_context: int,
    pred_len: int,
    clip: float = 5.0,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.99,
    sample_count: int = 5,
    verbose: bool = False
):
    """
    Generate s1/s2 tokens autoregressively (as usual) and compute next-step close prediction
    at each step using the CloseHead on top of the normalized transformer context.
    Returns: preds_close [B, pred_len] on CPU numpy.
    """
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)

        device = x.device
        # expand for sampling ensemble
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        # initial tokens from context
        x_token = tokenizer.encode(x, half=True)  # [s1_ids, s2_ids]

        def get_dynamic_stamp(x_stamp_t, y_stamp_t, current_seq_len, pred_step):
            if current_seq_len <= max_context - pred_step:
                return torch.cat([x_stamp_t, y_stamp_t[:, :pred_step, :]], dim=1)
            else:
                start_idx = max_context - pred_step
                return torch.cat([x_stamp_t[:, -start_idx:, :], y_stamp_t[:, :pred_step, :]], dim=1)

        rng = range if not verbose else __import__("tqdm").trange
        step_preds = []

        for i in rng(pred_len):
            current_seq_len = initial_seq_len + i
            # crop tokens to max_context for model input
            if current_seq_len <= max_context:
                input_tokens = x_token
            else:
                input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

            current_stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i)

            # decode_s1 returns s1_logits and normalized context (post-norm)
            s1_logits, context = base_model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_last = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_last, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = base_model.decode_s2(context, sample_pre)
            s2_last = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_last, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            # append sampled tokens
            x_token[0] = torch.cat([x_token[0], sample_pre], dim=1)
            x_token[1] = torch.cat([x_token[1], sample_post], dim=1)

            # compute close prediction for "next" step: pass current context (length L) into head
            # preds align with input tokens (length L), so last index corresponds to next-step target
            preds_all = close_head(context)  # [B_ext, L]
            step_pred = preds_all[:, -1]     # [B_ext]
            step_preds.append(step_pred)

            torch.cuda.empty_cache()

        # shape to [B, pred_len] and average samples
        preds_close = torch.stack(step_preds, dim=1)          # [B_ext, pred_len]
        preds_close = preds_close.reshape(batch_size, sample_count, pred_len)
        preds_close = preds_close.mean(dim=1)                 # [B, pred_len]
        return preds_close.cpu().numpy()


def _safe_get_1d_series(df: pd.DataFrame, col_name: str):
    col = df[col_name] if col_name in df.columns else None
    if col is None:
        return None
    if isinstance(col, pd.DataFrame):
        for j in range(col.shape[1]):
            ser = col.iloc[:, j]
            try:
                if not ser.isna().all():
                    return ser
            except Exception:
                return col.iloc[:, 0]
        return col.iloc[:, 0]
    return col

def _extract_feature_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    mats = []
    for c in cols:
        ser = _safe_get_1d_series(df, c)
        if ser is None:
            raise KeyError(f"Required feature '{c}' not found.")
        vals = pd.to_numeric(ser, errors="coerce").to_numpy().astype(np.float32)
        if np.isnan(vals).any():
            raise ValueError(f"NaNs in feature '{c}'.")
        mats.append(vals)
    return np.stack(mats, axis=1)


class TargetPredictor:
    """
    Predictor for a single target (e.g., 'close') using the finetuned CloseHead.
    Generates s1/s2 tokens with the base Kronos model and predicts next-step target via CloseHead.
    """

    def __init__(self, base_model: Kronos, tokenizer: KronosTokenizer, close_head: CloseHead,
                 device="cuda:0", max_context=512, clip=5.0,
                 feature_cols=None, time_cols=None, target_col="close"):
        self.base_model = base_model.to(device)
        self.tokenizer = tokenizer.to(device)
        self.close_head = close_head.to(device)
        self.device = device
        self.max_context = max_context
        self.clip = clip
        self.feature_cols = feature_cols
        self.time_cols = list(time_cols) if time_cols is not None else ['minute', 'hour', 'weekday', 'day', 'month']
        self.target_col = target_col

        # Align input feature set with tokenizer input dim; drop time-only features like ts_since_listing
        in_dim = None
        try:
            in_dim = int(tokenizer.embed.in_features)
        except Exception:
            pass

        col_set_time = set(self.time_cols or [])
        base_cols = []
        seen = set()
        for c in (self.feature_cols or []):
            if c in col_set_time:
                continue
            if c in seen:
                continue
            seen.add(c)
            base_cols.append(c)

        # Ensure target_col is kept
        if self.target_col not in base_cols:
            raise ValueError(f"Target '{self.target_col}' must be in feature columns.")

        if in_dim is not None:
            if len(base_cols) > in_dim:
                # Keep target first, then common price/volume, then others until in_dim
                prefer_keep = [self.target_col, 'open', 'high', 'low', 'volume']
                ordered = []
                for c in prefer_keep:
                    if c in base_cols and c not in ordered:
                        ordered.append(c)
                for c in base_cols:
                    if c not in ordered:
                        ordered.append(c)
                base_cols = ordered[:in_dim]
                if self.target_col not in base_cols:
                    # force-include target by replacing last
                    base_cols[-1] = self.target_col
                print(f"[INFO] TargetPredictor features aligned to {in_dim}: {base_cols}")
            elif len(base_cols) < in_dim:
                raise ValueError(f"Configured features ({len(base_cols)}) < tokenizer input dim ({in_dim}).")
        self.input_feature_cols = base_cols

    def generate_close(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds_close = auto_regressive_close_inference(
            tokenizer=self.tokenizer,
            base_model=self.base_model,
            close_head=self.close_head,
            x=x_tensor,
            x_stamp=x_stamp_tensor,
            y_stamp=y_stamp_tensor,
            max_context=self.max_context,
            pred_len=pred_len,
            clip=self.clip,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        preds_close = preds_close[:, -pred_len:]
        return preds_close

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all(col in df.columns for col in self.feature_cols):
            raise ValueError(f"Required feature columns {self.feature_cols} not found in DataFrame.")

        df = df.copy()
        if df[self.feature_cols].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in required feature columns.")

        # Build stamps; include ts_since_listing from df for x, extend for y
        last_ts_since = None
        if self.time_cols and ('ts_since_listing' in self.time_cols) and ('ts_since_listing' in df.columns):
            last_ts_since = int(df['ts_since_listing'].iloc[-1])

        x_time_df = calc_time_stamps(x_timestamp, extra_df=df, time_feature_list=self.time_cols)
        y_time_df = calc_time_stamps(y_timestamp, time_feature_list=self.time_cols, base_ts_since_listing=last_ts_since)

        if self.time_cols:
            missing_x = [c for c in self.time_cols if c not in x_time_df.columns]
            missing_y = [c for c in self.time_cols if c not in y_time_df.columns]
            if missing_x or missing_y:
                raise ValueError(f"Missing time columns in stamps. x missing={missing_x}, y missing={missing_y}")
            x_stamp = x_time_df[self.time_cols].values.astype(np.float32)
            y_stamp = y_time_df[self.time_cols].values.astype(np.float32)
        else:
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

        x = _extract_feature_matrix(df, self.input_feature_cols).astype(np.float32)
        # stats per feature for normalization + target unscale
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -self.clip, self.clip)

        preds_close_norm = self.generate_close(x_norm[np.newaxis, :], x_stamp[np.newaxis, :], y_stamp[np.newaxis, :],
                                               pred_len, T, top_k, top_p, sample_count, verbose)
        # denormalize with target stats
        if self.target_col not in self.feature_cols:
            raise ValueError(f"Target column '{self.target_col}' not found in feature_cols.")
        t_idx = self.feature_cols.index(self.target_col)
        preds_close = preds_close_norm.squeeze(0) * (x_std[t_idx] + 1e-5) + x_mean[t_idx]

        pred_df = pd.DataFrame({self.target_col: preds_close}, index=y_timestamp)
        return pred_df

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)
        x_list, x_stamp_list, y_stamp_list = [], [], []
        means, stds = [], []
        seq_lens, y_lens = [], []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.feature_cols):
                raise ValueError(f"DataFrame at index {i} is missing required feature columns {self.feature_cols}.")
            df = df.copy()
            if df[self.feature_cols].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in required feature columns.")

            # Prepare stamps with ts_since_listing injection/extension
            last_ts_since = None
            if self.time_cols and ('ts_since_listing' in self.time_cols) and ('ts_since_listing' in df.columns):
                last_ts_since = int(df['ts_since_listing'].iloc[-1])

            x_timestamp = x_timestamp_list[i]; y_timestamp = y_timestamp_list[i]
            x_time_df = calc_time_stamps(x_timestamp, extra_df=df, time_feature_list=self.time_cols)
            y_time_df = calc_time_stamps(y_timestamp, time_feature_list=self.time_cols, base_ts_since_listing=last_ts_since)

            x = _extract_feature_matrix(df, self.input_feature_cols).astype(np.float32)
            if self.time_cols:
                missing_x = [c for c in self.time_cols if c not in x_time_df.columns]
                missing_y = [c for c in self.time_cols if c not in y_time_df.columns]
                if missing_x or missing_y:
                    raise ValueError(f"Missing time columns in stamps at series {i}. x missing={missing_x}, y missing={missing_y}")
                x_stamp = x_time_df[self.time_cols].values.astype(np.float32)
                y_stamp = y_time_df[self.time_cols].values.astype(np.float32)
            else:
                x_stamp = x_time_df.values.astype(np.float32)
                y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm); x_stamp_list.append(x_stamp); y_stamp_list.append(y_stamp)
            means.append(x_mean); stds.append(x_std)
            seq_lens.append(x_norm.shape[0]); y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

        preds_close_norm = self.generate_close(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)
        pred_dfs = []
        if self.target_col not in self.feature_cols:
            raise ValueError(f"Target column '{self.target_col}' not found in feature_cols.")
        t_idx = self.feature_cols.index(self.target_col)
        for i in range(num_series):
            preds_i = preds_close_norm[i] * (stds[i][t_idx] + 1e-5) + means[i][t_idx]
            pred_df = pd.DataFrame({self.target_col: preds_i}, index=y_timestamp_list[i])
            pred_dfs.append(pred_df)
        return pred_dfs


class CSVWindowInferenceTarget:
    """
    Single-window inference runner using TargetPredictor (CloseHead).
    Loads base Kronos from target_finetuning checkpoint folder and its CloseHead.
    """

    def __init__(
        self,
        csv_path: str,
        datetime_col: str,
        feature_cols: List[str],
        time_cols: List[str],
        device: str,
        tokenizer_id: str,
        model_id: str,
        max_context: int,
        clip: float,
        use_test_pickle: bool,
        processed_test_path: str,
        test_symbol: str | None,
        target_col: str = "close",
    ):
        self.csv_path = csv_path
        self.datetime_col = datetime_col
        self.feature_cols = list(feature_cols)
        self.time_cols = list(time_cols) if time_cols is not None else None
        self.device = device
        self.tokenizer_id = tokenizer_id
        self.model_id = model_id
        self.max_context = max_context
        self.clip = clip
        self.use_test_pickle = use_test_pickle
        self.processed_test_path = processed_test_path
        self.test_symbol = test_symbol
        self.target_col = target_col

        self._tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        self._model = Kronos.from_pretrained(self.model_id)
        self._predictor = TargetPredictor(
            base_model=self._model,
            tokenizer=self._tokenizer,
            close_head=CloseHead(self._model.d_model),
            device=self.device,
            max_context=self.max_context,
            clip=self.clip,
            feature_cols=self.feature_cols,
            time_cols=self.time_cols,
            target_col=self.target_col,
        )

        self._data: Dict[str, pd.DataFrame] = self._load_source()

    def _load_source(self) -> Dict[str, pd.DataFrame]:
        if self.use_test_pickle:
            if not os.path.isfile(self.processed_test_path):
                raise FileNotFoundError(f"Processed test pickle not found: {self.processed_test_path}")
            with open(self.processed_test_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or not data:
                raise ValueError("Processed test pickle must be dict[symbol -> DataFrame].")
            out = {}
            for sym, df in data.items():
                if self.test_symbol and sym != self.test_symbol:
                    continue
                if not isinstance(df.index, pd.DatetimeIndex):
                    try: df.index = pd.to_datetime(df.index)
                    except Exception: continue
                # Keep only needed columns
                keep = [c for c in self.feature_cols if c in df.columns]
                # keep optional ts_since_listing if present (for time features)
                if self.time_cols and ('ts_since_listing' in self.time_cols) and ('ts_since_listing' in df.columns):
                    if 'ts_since_listing' not in keep:
                        keep.append('ts_since_listing')
                df2 = df[keep].dropna()
                if df2.shape[0] > 0:
                    out[sym] = df2
            return out
        else:
            if not os.path.isfile(self.csv_path):
                raise FileNotFoundError(f"CSV not found: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            if self.datetime_col not in df.columns:
                raise ValueError(f"Missing datetime column '{self.datetime_col}' in CSV.")
            # Require symbol column
            sym_col = None
            for c in ["instrument_id", "symbol", "ticker"]:
                if c in df.columns:
                    sym_col = c
                    break
            if sym_col is None:
                raise ValueError("CSV must contain a symbol column (e.g., 'instrument_id').")
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
            df = df.sort_values(self.datetime_col)
            out = {}
            for sym, g in df.groupby(sym_col):
                if self.test_symbol and sym != self.test_symbol:
                    continue
                g = g.set_index(self.datetime_col)
                keep = [c for c in self.feature_cols if c in g.columns]
                if self.time_cols and ('ts_since_listing' in self.time_cols) and ('ts_since_listing' in g.columns):
                    if 'ts_since_listing' not in keep: keep.append('ts_since_listing')
                gg = g[keep].dropna()
                if gg.shape[0] > 0:
                    out[sym] = gg
            return out

    def _iter_windows(self, df: pd.DataFrame, lookback: int, pred_len: int, start_date: str, end_date: str, step: int):
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        df = df.sort_index()
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        n = len(df)
        need = lookback + pred_len
        if n < need:
            return
        max_s = n - need
        for s in range(0, max_s + 1, max(1, int(step))):
            block = df.iloc[s:s + need]
            x_df = block.iloc[:lookback]
            future_df = block.iloc[lookback:]
            yield x_df, future_df

    def run_sliding(
        self,
        lookback: int,
        pred_len: int,
        start_date: str,
        end_date: str,
        T: float,
        top_k: int,
        top_p: float,
        sample_count: int,
        verbose: bool,
        batch_size: int,
        step: int,
        live_plot: bool = False,
        live_plot_dir: str = None,
        live_plot_mode: str = "save",
        periodic_analysis_n: int | None = None
    ):
        # Collect all windows across symbols
        x_df_list, x_ts_list, y_ts_list, metas = [], [], [], []
        for sym, df in self._data.items():
            for x_df, fut_df in self._iter_windows(df, lookback, pred_len, start_date, end_date, step):
                x_df_list.append(x_df)
                x_ts_list.append(x_df.index)
                y_ts_list.append(fut_df.index)
                metas.append((sym, x_df, fut_df))
        if not x_df_list:
            return []

        results = []
        bs = max(1, min(batch_size, len(x_df_list)))
        for i in range(0, len(x_df_list), bs):
            df_chunk = x_df_list[i:i+bs]
            x_ts_chunk = x_ts_list[i:i+bs]
            y_ts_chunk = y_ts_list[i:i+bs]
            preds = self._predictor.predict_batch(
                df_chunk, x_ts_chunk, y_ts_chunk,
                pred_len=pred_len, T=T, top_k=top_k, top_p=top_p, sample_count=sample_count, verbose=verbose
            )
            # Map predictions back to metas (trim if needed)
            m_chunk = metas[i:i+bs][:len(preds)]
            for (sym, x_df, fut_df), pred_df in zip(m_chunk, preds):
                results.append({
                    "symbol": sym,
                    "start": x_df.index[0],
                    "end": fut_df.index[-1],
                    "pred": pred_df,
                    "kline": pd.concat([x_df[self.feature_cols], fut_df[self.feature_cols]], axis=0)
                })
        return results


def main():
    cfg = Config()
    csv_path = cfg.raw_csv_path
    datetime_col = cfg.csv_datetime_col
    device = "cuda:0"
    lookback = cfg.lookback_window
    pred_len = cfg.predict_window
    max_context = cfg.max_context
    clip = cfg.clip
    T = cfg.inference_T
    top_k = cfg.inference_top_k
    top_p = cfg.inference_top_p
    sample_count = cfg.inference_sample_count
    SHOW_PLOT = True
    test_start, test_end = cfg.test_time_range
    plot_amount = getattr(cfg, "vis_plot_amount", 10)
    slide_batch_size = getattr(cfg, "backtest_batch_size", 64)
    slide_step = getattr(cfg, "slide_step", 1)
    analysis_target_col = getattr(cfg, "analysis_target_col", "close")
    live_plot = getattr(cfg, "live_plot_sliding", False)
    live_plot_mode = getattr(cfg, "inference_live_plot_mode", "save")
    live_plot_dir = getattr(cfg, "inference_plot_output_dir", "./inference_outputs")
    save_analysis = getattr(cfg, "inference_save_analysis_plots", False)
    periodic_analysis_n = getattr(cfg, "inference_analyze_every_n_batches", 0)

    tokenizer_id = cfg.finetuned_tokenizer_path
    # Prefer explicit target finetune path if available
    model_id = getattr(cfg, "finetuned_target_predictor_path", cfg.finetuned_predictor_path)

    wandb_run = None
    if getattr(cfg, "inference_analysis_wandb_log", False) and getattr(cfg, "use_wandb", False) and (wandb is not None):
        try:
            if getattr(wandb, "run", None) is None:
                base_name = getattr(cfg, "inference_wandb_name", "inference-target")
                ts = strftime("%Y%m%d_%H%M%S", gmtime())
                run_name = f"{base_name}-{ts}"
                wandb_run = wandb.init(
                    project=cfg.wandb_config.get("project"),
                    entity=cfg.wandb_config.get("entity"),
                    name=run_name,
                    tags=[getattr(cfg, "inference_wandb_tag", "inference-target")]
                )
                wandb.config.update(cfg.__dict__, allow_val_change=True)
                print(f"Weights & Biases (inference-target) initialized: {run_name}")
        except Exception as e:
            print(f"[WARN] Failed to init W&B for inference: {e}")

    os.makedirs(live_plot_dir, exist_ok=True)
    analysis_dir = os.path.join(live_plot_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    use_pickle = getattr(cfg, "inference_use_test_pickle", False)
    processed_test_path = os.path.join(cfg.dataset_path, getattr(cfg, "processed_test_pickle_name", "test_data.pkl"))
    runner = CSVWindowInferenceTarget(
        csv_path=csv_path,
        datetime_col=datetime_col,
        feature_cols=cfg.feature_list,
        time_cols=cfg.time_feature_list,
        device=device,
        tokenizer_id=tokenizer_id,
        model_id=model_id,
        max_context=max_context,
        clip=clip,
        use_test_pickle=use_pickle,
        processed_test_path=processed_test_path,
        test_symbol=None,
        target_col=analysis_target_col,
    )

    slide_results = runner.run_sliding(
        lookback=lookback,
        pred_len=pred_len,
        start_date=test_start,
        end_date=test_end,
        T=T,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=True,
        batch_size=slide_batch_size,
        step=slide_step,
        live_plot=live_plot,
        live_plot_dir=live_plot_dir,
        live_plot_mode=live_plot_mode,
        periodic_analysis_n=periodic_analysis_n if periodic_analysis_n > 0 else None
    )
    print(f"Generated {len(slide_results)} sliding windows.")
    if slide_results and SHOW_PLOT:
        analyzer = SlidingWindowAnalyzer(
            target_col=analysis_target_col,
            wandb_log=getattr(cfg, "inference_analysis_wandb_log", False),
            wandb_prefix=getattr(cfg, "inference_analysis_wandb_prefix", "analysis/")
        )

        show_analysis = SHOW_PLOT and (matplotlib.get_backend().lower() != "agg")

        mpae = analyzer.compute_mpae_bar_by_bar(slide_results)
        analyzer.plot_mpae_curve(
            mpae,
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "mpae_curve.png") if save_analysis else None
        )

        da = analyzer.compute_directional_accuracy_bar_by_bar(slide_results)
        analyzer.plot_da_curve(
            da,
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "directional_accuracy_curve.png") if save_analysis else None
        )

        rmse = analyzer.compute_rmse_bar_by_bar(slide_results)
        analyzer.plot_curve(
            rmse, ylabel="RMSE", title=f"RMSE per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "rmse_curve.png") if save_analysis else None
        )

        smape = analyzer.compute_smape_bar_by_bar(slide_results)
        analyzer.plot_curve(
            smape, ylabel="sMAPE", title=f"sMAPE per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "smape_curve.png") if save_analysis else None
        )

        ret_corr = analyzer.compute_return_corr_bar_by_bar(slide_results)
        analyzer.plot_curve(
            ret_corr, ylabel="Return correlation", title=f"Return Corr per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "return_corr_curve.png") if save_analysis else None
        )

        r2 = analyzer.compute_r2_bar_by_bar(slide_results)
        analyzer.plot_curve(
            r2, ylabel="R²", title=f"R² per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "r2_curve.png") if save_analysis else None
        )

        mspe = analyzer.compute_mspe_bar_by_bar(slide_results)
        analyzer.plot_curve(
            mspe, ylabel="VSE", title=f"VSE per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "vse_curve.png") if save_analysis else None
        )

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
