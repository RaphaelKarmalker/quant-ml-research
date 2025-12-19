import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Optional Weights & Biases
try:
    import wandb  # type: ignore
except Exception:  # keep analyzer usable without wandb installed
    wandb = None

"""
MPAE (Mean Percentage Absolute Error)

Definition: mean_h(|ŷ_t+h − y_t+h| / (|y_t+h| + ε))
Bereich: [0, ∞), kleiner ist besser.
Vorteil: skaleninvariant; Nachteil: empfindlich bei kleinen |y|.
Implementierung: pro Horizont APE berechnen und über alle Windows mitteln.

DA (Directional Accuracy, relativ zu t0)
Definition: hit_t+h = 1, wenn sign(ŷ_t+h − y_t) = sign(y_t+h − y_t), sonst 0; dann mean_h(hit).
Bereich: [0, 1], höher ist besser; 0.5 ≈ Zufall.
Wichtig: Richtung wird gegen den letzten Kontextpunkt t0 geprüft, nicht sequenziell.

RMSE (Root Mean Squared Error)
Definition: sqrt(mean_h((ŷ_t+h − y_t+h)^2))
Bereich: [0, ∞), kleiner ist besser.
Vorteil: starke Fehler werden stärker gewichtet; skalenabhängig.

sMAPE (Symmetric MAPE)
Definition: mean_h(2·|ŷ − y| / (|ŷ| + |y| + ε))
Bereich: [0, 2], oft in Prozent interpretiert (×100).
Vorteil: symmetrischer als MAPE; robuster bei Niveauwechseln.
Return-Korrelation (relativ zu t0)

Definition: Pearson-Korrelation zwischen (y_t+h − y_t) und (ŷ_t+h − y_t) pro Horizont.
Bereich: [-1, 1], näher an 1 ist besser.
Misst, ob relative Bewegungen um t0 herum korrekt mitlaufen (Trendtiming).
"""

# Ensure project root is importable
sys.path.append("../")
class SlidingWindowAnalyzer:
    def __init__(self, target_col: str = "close", eps: float = 1e-8, wandb_log: bool = False, wandb_prefix: str = "analysis/"):
        self.target_col = target_col
        self.eps = eps
        self.wandb_log = bool(wandb_log)
        # Ensure consistent overlay keys across runs by using a stable prefix
        self.wandb_prefix = str(wandb_prefix or "")

    # --- W&B helper ---
    # --- W&B helper ---
    def _log_series(self, metric_key: str, series: pd.Series, title: Optional[str] = None):
        """
        Log a per-horizon pd.Series to W&B as a line plot only.
        metric_key: short metric name, e.g. "mpae", "directional_accuracy", "rmse", "smape", "return_correlation"
        """
        if not (self.wandb_log and wandb is not None):
            return
        # Ensure a run exists, otherwise skip logging silently
        if getattr(wandb, "run", None) is None:
            return
        if series is None or series.empty:
            return

        # Build table just for plotting
        table = wandb.Table(columns=["horizon", "value"])
        for h, v in series.items():
            try:
                h_int = int(h)
            except Exception:
                h_int = h
            table.add_data(h_int, float(v))

        # Stable key for overlaying across runs
        key_base = f"{self.wandb_prefix}{metric_key}"
        chart_title = title or f"{metric_key.replace('_', ' ').title()} per horizon ({self.target_col})"
        chart = wandb.plot.line(table, "horizon", "value", title=chart_title)

        # Log only the chart (no *_table summary)
        wandb.log({key_base: chart})


    def compute_mpae_bar_by_bar(self, slide_results: List[dict]) -> pd.Series:
        """
        Compute Mean Percentage Absolute Error (MPAE) per prediction step across all sliding windows.

        Returns:
            pd.Series of shape (pred_len,) indexed by horizon 1..pred_len with MPAE values.
        """
        if not slide_results:
            return pd.Series(dtype=float)

        apes = []
        pred_len_ref = None

        for item in slide_results:
            pred_df = item["pred"]
            kline_df = item["kline"]

            if self.target_col not in pred_df.columns or self.target_col not in kline_df.columns:
                continue

            true_future = kline_df[self.target_col].iloc[-len(pred_df):]
            pred_future = pred_df[self.target_col]

            aligned = pd.concat(
                [true_future.rename("y"), pred_future.rename("yhat")],
                axis=1
            ).dropna()

            if aligned.empty:
                continue

            if pred_len_ref is None:
                pred_len_ref = aligned.shape[0]
            else:
                if aligned.shape[0] != pred_len_ref:
                    continue

            ape = (aligned["yhat"] - aligned["y"]).abs() / (aligned["y"].abs() + self.eps)
            apes.append(ape.values)

        if not apes:
            return pd.Series(dtype=float)

        apes_np = np.vstack(apes)  # (num_windows, pred_len)
        mpae = apes_np.mean(axis=0)  # (pred_len,)

        horizons = np.arange(1, mpae.shape[0] + 1)  # 1..pred_len
        series = pd.Series(mpae, index=horizons, name=f"MPAE_{self.target_col}")
        # W&B logging
        self._log_series("mpae", series, title=f"MPAE per horizon ({self.target_col})")
        return series

    def plot_mpae_curve(self, mpae_series: pd.Series, title: Optional[str] = None, show: bool = True, save_path: str = None):
        if mpae_series is None or mpae_series.empty:
            print("No MPAE data to plot.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(mpae_series.index, mpae_series.values, marker="o")
        plt.xlabel("Prediction step (t + h)")
        plt.ylabel("Mean Percentage Absolute Error")
        plt.title(title or f"MPAE per prediction horizon ({self.target_col})")
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def compute_directional_accuracy_bar_by_bar(self, slide_results: List[dict]) -> pd.Series:
        """
        Compute directional accuracy per prediction step across all sliding windows, relative to t0 truth.
        For each horizon h:
          hit = sign(pred[t+h] - truth[t0]) == sign(truth[t+h] - truth[t0])
        Returns:
          pd.Series of shape (pred_len,) indexed by horizon 1..pred_len with mean hitrate.
        """
        if not slide_results:
            return pd.Series(dtype=float)

        hits = []
        pred_len_ref = None

        for item in slide_results:
            pred_df = item["pred"]
            kline_df = item["kline"]

            if self.target_col not in pred_df.columns or self.target_col not in kline_df.columns:
                continue

            pred_len = len(pred_df)
            if len(kline_df) < pred_len + 1:
                continue  # need at least one context bar before future

            # Ground-truth future window and baseline t0 (last context)
            true_future = kline_df[self.target_col].iloc[-pred_len:]
            base_idx = len(kline_df) - pred_len - 1
            if base_idx < 0:
                continue
            base_val = kline_df[self.target_col].iloc[base_idx]

            pred_future = pred_df[self.target_col]

            aligned = pd.concat(
                [true_future.rename("y"), pred_future.rename("yhat")],
                axis=1
            ).dropna()

            if aligned.empty:
                continue

            if pred_len_ref is None:
                pred_len_ref = aligned.shape[0]
            else:
                if aligned.shape[0] != pred_len_ref:
                    continue

            # Directions relative to t0 baseline
            pred_dir = np.sign(aligned["yhat"].values - base_val)
            true_dir = np.sign(aligned["y"].values - base_val)

            hit = (pred_dir == true_dir).astype(float)  # 1 if same side (including both 0), else 0
            hits.append(hit)

        if not hits:
            return pd.Series(dtype=float)

        hits_np = np.vstack(hits)  # (num_windows, pred_len)
        da = hits_np.mean(axis=0)  # (pred_len,)

        horizons = np.arange(1, da.shape[0] + 1)
        series = pd.Series(da, index=horizons, name=f"DA_{self.target_col}")
        # W&B logging
        self._log_series("directional_accuracy", series, title=f"Directional Accuracy per horizon ({self.target_col})")
        return series

    def plot_da_curve(self, da_series: pd.Series, title: Optional[str] = None, show: bool = True, save_path: str = None):
        if da_series is None or da_series.empty:
            print("No DA data to plot.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(da_series.index, da_series.values, marker="o")
        plt.xlabel("Prediction step (t + h)")
        plt.ylabel("Directional Accuracy (hitrate)")
        plt.title(title or f"Directional Accuracy per horizon ({self.target_col})")
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()

    def compute_rmse_bar_by_bar(self, slide_results: List[dict]) -> pd.Series:
        """
        RMSE per prediction step across all sliding windows (absolute values).
        """
        if not slide_results:
            return pd.Series(dtype=float)
        errs = []
        pred_len_ref = None
        for item in slide_results:
            pred_df = item["pred"]; kline_df = item["kline"]
            if self.target_col not in pred_df.columns or self.target_col not in kline_df.columns:
                continue
            true_future = kline_df[self.target_col].iloc[-len(pred_df):]
            pred_future = pred_df[self.target_col]
            aligned = pd.concat([true_future.rename("y"), pred_future.rename("yhat")], axis=1).dropna()
            if aligned.empty:
                continue
            if pred_len_ref is None:
                pred_len_ref = aligned.shape[0]
            elif aligned.shape[0] != pred_len_ref:
                continue
            errs.append((aligned["yhat"].values - aligned["y"].values) ** 2)
        if not errs:
            return pd.Series(dtype=float)
        rmse = np.sqrt(np.mean(np.vstack(errs), axis=0))
        horizons = np.arange(1, rmse.shape[0] + 1)
        series = pd.Series(rmse, index=horizons, name=f"RMSE_{self.target_col}")
        # W&B logging
        self._log_series("rmse", series, title=f"RMSE per horizon ({self.target_col})")
        return series

    def compute_smape_bar_by_bar(self, slide_results: List[dict]) -> pd.Series:
        """
        sMAPE per prediction step across all sliding windows.
        sMAPE = 2*|yhat - y| / (|y| + |yhat| + eps)
        """
        if not slide_results:
            return pd.Series(dtype=float)
        smapes = []
        pred_len_ref = None
        for item in slide_results:
            pred_df = item["pred"]; kline_df = item["kline"]
            if self.target_col not in pred_df.columns or self.target_col not in kline_df.columns:
                continue
            true_future = kline_df[self.target_col].iloc[-len(pred_df):]
            pred_future = pred_df[self.target_col]
            aligned = pd.concat([true_future.rename("y"), pred_future.rename("yhat")], axis=1).dropna()
            if aligned.empty:
                continue
            if pred_len_ref is None:
                pred_len_ref = aligned.shape[0]
            elif aligned.shape[0] != pred_len_ref:
                continue
            num = 2.0 * np.abs(aligned["yhat"].values - aligned["y"].values)
            den = np.abs(aligned["yhat"].values) + np.abs(aligned["y"].values) + self.eps
            smapes.append(num / den)
        if not smapes:
            return pd.Series(dtype=float)
        smape = np.mean(np.vstack(smapes), axis=0)
        horizons = np.arange(1, smape.shape[0] + 1)
        series = pd.Series(smape, index=horizons, name=f"sMAPE_{self.target_col}")
        # W&B logging
        self._log_series("smape", series, title=f"sMAPE per horizon ({self.target_col})")
        return series

    def compute_mspe_bar_by_bar(self, slide_results: List[dict]) -> pd.Series:
        """
        Volatility-Scaled Error (rolling-move) per prediction step across all sliding windows.
        VSE_h = |yhat_{t+h} - y_{t+h}| / ( (1/k_h) * sum_{i=1..k_h} |y_{t+i} - y_{t+i-1}| + eps )
        with k_h = min(h + 2, available_future_moves). The extra 2 steps are used only in the denominator if available.
        """
        if not slide_results:
            return pd.Series(dtype=float)

        ratios = []
        pred_len_ref = None

        for item in slide_results:
            pred_df = item["pred"]; kline_df = item["kline"]
            if self.target_col not in pred_df.columns or self.target_col not in kline_df.columns:
                continue

            pred_len = len(pred_df)
            if len(kline_df) < pred_len + 1:
                continue  # need at least one context point before future

            true_future = kline_df[self.target_col].iloc[-pred_len:]
            base_idx = len(kline_df) - pred_len - 1
            if base_idx < 0:
                continue
            base_val = kline_df[self.target_col].iloc[base_idx]
            pred_future = pred_df[self.target_col]

            aligned = pd.concat(
                [true_future.rename("y"), pred_future.rename("yhat")],
                axis=1
            ).dropna()
            if aligned.empty:
                continue

            if pred_len_ref is None:
                pred_len_ref = aligned.shape[0]
            elif aligned.shape[0] != pred_len_ref:
                continue

            # Numerator: absolute prediction error per horizon
            num = np.abs(aligned["yhat"].values - aligned["y"].values)

            # Denominator: rolling average absolute one-bar move using k_h = min(h+2, L)
            diffs = np.abs(np.diff(np.r_[base_val, aligned["y"].values]))  # length L = pred_len
            L = diffs.shape[0]
            cs = np.cumsum(diffs)
            ks = np.minimum(np.arange(1, L + 1) + 2, L)  # vector of k_h per horizon
            den = (cs[ks - 1] / ks) + self.eps

            ratios.append(num / den)

        if not ratios:
            return pd.Series(dtype=float)

        vse = np.mean(np.vstack(ratios), axis=0)
        horizons = np.arange(1, vse.shape[0] + 1)
        series = pd.Series(vse, index=horizons, name=f"VSE_{self.target_col}")
        self._log_series("vse", series, title=f"VSE per horizon ({self.target_col})")
        return series

    def compute_return_corr_bar_by_bar(self, slide_results: List[dict]) -> pd.Series:
        """
        Pearson correlation of returns (relative to t0) per prediction step across all windows.
        """
        if not slide_results:
            return pd.Series(dtype=float)
        dy_list, dyhat_list = [], []
        pred_len_ref = None
        for item in slide_results:
            pred_df = item["pred"]; kline_df = item["kline"]
            if self.target_col not in pred_df.columns or self.target_col not in kline_df.columns:
                continue
            pred_len = len(pred_df)
            if len(kline_df) < pred_len + 1:
                continue
            true_future = kline_df[self.target_col].iloc[-pred_len:]
            base_idx = len(kline_df) - pred_len - 1
            if base_idx < 0:
                continue
            base_val = kline_df[self.target_col].iloc[base_idx]
            pred_future = pred_df[self.target_col]
            aligned = pd.concat([true_future.rename("y"), pred_future.rename("yhat")], axis=1).dropna()
            if aligned.empty:
                continue
            if pred_len_ref is None:
                pred_len_ref = aligned.shape[0]
            elif aligned.shape[0] != pred_len_ref:
                continue
            dy_list.append(aligned["y"].values - base_val)
            dyhat_list.append(aligned["yhat"].values - base_val)
        if not dy_list:
            return pd.Series(dtype=float)
        dy = np.vstack(dy_list)
        dyhat = np.vstack(dyhat_list)
        corr = []
        for h in range(dy.shape[1]):
            a = dy[:, h]; b = dyhat[:, h]
            if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                corr.append(0.0)
            else:
                c = np.corrcoef(a, b)[0, 1]
                corr.append(np.nan_to_num(c, nan=0.0))
        horizons = np.arange(1, len(corr) + 1)
        series = pd.Series(corr, index=horizons, name=f"RetCorr_{self.target_col}")
        # W&B logging
        self._log_series("return_correlation", series, title=f"Return Correlation per horizon ({self.target_col})")
        return series

    def compute_r2_bar_by_bar(self, slide_results: List[dict]) -> pd.Series:
        """
        R² (coefficient of determination) per prediction step across all sliding windows.
        For each horizon h:
            R2_h = 1 - sum((ŷ_{t+h} - y_{t+h})^2) / sum((y_{t+h} - ȳ_h)^2)
        """
        if not slide_results:
            return pd.Series(dtype=float)

        y_list, yhat_list = [], []
        pred_len_ref = None

        for item in slide_results:
            pred_df = item["pred"]; kline_df = item["kline"]
            if self.target_col not in pred_df.columns or self.target_col not in kline_df.columns:
                continue

            true_future = kline_df[self.target_col].iloc[-len(pred_df):]
            pred_future = pred_df[self.target_col]
            aligned = pd.concat([true_future.rename("y"), pred_future.rename("yhat")], axis=1).dropna()
            if aligned.empty:
                continue

            if pred_len_ref is None:
                pred_len_ref = aligned.shape[0]
            elif aligned.shape[0] != pred_len_ref:
                continue

            y_list.append(aligned["y"].values)
            yhat_list.append(aligned["yhat"].values)

        if not y_list:
            return pd.Series(dtype=float)

        y = np.vstack(y_list)       # (num_windows, pred_len)
        yhat = np.vstack(yhat_list) # (num_windows, pred_len)

        y_bar = y.mean(axis=0)                      # (pred_len,)
        sse = np.sum((yhat - y) ** 2, axis=0)       # (pred_len,)
        sst = np.sum((y - y_bar) ** 2, axis=0)      # (pred_len,)

        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = np.where(sst > 0, 1.0 - (sse / sst), 0.0)

        horizons = np.arange(1, r2.shape[0] + 1)
        series = pd.Series(r2, index=horizons, name=f"R2_{self.target_col}")
        # W&B logging
        self._log_series("r2", series, title=f"R² per horizon ({self.target_col})")
        return series

    # --- generic plotting helper used by periodic analysis ---
    def plot_curve(self, series: pd.Series, ylabel: str, title: Optional[str] = None, show: bool = True, save_path: str = None):
        if series is None or series.empty:
            print(f"No data to plot for {ylabel}.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(series.index, series.values, marker="o")
        plt.xlabel("Prediction step (t + h)")
        plt.ylabel(ylabel)
        plt.title(title or ylabel)
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=130, bbox_inches="tight")
            plt.close()
        elif show:
            plt.show()
        else:
            plt.close()