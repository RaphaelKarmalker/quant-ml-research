"""
Kronos Prediction Benchmark

ECHTE PREDICTION: Context → Predictor → Future Tokens → Decode → Compare vs Actual

Was es macht:
1. Context Window (168h) tokenizen
2. Predictor generiert autoregessiv zukünftige Tokens (8h)
3. Tokens decoden zurück zu Features  
4. Vergleich: Predicted vs Actual Future
5. Screenshots speichern (nur jedes N-te Window)
6. Umfangreiche Metriken berechnen

Metriken:
- Regression: RMSE, MAE, MSE, R², MAPE
- Directional: Accuracy, Precision, Recall, F1 (Up/Down)
- Correlation: Pearson, Spearman (IC)
- Per-Horizon: Alle Metriken pro Stunde (h1-h8)

Usage:
    python benchmark_prediction.py                     # Alle Symbole
    python benchmark_prediction.py --max-symbols 20   # Nur 20 Symbole
    python benchmark_prediction.py --save-every 50    # Jedes 50. Bild speichern

Author: Kronos ML Team
Date: 2026-01-22
"""

import os
import sys
import json
import argparse
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Headless
import matplotlib.pyplot as plt
from scipy import stats

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model.kronos import KronosTokenizer, Kronos, calc_time_stamps


# =============================================================================
# Comprehensive Metrics
# =============================================================================

def compute_regression_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    pred = pred.flatten()
    actual = actual.flatten()
    
    # Remove NaN
    mask = ~(np.isnan(pred) | np.isnan(actual))
    pred, actual = pred[mask], actual[mask]
    
    if len(pred) == 0:
        return {'error': 'No valid data'}
    
    mse = np.mean((pred - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - actual))
    
    # R²
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # MAPE (avoid division by zero)
    nonzero = np.abs(actual) > 1e-8
    if np.sum(nonzero) > 0:
        mape = np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100
    else:
        mape = 0.0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2),
        'mape': float(mape),
        'n_samples': int(len(pred))
    }


def compute_directional_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Compute directional classification metrics."""
    pred = pred.flatten()
    actual = actual.flatten()
    
    mask = ~(np.isnan(pred) | np.isnan(actual))
    pred, actual = pred[mask], actual[mask]
    
    if len(pred) == 0:
        return {'direction_accuracy': 0.0}
    
    pred_sign = np.sign(pred)
    actual_sign = np.sign(actual)
    
    # Overall accuracy
    direction_accuracy = np.mean(pred_sign == actual_sign)
    
    # True Positives, False Positives, etc. for UP
    tp_up = np.sum((pred_sign == 1) & (actual_sign == 1))
    fp_up = np.sum((pred_sign == 1) & (actual_sign != 1))
    fn_up = np.sum((pred_sign != 1) & (actual_sign == 1))
    
    # True Positives, False Positives, etc. for DOWN
    tp_down = np.sum((pred_sign == -1) & (actual_sign == -1))
    fp_down = np.sum((pred_sign == -1) & (actual_sign != -1))
    fn_down = np.sum((pred_sign != -1) & (actual_sign == -1))
    
    # Precision, Recall, F1 for UP
    precision_up = tp_up / (tp_up + fp_up + 1e-8)
    recall_up = tp_up / (tp_up + fn_up + 1e-8)
    f1_up = 2 * precision_up * recall_up / (precision_up + recall_up + 1e-8)
    
    # Precision, Recall, F1 for DOWN
    precision_down = tp_down / (tp_down + fp_down + 1e-8)
    recall_down = tp_down / (tp_down + fn_down + 1e-8)
    f1_down = 2 * precision_down * recall_down / (precision_down + recall_down + 1e-8)
    
    # Macro F1
    f1_macro = (f1_up + f1_down) / 2
    
    return {
        'direction_accuracy': float(direction_accuracy),
        'precision_up': float(precision_up),
        'recall_up': float(recall_up),
        'f1_up': float(f1_up),
        'precision_down': float(precision_down),
        'recall_down': float(recall_down),
        'f1_down': float(f1_down),
        'f1_macro': float(f1_macro),
    }


def compute_correlation_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Compute correlation metrics."""
    pred = pred.flatten()
    actual = actual.flatten()
    
    mask = ~(np.isnan(pred) | np.isnan(actual))
    pred, actual = pred[mask], actual[mask]
    
    if len(pred) < 3:
        return {'pearson': 0.0, 'spearman_ic': 0.0}
    
    # Pearson correlation
    if np.std(pred) > 1e-8 and np.std(actual) > 1e-8:
        pearson = np.corrcoef(pred, actual)[0, 1]
    else:
        pearson = 0.0
    
    # Spearman (Information Coefficient)
    spearman, _ = stats.spearmanr(pred, actual)
    
    return {
        'pearson': float(pearson) if not np.isnan(pearson) else 0.0,
        'spearman_ic': float(spearman) if not np.isnan(spearman) else 0.0,
    }


def compute_per_horizon_metrics(pred: np.ndarray, actual: np.ndarray, 
                                 horizon: int) -> Dict[str, Dict]:
    """
    Compute metrics per prediction horizon.
    
    Args:
        pred: (n_windows, horizon, n_features) or (n_windows, horizon)
        actual: same shape
        horizon: number of prediction steps
    """
    metrics = {}
    
    for h in range(horizon):
        if len(pred.shape) == 3:
            p = pred[:, h, :].flatten()
            a = actual[:, h, :].flatten()
        else:
            p = pred[:, h] if pred.shape[1] > h else pred.flatten()
            a = actual[:, h] if actual.shape[1] > h else actual.flatten()
        
        metrics[f'h{h+1}'] = {
            **compute_regression_metrics(p, a),
            **compute_directional_metrics(p, a),
            **compute_correlation_metrics(p, a),
        }
    
    return metrics


def compute_ts_since_listing_metrics(pred: np.ndarray, actual: np.ndarray,
                                      ts_since_listing: np.ndarray,
                                      n_bins: int = 5) -> Dict[str, Dict]:
    """
    Compute metrics binned by ts_since_listing.
    
    Analyzes whether predictions are better for young coins (shortly after listing)
    vs mature coins (long after listing).
    
    Args:
        pred: (n_windows, horizon, n_features) or flattened
        actual: same shape
        ts_since_listing: ts_since_listing value for each window (n_windows,)
        n_bins: Number of bins to divide ts_since_listing into
    
    Returns:
        Dict with metrics per bin and trend analysis
    """
    # Flatten if needed
    if len(pred.shape) > 1:
        pred_flat = pred.reshape(pred.shape[0], -1)
        actual_flat = actual.reshape(actual.shape[0], -1)
    else:
        pred_flat = pred.reshape(-1, 1)
        actual_flat = actual.reshape(-1, 1)
    
    # Remove NaN ts_since_listing
    valid_mask = ~np.isnan(ts_since_listing) & (ts_since_listing > 0)
    if np.sum(valid_mask) < 10:
        return {'error': 'Not enough valid ts_since_listing values'}
    
    ts_valid = ts_since_listing[valid_mask]
    pred_valid = pred_flat[valid_mask]
    actual_valid = actual_flat[valid_mask]
    
    # Create bins (quantile-based for even distribution)
    try:
        bin_edges = np.percentile(ts_valid, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        if len(bin_edges) < 2:
            bin_edges = np.array([ts_valid.min(), ts_valid.max()])
    except Exception:
        return {'error': 'Could not create bins'}
    
    bin_indices = np.digitize(ts_valid, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    metrics = {
        'n_bins': len(bin_edges) - 1,
        'bin_edges': bin_edges.tolist(),
        'bins': {}
    }
    
    bin_dir_acc = []
    bin_rmse = []
    bin_corr = []
    bin_centers = []
    
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        if np.sum(mask) < 5:
            continue
        
        p = pred_valid[mask].flatten()
        a = actual_valid[mask].flatten()
        
        bin_min = bin_edges[b]
        bin_max = bin_edges[b + 1]
        bin_center = (bin_min + bin_max) / 2
        
        reg_metrics = compute_regression_metrics(p, a)
        dir_metrics = compute_directional_metrics(p, a)
        corr_metrics = compute_correlation_metrics(p, a)
        
        bin_label = f'bin_{b}'
        metrics['bins'][bin_label] = {
            'ts_range': [float(bin_min), float(bin_max)],
            'ts_center': float(bin_center),
            'n_samples': int(np.sum(mask)),
            **reg_metrics,
            **dir_metrics,
            **corr_metrics,
        }
        
        bin_centers.append(bin_center)
        bin_dir_acc.append(dir_metrics.get('direction_accuracy', 0))
        bin_rmse.append(reg_metrics.get('rmse', 0))
        bin_corr.append(corr_metrics.get('pearson', 0))
    
    # Compute trend (correlation between ts_since_listing bin and metric)
    if len(bin_centers) >= 3:
        try:
            # Spearman correlation to see if metrics improve/degrade with coin age
            trend_dir_acc, _ = stats.spearmanr(bin_centers, bin_dir_acc)
            trend_rmse, _ = stats.spearmanr(bin_centers, bin_rmse)
            trend_corr, _ = stats.spearmanr(bin_centers, bin_corr)
            
            metrics['trend'] = {
                'direction_accuracy_vs_age': float(trend_dir_acc) if not np.isnan(trend_dir_acc) else 0,
                'rmse_vs_age': float(trend_rmse) if not np.isnan(trend_rmse) else 0,
                'correlation_vs_age': float(trend_corr) if not np.isnan(trend_corr) else 0,
                'interpretation': {
                    'direction_accuracy': 'better for older coins' if trend_dir_acc > 0.3 else ('better for younger coins' if trend_dir_acc < -0.3 else 'no clear trend'),
                    'rmse': 'worse for older coins' if trend_rmse > 0.3 else ('worse for younger coins' if trend_rmse < -0.3 else 'no clear trend'),
                    'correlation': 'better for older coins' if trend_corr > 0.3 else ('better for younger coins' if trend_corr < -0.3 else 'no clear trend'),
                }
            }
        except Exception:
            metrics['trend'] = {'error': 'Could not compute trend'}
    
    return metrics


# =============================================================================
# Plotting
# =============================================================================

def plot_prediction_comparison(context: np.ndarray, pred_future: np.ndarray, 
                                actual_future: np.ndarray, feature_names: List[str],
                                symbol: str, window_idx: int, save_path: str,
                                feature_indices: List[int] = None):
    """
    Plot prediction vs actual for selected features.
    
    Args:
        context: Context window features (lookback, n_features)
        pred_future: Predicted future (horizon, n_features)
        actual_future: Actual future (horizon, n_features)
        feature_names: Names of features
        symbol: Symbol name
        window_idx: Window index for title
        save_path: Path to save the figure
        feature_indices: Which features to plot (default: first 4 return features)
    """
    if feature_indices is None:
        # Default: plot return features
        feature_indices = [i for i, f in enumerate(feature_names) if 'returns' in f.lower()][:4]
    
    n_features = len(feature_indices)
    if n_features == 0:
        return
    
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]
    
    lookback = len(context)
    horizon = len(pred_future)
    
    x_context = np.arange(lookback)
    x_future = np.arange(lookback, lookback + horizon)
    x_all = np.arange(lookback + horizon)
    
    for ax, feat_idx in zip(axes, feature_indices):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature {feat_idx}'
        
        # Context (history)
        ax.plot(x_context, context[:, feat_idx], 'b-', label='Context', alpha=0.7, linewidth=1)
        
        # Actual future
        ax.plot(x_future, actual_future[:, feat_idx], 'g-', label='Actual', linewidth=2)
        ax.scatter(x_future, actual_future[:, feat_idx], c='green', s=30, zorder=5)
        
        # Predicted future
        ax.plot(x_future, pred_future[:, feat_idx], 'r--', label='Predicted', linewidth=2)
        ax.scatter(x_future, pred_future[:, feat_idx], c='red', s=30, zorder=5, marker='x')
        
        # Vertical line at prediction start
        ax.axvline(x=lookback - 0.5, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_ylabel(feat_name, fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Highlight prediction area
        ax.axvspan(lookback - 0.5, lookback + horizon - 0.5, alpha=0.1, color='yellow')
    
    axes[-1].set_xlabel('Time Step (hours)', fontsize=10)
    fig.suptitle(f'{symbol} - Window {window_idx}\nContext: {lookback}h → Predict: {horizon}h', 
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_ts_since_listing_analysis(metrics: Dict, save_path: str):
    """
    Plot prediction quality vs ts_since_listing (coin age).
    
    Shows how direction accuracy, RMSE, and correlation change
    as coins get older (more time since listing).
    """
    if 'ts_since_listing' not in metrics or 'bins' not in metrics.get('ts_since_listing', {}):
        return
    
    ts_metrics = metrics['ts_since_listing']
    bins = ts_metrics['bins']
    
    if len(bins) < 2:
        return
    
    # Extract data
    bin_labels = sorted(bins.keys())
    centers = [bins[b]['ts_center'] for b in bin_labels]
    dir_acc = [bins[b].get('direction_accuracy', 0) * 100 for b in bin_labels]
    rmse = [bins[b].get('rmse', 0) for b in bin_labels]
    corr = [bins[b].get('pearson', 0) for b in bin_labels]
    n_samples = [bins[b].get('n_samples', 0) for b in bin_labels]
    
    # Convert centers to readable labels (hours/days)
    x_labels = []
    for c in centers:
        if c < 24:
            x_labels.append(f'{c:.0f}h')
        elif c < 24 * 30:
            x_labels.append(f'{c/24:.0f}d')
        else:
            x_labels.append(f'{c/(24*30):.1f}mo')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Direction Accuracy vs Coin Age
    ax = axes[0, 0]
    bars = ax.bar(range(len(bin_labels)), dir_acc, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel('Direction Accuracy (%)')
    ax.set_xlabel('Time Since Listing (bin center)')
    ax.set_title('Direction Accuracy vs Coin Age')
    ax.axhline(y=50, color='red', linestyle='--', label='Random (50%)', alpha=0.7)
    ax.legend()
    ax.set_ylim(0, 100)
    for i, v in enumerate(dir_acc):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=8)
    
    # 2. RMSE vs Coin Age
    ax = axes[0, 1]
    ax.bar(range(len(bin_labels)), rmse, color='coral', alpha=0.8)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Time Since Listing (bin center)')
    ax.set_title('RMSE vs Coin Age (lower is better)')
    for i, v in enumerate(rmse):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=8)
    
    # 3. Correlation vs Coin Age
    ax = axes[1, 0]
    colors = ['green' if c > 0 else 'red' for c in corr]
    ax.bar(range(len(bin_labels)), corr, color=colors, alpha=0.8)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel('Pearson Correlation')
    ax.set_xlabel('Time Since Listing (bin center)')
    ax.set_title('Prediction Correlation vs Coin Age')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(-1, 1)
    
    # 4. Sample distribution & Trend summary
    ax = axes[1, 1]
    ax.bar(range(len(bin_labels)), n_samples, color='purple', alpha=0.6)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylabel('Number of Windows')
    ax.set_xlabel('Time Since Listing (bin center)')
    ax.set_title('Sample Distribution by Coin Age')
    
    # Add trend text
    if 'trend' in ts_metrics and 'interpretation' not in ts_metrics['trend'].get('error', ''):
        trend = ts_metrics['trend']
        trend_text = f"Trends:\n"
        trend_text += f"  Dir. Acc: {trend.get('interpretation', {}).get('direction_accuracy', 'N/A')}\n"
        trend_text += f"  RMSE: {trend.get('interpretation', {}).get('rmse', 'N/A')}\n"
        trend_text += f"  Corr: {trend.get('interpretation', {}).get('correlation', 'N/A')}"
        ax.text(0.95, 0.95, trend_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Prediction Quality vs Time Since Listing (Coin Age)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_summary(metrics: Dict, save_path: str):
    """Plot summary of metrics as bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Helper to extract value (handles both flat metrics and aggregated dicts)
    def get_metric_value(d, key, default=0):
        val = d.get(key, default)
        if isinstance(val, dict):
            return val.get('mean', default)
        return val
    
    # 1. Per-horizon Direction Accuracy
    ax = axes[0, 0]
    horizons = sorted([k for k in metrics.get('per_horizon', {}).keys()])
    if horizons:
        dir_acc = [get_metric_value(metrics['per_horizon'][h], 'direction_accuracy', 0) * 100 for h in horizons]
        ax.bar(range(len(horizons)), dir_acc, color='steelblue')
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons)
        ax.set_ylabel('Direction Accuracy (%)')
        ax.set_xlabel('Horizon')
        ax.set_title('Direction Accuracy per Horizon')
        ax.axhline(y=50, color='red', linestyle='--', label='Random (50%)')
        ax.legend()
        ax.set_ylim(0, 100)
        for i, v in enumerate(dir_acc):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9)
    
    # 2. Per-horizon RMSE
    ax = axes[0, 1]
    if horizons:
        rmse = [get_metric_value(metrics['per_horizon'][h], 'rmse', 0) for h in horizons]
        ax.bar(range(len(horizons)), rmse, color='coral')
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons)
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Horizon')
        ax.set_title('RMSE per Horizon')
    
    # 3. Per-horizon Correlation
    ax = axes[1, 0]
    if horizons:
        corr = [get_metric_value(metrics['per_horizon'][h], 'pearson', 0) for h in horizons]
        ax.bar(range(len(horizons)), corr, color='seagreen')
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(horizons)
        ax.set_ylabel('Pearson Correlation')
        ax.set_xlabel('Horizon')
        ax.set_title('Correlation per Horizon')
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='gray', linestyle='--')
    
    # 4. Overall metrics summary
    ax = axes[1, 1]
    overall = metrics.get('overall', {})
    if overall:
        metric_names = ['direction_accuracy', 'f1_macro', 'pearson', 'r2']
        metric_labels = ['Dir. Accuracy', 'F1 Macro', 'Correlation', 'R²']
        values = [get_metric_value(overall, m, 0) for m in metric_names]
        colors = ['steelblue', 'orange', 'seagreen', 'purple']
        
        bars = ax.bar(metric_labels, values, color=colors)
        ax.set_ylabel('Value')
        ax.set_title('Overall Metrics Summary')
        ax.set_ylim(0, 1)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', 
                    ha='center', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Prediction Engine
# =============================================================================

class PredictionBenchmark:
    """
    Benchmark engine for REAL prediction (not just reconstruction).
    
    Pipeline:
    1. Tokenize context window
    2. Predictor generates future tokens autoregressively
    3. Decode tokens back to features
    4. Compare with actual future
    """
    
    def __init__(self, config: Config = None, device: str = None):
        self.config = config or Config()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.predictor = None
        self.val_data = None
        self.symbols = []
        
    def load_models(self):
        """Load tokenizer and predictor."""
        print("Loading models...")
        
        # Tokenizer
        tokenizer_path = os.path.join(
            self.config.finetuned_tokenizer_path, 'tokenizer.pt'
        )
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        checkpoint = torch.load(tokenizer_path, map_location=self.device, weights_only=False)
        arch = checkpoint.get('config', {}).get('tokenizer_arch', self.config.tokenizer_arch)
        
        self.tokenizer = KronosTokenizer(
            arch['d_in'], arch['d_model'], arch['n_heads'], arch['ff_dim'],
            arch['n_enc_layers'], arch['n_dec_layers'],
            arch['ffn_dropout_p'], arch['attn_dropout_p'], arch['resid_dropout_p'],
            arch['s1_bits'], arch['s2_bits'],
            arch['beta'], arch['gamma0'], arch['gamma'], arch['zeta'], arch['group_size']
        )
        self.tokenizer.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer.eval().to(self.device)
        print(f"  Tokenizer loaded (val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        
        # Predictor
        predictor_path = os.path.join(
            self.config.finetuned_predictor_path, 'predictor.pt'
        )
        
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")
        
        checkpoint = torch.load(predictor_path, map_location=self.device, weights_only=False)
        arch = checkpoint.get('config', {}).get('predictor_arch', self.config.predictor_arch)
        
        self.predictor = Kronos(
            arch['s1_bits'], arch['s2_bits'], arch['n_layers'], arch['d_model'],
            arch['n_heads'], arch['ff_dim'],
            arch['ffn_dropout_p'], arch['attn_dropout_p'], arch['resid_dropout_p'],
            arch['token_dropout_p'], arch['learn_te'],
            time_feature_list=self.config.time_feature_list
        )
        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.predictor.eval().to(self.device)
        print(f"  Predictor loaded (val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        
        print("Models loaded!\n")
    
    def load_data(self):
        """Load validation data."""
        val_path = os.path.join(self.config.dataset_path, 'val_data.pkl')
        
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        
        with open(val_path, 'rb') as f:
            self.val_data = pickle.load(f)
        
        self.symbols = list(self.val_data.keys())
        print(f"Loaded validation data: {len(self.symbols)} symbols\n")
    
    def _normalize_window(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features per-window."""
        x_norm = np.zeros_like(x)
        means = np.zeros(x.shape[1])
        stds = np.zeros(x.shape[1])
        
        return_features = [f for f in self.config.feature_list if 'returns' in f.lower()]
        n_return_features = len(return_features)
        
        for i in range(x.shape[1]):
            col = x[:, i]
            
            if i < n_return_features:
                median = np.median(col)
                q75, q25 = np.percentile(col, [75, 25])
                iqr = q75 - q25
                if iqr < 1e-8:
                    iqr = np.std(col) + 1e-8
                x_norm[:, i] = (col - median) / (iqr + 1e-8)
                means[i] = median
                stds[i] = iqr
            else:
                mean = np.mean(col)
                std = np.std(col)
                if std < 1e-8:
                    std = 1e-8
                x_norm[:, i] = (col - mean) / (std + 1e-5)
                means[i] = mean
                stds[i] = std
        
        x_norm = np.clip(x_norm, -self.config.clip, self.config.clip)
        return x_norm, means, stds
    
    def _denormalize(self, x_norm: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Denormalize features."""
        return x_norm * stds + means
    
    @torch.no_grad()
    def predict_window(self, context: np.ndarray, context_stamp: np.ndarray,
                       future_stamp: np.ndarray, means: np.ndarray, 
                       stds: np.ndarray) -> np.ndarray:
        """
        Predict future features from context using autoregressive generation.
        
        Args:
            context: Normalized context features (lookback, n_features)
            context_stamp: Time features for context (lookback, n_time_features)
            future_stamp: Time features for future (horizon, n_time_features)
            means: Normalization means
            stds: Normalization stds
        
        Returns:
            Denormalized predicted future (horizon, n_features)
        """
        horizon = len(future_stamp)
        
        # To tensors
        context_t = torch.from_numpy(context).float().unsqueeze(0).to(self.device)
        context_stamp_t = torch.from_numpy(context_stamp).float().unsqueeze(0).to(self.device)
        
        # Encode context to tokens
        token_seq_0, token_seq_1 = self.tokenizer.encode(context_t, half=True)
        
        # Autoregressive generation
        predicted_features = []
        
        current_tokens_0 = token_seq_0
        current_tokens_1 = token_seq_1
        current_stamp = context_stamp_t
        
        for step in range(horizon):
            # Get future timestamp for this step
            future_step = torch.from_numpy(
                future_stamp[step:step+1]
            ).float().unsqueeze(0).to(self.device)
            
            # Full stamp sequence (context + current future step)
            full_stamp = torch.cat([current_stamp, future_step], dim=1)
            
            # Predict next token logits
            logits = self.predictor(current_tokens_0, current_tokens_1, full_stamp[:, :-1])
            
            # Greedy decode (take argmax)
            pred_token_0 = logits[0][:, -1:].argmax(dim=-1)
            pred_token_1 = logits[1][:, -1:].argmax(dim=-1)
            
            # Use tokenizer's built-in decode method (handles scaling correctly)
            decoded = self.tokenizer.decode([pred_token_0, pred_token_1], half=True)
            
            pred_step = decoded.squeeze(0).squeeze(0).cpu().numpy()
            predicted_features.append(pred_step)
            
            # Update for next step
            current_tokens_0 = torch.cat([current_tokens_0, pred_token_0], dim=1)
            current_tokens_1 = torch.cat([current_tokens_1, pred_token_1], dim=1)
            current_stamp = full_stamp
        
        # Stack and denormalize
        predicted_norm = np.vstack(predicted_features)
        predicted = self._denormalize(predicted_norm, means, stds)
        
        return predicted
    
    def benchmark_symbol(self, symbol: str, stride: int = 8, 
                         max_windows: int = 100, save_plots_every: int = 50,
                         plot_dir: str = None) -> Dict:
        """
        Benchmark prediction for a single symbol.
        
        Args:
            symbol: Symbol to benchmark
            stride: Sliding window stride
            max_windows: Maximum windows to evaluate
            save_plots_every: Save plot every N windows (0 = no plots)
            plot_dir: Directory to save plots
        """
        if symbol not in self.val_data:
            return {'error': f'Symbol {symbol} not found'}
        
        df = self.val_data[symbol]
        feature_list = self.config.feature_list
        time_feature_list = self.config.time_feature_list
        
        available_features = [f for f in feature_list if f in df.columns]
        if len(available_features) < 5:
            return {'error': f'Not enough features'}
        
        lookback = self.config.lookback_window
        horizon = self.config.predict_window
        window_size = lookback + horizon
        
        if len(df) < window_size:
            return {'error': f'Not enough data'}
        
        # Collect predictions
        all_pred = []
        all_actual = []
        all_context = []
        all_ts_since_listing = []  # Track coin age for each window
        
        n_windows = min((len(df) - window_size) // stride + 1, max_windows)
        plots_saved = 0
        max_plots = 20  # Max plots per symbol
        
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            if end_idx > len(df):
                break
            
            window_df = df.iloc[start_idx:end_idx]
            
            # Extract features
            x = window_df[available_features].values.astype(np.float32)
            
            # Pad missing features
            if len(available_features) < len(feature_list):
                full_x = np.zeros((x.shape[0], len(feature_list)), dtype=np.float32)
                for j, f in enumerate(feature_list):
                    if f in available_features:
                        feat_idx = available_features.index(f)
                        full_x[:, j] = x[:, feat_idx]
                x = full_x
            
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get time features
            stamp_df = calc_time_stamps(
                window_df.index,
                extra_df=window_df,
                time_feature_list=time_feature_list
            )
            available_time = [c for c in time_feature_list if c in stamp_df.columns]
            if available_time:
                stamp = stamp_df[available_time].values.astype(np.float32)
            else:
                stamp = np.zeros((len(window_df), len(time_feature_list)), dtype=np.float32)
            
            # Split context and future
            context = x[:lookback]
            actual_future = x[lookback:]
            context_stamp = stamp[:lookback]
            future_stamp = stamp[lookback:]
            
            # Normalize (on context only to avoid data leakage)
            context_norm, means, stds = self._normalize_window(context)
            
            # Predict
            pred_future = self.predict_window(
                context_norm, context_stamp, future_stamp, means, stds
            )
            
            # Denormalize actual future for comparison
            actual_future_denorm = actual_future  # Already in original scale
            
            all_pred.append(pred_future)
            all_actual.append(actual_future_denorm)
            all_context.append(self._denormalize(context_norm, means, stds))
            
            # Track ts_since_listing (use mean of context window)
            if 'ts_since_listing' in window_df.columns:
                ts_val = window_df['ts_since_listing'].iloc[:lookback].mean()
            elif 'ts_since_listing_log' in window_df.columns:
                # Convert from log scale
                ts_val = np.exp(window_df['ts_since_listing_log'].iloc[:lookback].mean())
            else:
                ts_val = np.nan
            all_ts_since_listing.append(ts_val)
            
            # Save plot periodically
            if plot_dir and save_plots_every > 0 and i % save_plots_every == 0 and plots_saved < max_plots:
                plot_path = os.path.join(plot_dir, symbol, f'window_{i:04d}.png')
                plot_prediction_comparison(
                    context=all_context[-1],
                    pred_future=pred_future,
                    actual_future=actual_future_denorm,
                    feature_names=feature_list,
                    symbol=symbol,
                    window_idx=i,
                    save_path=plot_path
                )
                plots_saved += 1
        
        if not all_pred:
            return {'error': 'No valid windows'}
        
        # Stack results
        pred_array = np.stack(all_pred)  # (n_windows, horizon, n_features)
        actual_array = np.stack(all_actual)
        ts_since_listing_array = np.array(all_ts_since_listing)
        
        # Compute metrics
        # Focus on return features for directional metrics
        return_indices = [j for j, f in enumerate(feature_list) if 'returns' in f.lower()]
        
        metrics = {
            'symbol': symbol,
            'n_windows': len(all_pred),
            'lookback': lookback,
            'horizon': horizon,
        }
        
        # Overall metrics (all features)
        metrics['overall_all_features'] = {
            **compute_regression_metrics(pred_array, actual_array),
            **compute_correlation_metrics(pred_array, actual_array),
        }
        
        # Return features only
        if return_indices:
            pred_returns = pred_array[:, :, return_indices]
            actual_returns = actual_array[:, :, return_indices]
            
            metrics['overall'] = {
                **compute_regression_metrics(pred_returns, actual_returns),
                **compute_directional_metrics(pred_returns, actual_returns),
                **compute_correlation_metrics(pred_returns, actual_returns),
            }
            
            # Per-horizon metrics
            metrics['per_horizon'] = compute_per_horizon_metrics(
                pred_returns, actual_returns, horizon
            )
            
            # Metrics by ts_since_listing (coin age)
            if len(ts_since_listing_array) > 0 and not np.all(np.isnan(ts_since_listing_array)):
                metrics['ts_since_listing'] = compute_ts_since_listing_metrics(
                    pred_returns, actual_returns, ts_since_listing_array, n_bins=5
                )
        
        return metrics
    
    def run_benchmark(self, symbols: List[str] = None, stride: int = 10,
                      max_windows_per_symbol: int = 100, max_symbols: int = None,
                      save_plots_every: int = 50, output_dir: str = None) -> Dict:
        """
        Run benchmark on multiple symbols.
        """
        if self.tokenizer is None:
            self.load_models()
        if self.val_data is None:
            self.load_data()
        
        if symbols is None:
            symbols = self.symbols
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        # Setup output directory with date and config info
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_dir is None:
            # Create structured results directory
            results_base = os.path.join(self.config.save_path, 'results')
            
            # Config summary for folder name
            config_summary = f"lb{self.config.lookback_window}_pred{self.config.predict_window}_stride{stride}"
            
            output_dir = os.path.join(results_base, f'{timestamp}_{config_summary}')
        
        os.makedirs(output_dir, exist_ok=True)
        
        plot_dir = os.path.join(output_dir, 'plots')
        
        # Save run config for reproducibility
        run_config = {
            'timestamp': timestamp,
            'stride': stride,
            'max_windows_per_symbol': max_windows_per_symbol,
            'max_symbols': max_symbols,
            'save_plots_every': save_plots_every,
            'n_symbols': len(symbols),
            'model_config': {
                'lookback_window': self.config.lookback_window,
                'predict_window': self.config.predict_window,
                'feature_list': self.config.feature_list,
                'time_feature_list': self.config.time_feature_list,
                'tokenizer_path': self.config.finetuned_tokenizer_path,
                'predictor_path': self.config.finetuned_predictor_path,
            }
        }
        
        config_file = os.path.join(output_dir, 'run_config.json')
        with open(config_file, 'w') as f:
            json.dump(run_config, f, indent=2, default=str)
        
        print(f"Benchmarking {len(symbols)} symbols...")
        print(f"  Stride: {stride}")
        print(f"  Max windows/symbol: {max_windows_per_symbol}")
        print(f"  Save plots every: {save_plots_every} windows")
        print(f"  Output: {output_dir}\n")
        
        all_metrics = []
        symbol_metrics = {}
        
        for symbol in tqdm(symbols, desc="Predicting"):
            try:
                metrics = self.benchmark_symbol(
                    symbol,
                    stride=stride,
                    max_windows=max_windows_per_symbol,
                    save_plots_every=save_plots_every,
                    plot_dir=plot_dir
                )
                
                if 'error' not in metrics:
                    all_metrics.append(metrics)
                    symbol_metrics[symbol] = metrics
                    
            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")
        
        # Aggregate metrics
        if all_metrics:
            aggregate = self._aggregate_metrics(all_metrics)
            
            # Save summary plot
            plot_metrics_summary(aggregate, os.path.join(output_dir, 'metrics_summary.png'))
            
            # Save ts_since_listing analysis plot
            if 'ts_since_listing' in aggregate:
                plot_ts_since_listing_analysis(
                    aggregate, 
                    os.path.join(output_dir, 'ts_since_listing_analysis.png')
                )
        else:
            aggregate = {'error': 'No valid symbols'}
        
        return {
            'aggregate': aggregate,
            'by_symbol': symbol_metrics,
            'output_dir': output_dir,
            'config': {
                'lookback_window': self.config.lookback_window,
                'predict_window': self.config.predict_window,
                'n_features': len(self.config.feature_list),
                'feature_list': self.config.feature_list,
            }
        }
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across symbols."""
        aggregate = {
            'n_symbols': len(all_metrics),
            'total_windows': sum(m['n_windows'] for m in all_metrics),
        }
        
        # Aggregate overall metrics
        overall_keys = ['rmse', 'mae', 'r2', 'direction_accuracy', 'f1_macro', 
                        'pearson', 'spearman_ic', 'precision_up', 'precision_down',
                        'recall_up', 'recall_down', 'f1_up', 'f1_down']
        
        overall_values = {k: [] for k in overall_keys}
        for m in all_metrics:
            if 'overall' in m:
                for k in overall_keys:
                    if k in m['overall']:
                        overall_values[k].append(m['overall'][k])
        
        aggregate['overall'] = {
            k: {
                'mean': float(np.mean(v)) if v else 0,
                'std': float(np.std(v)) if v else 0,
                'min': float(np.min(v)) if v else 0,
                'max': float(np.max(v)) if v else 0,
            }
            for k, v in overall_values.items() if v
        }
        
        # Aggregate per-horizon metrics
        horizon = self.config.predict_window
        per_horizon = {}
        
        for h in range(1, horizon + 1):
            h_key = f'h{h}'
            h_values = {k: [] for k in overall_keys}
            
            for m in all_metrics:
                if 'per_horizon' in m and h_key in m['per_horizon']:
                    for k in overall_keys:
                        if k in m['per_horizon'][h_key]:
                            h_values[k].append(m['per_horizon'][h_key][k])
            
            per_horizon[h_key] = {
                k: float(np.mean(v)) if v else 0
                for k, v in h_values.items() if v
            }
        
        aggregate['per_horizon'] = per_horizon
        
        # Aggregate ts_since_listing metrics
        ts_bins_all = {}  # bin_label -> list of metrics from each symbol
        for m in all_metrics:
            if 'ts_since_listing' in m and 'bins' in m['ts_since_listing']:
                for bin_label, bin_data in m['ts_since_listing']['bins'].items():
                    if bin_label not in ts_bins_all:
                        ts_bins_all[bin_label] = []
                    ts_bins_all[bin_label].append(bin_data)
        
        if ts_bins_all:
            # Average metrics per bin across symbols
            ts_agg_bins = {}
            for bin_label, bin_data_list in sorted(ts_bins_all.items()):
                ts_agg_bins[bin_label] = {
                    'ts_center': np.mean([d['ts_center'] for d in bin_data_list]),
                    'n_samples': sum(d['n_samples'] for d in bin_data_list),
                    'direction_accuracy': np.mean([d.get('direction_accuracy', 0) for d in bin_data_list]),
                    'rmse': np.mean([d.get('rmse', 0) for d in bin_data_list]),
                    'pearson': np.mean([d.get('pearson', 0) for d in bin_data_list]),
                    'f1_macro': np.mean([d.get('f1_macro', 0) for d in bin_data_list]),
                }
            
            # Compute trend on aggregated data
            if len(ts_agg_bins) >= 3:
                centers = [ts_agg_bins[b]['ts_center'] for b in sorted(ts_agg_bins.keys())]
                dir_accs = [ts_agg_bins[b]['direction_accuracy'] for b in sorted(ts_agg_bins.keys())]
                rmses = [ts_agg_bins[b]['rmse'] for b in sorted(ts_agg_bins.keys())]
                corrs = [ts_agg_bins[b]['pearson'] for b in sorted(ts_agg_bins.keys())]
                
                try:
                    trend_dir, _ = stats.spearmanr(centers, dir_accs)
                    trend_rmse, _ = stats.spearmanr(centers, rmses)
                    trend_corr, _ = stats.spearmanr(centers, corrs)
                    
                    trend = {
                        'direction_accuracy_vs_age': float(trend_dir) if not np.isnan(trend_dir) else 0,
                        'rmse_vs_age': float(trend_rmse) if not np.isnan(trend_rmse) else 0,
                        'correlation_vs_age': float(trend_corr) if not np.isnan(trend_corr) else 0,
                        'interpretation': {
                            'direction_accuracy': 'better for older coins' if trend_dir > 0.3 else ('better for younger coins' if trend_dir < -0.3 else 'no clear trend'),
                            'rmse': 'worse for older coins' if trend_rmse > 0.3 else ('worse for younger coins' if trend_rmse < -0.3 else 'no clear trend'),
                            'correlation': 'better for older coins' if trend_corr > 0.3 else ('better for younger coins' if trend_corr < -0.3 else 'no clear trend'),
                        }
                    }
                except Exception:
                    trend = {}
            else:
                trend = {}
            
            aggregate['ts_since_listing'] = {
                'bins': ts_agg_bins,
                'trend': trend,
            }
        
        return aggregate


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos Prediction Benchmark')
    parser.add_argument('--symbol', type=str, default=None, help='Specific symbol')
    parser.add_argument('--stride', type=int, default=10, help='Sliding window stride (default: 10)')
    parser.add_argument('--max-windows', type=int, default=100, help='Max windows per symbol')
    parser.add_argument('--max-symbols', type=int, default=None, help='Max symbols')
    parser.add_argument('--save-every', type=int, default=50, help='Save plot every N windows')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    print("=" * 70)
    print("KRONOS PREDICTION BENCHMARK")
    print("Context → Predictor → Future Tokens → Decode → Compare")
    print("=" * 70)
    
    config = Config()
    engine = PredictionBenchmark(config)
    
    results = engine.run_benchmark(
        symbols=[args.symbol] if args.symbol else None,
        stride=args.stride,
        max_windows_per_symbol=args.max_windows,
        max_symbols=args.max_symbols,
        save_plots_every=args.save_every,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("PREDICTION BENCHMARK RESULTS")
    print("=" * 70)
    
    agg = results.get('aggregate', {})
    print(f"\nAggregate ({agg.get('n_symbols', 0)} symbols, {agg.get('total_windows', 0)} windows):")
    
    if 'overall' in agg:
        ov = agg['overall']
        print(f"\n  === RETURN PREDICTION METRICS ===")
        print(f"  Direction Accuracy: {ov.get('direction_accuracy', {}).get('mean', 0):.2%} ± {ov.get('direction_accuracy', {}).get('std', 0):.2%}")
        print(f"  F1 Macro:           {ov.get('f1_macro', {}).get('mean', 0):.4f}")
        print(f"  Precision (Up):     {ov.get('precision_up', {}).get('mean', 0):.2%}")
        print(f"  Precision (Down):   {ov.get('precision_down', {}).get('mean', 0):.2%}")
        print(f"  Recall (Up):        {ov.get('recall_up', {}).get('mean', 0):.2%}")
        print(f"  Recall (Down):      {ov.get('recall_down', {}).get('mean', 0):.2%}")
        print(f"  Pearson Corr:       {ov.get('pearson', {}).get('mean', 0):.4f}")
        print(f"  Spearman IC:        {ov.get('spearman_ic', {}).get('mean', 0):.4f}")
        print(f"  RMSE:               {ov.get('rmse', {}).get('mean', 0):.4f}")
        print(f"  R²:                 {ov.get('r2', {}).get('mean', 0):.4f}")
    
    if 'per_horizon' in agg:
        print(f"\n  === PER-HORIZON DIRECTION ACCURACY ===")
        for h_key in sorted(agg['per_horizon'].keys()):
            h_metrics = agg['per_horizon'][h_key]
            print(f"    {h_key}: {h_metrics.get('direction_accuracy', 0):.2%}  (F1: {h_metrics.get('f1_macro', 0):.3f}, Corr: {h_metrics.get('pearson', 0):.3f})")
    
    # Print ts_since_listing analysis
    if 'ts_since_listing' in agg:
        ts_agg = agg['ts_since_listing']
        print(f"\n  === PREDICTION QUALITY vs COIN AGE (ts_since_listing) ===")
        
        if 'bins' in ts_agg:
            print(f"  Bins (by time since listing):")
            for bin_label in sorted(ts_agg['bins'].keys()):
                b = ts_agg['bins'][bin_label]
                center_h = b['ts_center']
                if center_h < 24:
                    center_str = f"{center_h:.0f}h"
                elif center_h < 24 * 30:
                    center_str = f"{center_h/24:.0f}d"
                else:
                    center_str = f"{center_h/(24*30):.1f}mo"
                print(f"    {bin_label} (~{center_str}): Dir.Acc={b['direction_accuracy']:.2%}, RMSE={b['rmse']:.4f}, Corr={b['pearson']:.3f} (n={b['n_samples']})")
        
        if 'trend' in ts_agg and ts_agg['trend']:
            trend = ts_agg['trend']
            print(f"\n  Trend Analysis:")
            if 'interpretation' in trend:
                print(f"    Direction Accuracy: {trend['interpretation'].get('direction_accuracy', 'N/A')} (ρ={trend.get('direction_accuracy_vs_age', 0):.3f})")
                print(f"    RMSE:               {trend['interpretation'].get('rmse', 'N/A')} (ρ={trend.get('rmse_vs_age', 0):.3f})")
                print(f"    Correlation:        {trend['interpretation'].get('correlation', 'N/A')} (ρ={trend.get('correlation_vs_age', 0):.3f})")
    
    # Save results - output_dir is already set by run_benchmark with proper structure
    # Get the actual output_dir from the results config
    actual_output_dir = results.get('output_dir', args.output_dir or os.path.join(config.save_path, 'results'))
    
    # Results file (main metrics)
    results_file = os.path.join(actual_output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Also save a compact summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_symbols': agg.get('n_symbols', 0),
        'total_windows': agg.get('total_windows', 0),
        'stride': args.stride,
        'overall_metrics': {
            k: v.get('mean', 0) for k, v in agg.get('overall', {}).items()
        } if 'overall' in agg else {},
        'per_horizon_direction_accuracy': {
            h: m.get('direction_accuracy', 0) 
            for h, m in agg.get('per_horizon', {}).items()
        },
        'ts_since_listing_trend': agg.get('ts_since_listing', {}).get('trend', {}),
    }
    
    summary_file = os.path.join(actual_output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  Output directory: {actual_output_dir}")
    print(f"  Results saved to: {results_file}")
    print(f"  Summary saved to: {summary_file}")
    print(f"  Plots saved to: {os.path.join(actual_output_dir, 'plots')}")
    print(f"  Config saved to: {os.path.join(actual_output_dir, 'run_config.json')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
