"""
Kronos Prediction Benchmark for Bitget Data

This script performs ACTUAL PREDICTION (not just reconstruction):
1. Encode context window (168h) to tokens
2. Use Predictor to autoregressively generate future tokens
3. Decode predicted tokens back to features
4. Compare: Predicted Future vs Actual Future
5. Generate plots and save metrics

Metrics:
- RMSE, MAE, MSE, R², MAPE
- Direction Accuracy, Precision, Recall, F1
- Pearson & Spearman Correlation
- Per-feature and per-horizon breakdown

Usage:
    python bitget_prediction_benchmark.py                    # All symbols
    python bitget_prediction_benchmark.py --symbol BTCUSDT   # One symbol
    python bitget_prediction_benchmark.py --max-symbols 20   # Limit symbols
    python bitget_prediction_benchmark.py --save-plots       # Save plots to disk

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model.kronos import KronosTokenizer, Kronos, calc_time_stamps


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_all_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for predictions.
    
    Args:
        predictions: Predicted values (N,) or (N, features)
        actuals: Actual values (N,) or (N, features)
    
    Returns:
        Dictionary with all metrics
    """
    pred_flat = predictions.flatten()
    actual_flat = actuals.flatten()
    
    # Remove NaN
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(actual_flat))
    pred_flat = pred_flat[valid_mask]
    actual_flat = actual_flat[valid_mask]
    
    if len(pred_flat) == 0:
        return {'error': 'No valid predictions', 'n_samples': 0}
    
    # Regression metrics
    mse = np.mean((pred_flat - actual_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - actual_flat))
    
    # R² (coefficient of determination)
    ss_res = np.sum((actual_flat - pred_flat) ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual_flat - pred_flat) / (np.abs(actual_flat) + 1e-8))) * 100
    
    # Correlation
    pearson_corr = np.corrcoef(pred_flat, actual_flat)[0, 1] if (np.std(pred_flat) > 1e-8 and np.std(actual_flat) > 1e-8) else 0.0
    spearman_corr, _ = stats.spearmanr(pred_flat, actual_flat)
    
    # Directional metrics
    pred_sign = np.sign(pred_flat)
    actual_sign = np.sign(actual_flat)
    direction_accuracy = np.mean(pred_sign == actual_sign)
    
    # Up/Down metrics
    up_mask = actual_flat > 0
    down_mask = actual_flat < 0
    zero_mask = actual_flat == 0
    
    # Recall: when actual is up/down, how often do we predict correctly?
    up_recall = np.mean(pred_sign[up_mask] == 1) if np.sum(up_mask) > 0 else 0.0
    down_recall = np.mean(pred_sign[down_mask] == -1) if np.sum(down_mask) > 0 else 0.0
    
    # Precision: when we predict up/down, how often is it correct?
    pred_up_mask = pred_flat > 0
    pred_down_mask = pred_flat < 0
    
    up_precision = np.mean(actual_flat[pred_up_mask] > 0) if np.sum(pred_up_mask) > 0 else 0.0
    down_precision = np.mean(actual_flat[pred_down_mask] < 0) if np.sum(pred_down_mask) > 0 else 0.0
    
    # F1 scores
    up_f1 = 2 * (up_precision * up_recall) / (up_precision + up_recall + 1e-8)
    down_f1 = 2 * (down_precision * down_recall) / (down_precision + down_recall + 1e-8)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2),
        'mape': float(mape),
        'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        'direction_accuracy': float(direction_accuracy),
        'up_recall': float(up_recall),
        'down_recall': float(down_recall),
        'up_precision': float(up_precision),
        'down_precision': float(down_precision),
        'up_f1': float(up_f1),
        'down_f1': float(down_f1),
        'n_samples': int(len(pred_flat))
    }


# =============================================================================
# Prediction Engine
# =============================================================================

class PredictionBenchmarkEngine:
    """
    Benchmark engine for ACTUAL PREDICTION using Predictor.
    """
    
    def __init__(self, config: Config = None, device: str = None, save_plots: bool = False):
        self.config = config or Config()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.tokenizer = None
        self.predictor = None
        self.val_data = None
        self.symbols = []
        self.save_plots = save_plots
        self.plots_dir = None
        
        print(f"Using device: {self.device}")
    
    def load_models(self):
        """Load tokenizer and predictor."""
        print("Loading models...")
        
        # Tokenizer
        tokenizer_path = os.path.join(self.config.finetuned_tokenizer_path, 'tokenizer.pt')
        if os.path.exists(tokenizer_path):
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
            print(f"  Tokenizer loaded (val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        else:
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        self.tokenizer.eval().to(self.device)
        
        # Predictor
        predictor_path = os.path.join(self.config.finetuned_predictor_path, 'predictor.pt')
        if os.path.exists(predictor_path):
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
            print(f"  Predictor loaded (val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        else:
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")
        
        self.predictor.eval().to(self.device)
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
    def predict_window(self, x_context: np.ndarray, x_future: np.ndarray,
                       x_stamp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future features autoregressively.
        
        Args:
            x_context: Context features (lookback, n_features)
            x_future: Future features for comparison (predict_window, n_features)
            x_stamp: Time stamps (lookback + predict_window, n_time_features)
        
        Returns:
            Tuple of (predicted_future_denorm, actual_future_denorm)
        """
        # Normalize full window
        full_window = np.vstack([x_context, x_future])
        x_norm, means, stds = self._normalize_window(full_window)
        
        context_norm = x_norm[:len(x_context)]
        future_norm = x_norm[len(x_context):]
        
        # To tensor
        context_t = torch.from_numpy(context_norm).float().unsqueeze(0).to(self.device)
        stamp_t = torch.from_numpy(x_stamp[:len(x_context)]).float().unsqueeze(0).to(self.device)
        
        # Encode context to tokens
        token_seq_0, token_seq_1 = self.tokenizer.encode(context_t, half=True)
        
        # Autoregressive prediction
        pred_len = len(x_future)
        predicted_features = []
        
        for step in range(pred_len):
            # Predictor forward
            future_stamp_step = torch.from_numpy(
                x_stamp[len(x_context) + step:len(x_context) + step + 1]
            ).float().unsqueeze(0).to(self.device)
            
            full_stamp = torch.cat([stamp_t, future_stamp_step], dim=1)
            
            logits = self.predictor(token_seq_0, token_seq_1, full_stamp[:, :-1])
            
            # Greedy decode: argmax
            pred_token_0 = logits[0][:, -1:].argmax(dim=-1)
            pred_token_1 = logits[1][:, -1:].argmax(dim=-1)
            
            # Decode token to features
            s1_bits = self.tokenizer.s1_bits
            s2_bits = self.tokenizer.s2_bits
            
            quant_0 = torch.zeros(1, 1, s1_bits, device=self.device)
            quant_1 = torch.zeros(1, 1, s2_bits, device=self.device)
            
            for b in range(s1_bits):
                quant_0[:, :, b] = ((pred_token_0 >> b) & 1).float() * 2 - 1
            for b in range(s2_bits):
                quant_1[:, :, b] = ((pred_token_1 >> b) & 1).float() * 2 - 1
            
            quantized = torch.cat([quant_0, quant_1], dim=-1)
            
            z = self.tokenizer.post_quant_embed(quantized)
            for layer in self.tokenizer.decoder:
                z = layer(z)
            decoded = self.tokenizer.head(z)
            
            pred_features_step = decoded.squeeze(0).cpu().numpy()
            predicted_features.append(pred_features_step)
            
            # Update for next step
            token_seq_0 = torch.cat([token_seq_0, pred_token_0], dim=1)
            token_seq_1 = torch.cat([token_seq_1, pred_token_1], dim=1)
            stamp_t = full_stamp
        
        # Stack and denormalize
        predicted_norm = np.vstack(predicted_features)
        predicted_denorm = self._denormalize(predicted_norm, means, stds)
        actual_denorm = self._denormalize(future_norm, means, stds)
        
        return predicted_denorm, actual_denorm
    
    def _plot_prediction(self, symbol: str, predicted: np.ndarray, actual: np.ndarray,
                         metrics: Dict, window_idx: int):
        """Save plot of prediction vs actual."""
        if not self.save_plots or predicted.shape[1] < 3:
            return
        
        # Use first 3 return features
        return_feature_indices = [i for i, f in enumerate(self.config.feature_list) 
                                  if 'returns' in f.lower()][:3]
        
        if not return_feature_indices:
            return
        
        fig, axes = plt.subplots(len(return_feature_indices), 1, figsize=(12, 3 * len(return_feature_indices)))
        if len(return_feature_indices) == 1:
            axes = [axes]
        
        for ax, feat_idx in zip(axes, return_feature_indices):
            feat_name = self.config.feature_list[feat_idx]
            
            x = np.arange(len(actual))
            ax.plot(x, actual[:, feat_idx], 'b-o', label='Actual', linewidth=2, markersize=6)
            ax.plot(x, predicted[:, feat_idx], 'r--s', label='Predicted', linewidth=2, markersize=6)
            ax.set_ylabel(feat_name, fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Horizon (hours)', fontsize=11)
        fig.suptitle(f'{symbol} - Window {window_idx}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(self.plots_dir, f'{symbol}_window_{window_idx}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def benchmark_symbol(self, symbol: str, stride: int = 8,
                         max_windows: int = None, save_first_n_plots: int = 5) -> Dict:
        """
        Benchmark a single symbol with PREDICTION.
        """
        if symbol not in self.val_data:
            return {'error': f'Symbol {symbol} not found'}
        
        df = self.val_data[symbol]
        feature_list = self.config.feature_list
        
        available_features = [f for f in feature_list if f in df.columns]
        if len(available_features) < 5:
            return {'error': f'Not enough features: {len(available_features)}'}
        
        lookback = self.config.lookback_window
        predict_window = self.config.predict_window
        window_size = lookback + predict_window
        
        if len(df) < window_size:
            return {'error': f'Not enough data: {len(df)} < {window_size}'}
        
        all_predictions = []
        all_actuals = []
        
        n_windows = (len(df) - window_size) // stride + 1
        if max_windows:
            n_windows = min(n_windows, max_windows)
        
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
            
            x_context = x[:lookback]
            x_future = x[lookback:]
            
            # Get time stamps
            x_stamp_df = calc_time_stamps(
                window_df.index,
                extra_df=window_df,
                time_feature_list=self.config.time_feature_list
            )
            available_time = [c for c in self.config.time_feature_list if c in x_stamp_df.columns]
            if available_time:
                x_stamp = x_stamp_df[available_time].values.astype(np.float32)
            else:
                x_stamp = np.zeros((len(window_df), len(self.config.time_feature_list)), dtype=np.float32)
            
            # Predict
            predicted, actual = self.predict_window(x_context, x_future, x_stamp)
            
            all_predictions.append(predicted)
            all_actuals.append(actual)
            
            # Save plot
            if i < save_first_n_plots:
                self._plot_prediction(symbol, predicted, actual, {}, i)
        
        if not all_predictions:
            return {'error': 'No valid windows'}
        
        # Stack
        pred_all = np.vstack(all_predictions)
        actual_all = np.vstack(all_actuals)
        
        # Metrics
        return_feature_indices = [i for i, f in enumerate(feature_list) if 'returns' in f.lower()]
        
        metrics = {
            'symbol': symbol,
            'n_windows': len(all_predictions),
            'overall': compute_all_metrics(pred_all, actual_all),
            'per_horizon': {},
            'per_feature': {}
        }
        
        # Per-horizon metrics
        for h in range(pred_all.shape[0]):
            if h < len(actual_all):
                metrics['per_horizon'][f'h{h+1}'] = compute_all_metrics(
                    pred_all[h:h+1], actual_all[h:h+1]
                )
        
        # Per-feature metrics
        for j, feat in enumerate(feature_list):
            if j < pred_all.shape[1]:
                feat_metrics = compute_all_metrics(pred_all[:, j], actual_all[:, j])
                metrics['per_feature'][feat] = feat_metrics
        
        # Aggregate returns directional
        if return_feature_indices:
            pred_returns = pred_all[:, return_feature_indices]
            actual_returns = actual_all[:, return_feature_indices]
            metrics['returns_directional'] = compute_all_metrics(pred_returns, actual_returns)
        
        return metrics
    
    def run_benchmark(self, symbols: List[str] = None, stride: int = 8,
                      max_windows_per_symbol: int = 50,
                      max_symbols: int = None) -> Dict:
        """Run benchmark on multiple symbols."""
        if self.tokenizer is None:
            self.load_models()
        if self.val_data is None:
            self.load_data()
        
        if self.save_plots:
            self.plots_dir = os.path.join(self.config.save_path, 'prediction_plots')
            os.makedirs(self.plots_dir, exist_ok=True)
            print(f"Saving plots to: {self.plots_dir}\n")
        
        if symbols is None:
            symbols = self.symbols
        
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        print(f"Benchmarking {len(symbols)} symbols (PREDICTION MODE)")
        print(f"  Stride: {stride}, Max windows/symbol: {max_windows_per_symbol}\n")
        
        all_metrics = []
        symbol_metrics = {}
        
        for symbol in tqdm(symbols, desc="Prediction Benchmark"):
            try:
                metrics = self.benchmark_symbol(
                    symbol,
                    stride=stride,
                    max_windows=max_windows_per_symbol,
                    save_first_n_plots=3 if self.save_plots else 0
                )
                
                if 'error' not in metrics:
                    all_metrics.append(metrics)
                    symbol_metrics[symbol] = metrics
            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")
        
        # Aggregate
        if all_metrics:
            aggregate = self._aggregate_metrics(all_metrics)
        else:
            aggregate = {'error': 'No valid symbols'}
        
        return {
            'aggregate': aggregate,
            'by_symbol': symbol_metrics,
            'config': {
                'lookback_window': self.config.lookback_window,
                'predict_window': self.config.predict_window,
                'n_features': len(self.config.feature_list),
            }
        }
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across symbols."""
        aggregate = {
            'n_symbols': len(all_metrics),
            'total_windows': sum(m['n_windows'] for m in all_metrics),
        }
        
        # Aggregate overall metrics
        metric_keys = ['rmse', 'mae', 'mse', 'r2', 'mape', 'pearson_correlation',
                       'spearman_correlation', 'direction_accuracy', 'up_recall',
                       'down_recall', 'up_precision', 'down_precision', 'up_f1', 'down_f1']
        
        aggregate['overall'] = {}
        for key in metric_keys:
            values = [m['overall'].get(key, 0) for m in all_metrics if 'overall' in m and key in m['overall']]
            if values:
                aggregate['overall'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return aggregate


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Kronos PREDICTION Benchmark')
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max-windows', type=int, default=50)
    parser.add_argument('--max-symbols', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--save-plots', action='store_true', help='Save prediction plots')
    args = parser.parse_args()
    
    print("=" * 70)
    print("KRONOS PREDICTION BENCHMARK")
    print("(Encode Context → Predict Future Tokens → Decode → Evaluate)")
    print("=" * 70)
    print()
    
    config = Config()
    engine = PredictionBenchmarkEngine(config, save_plots=args.save_plots)
    
    if args.symbol:
        engine.load_models()
        engine.load_data()
        results = engine.benchmark_symbol(args.symbol, stride=args.stride, max_windows=args.max_windows)
        results = {'aggregate': results, 'by_symbol': {args.symbol: results}}
    else:
        results = engine.run_benchmark(
            stride=args.stride,
            max_windows_per_symbol=args.max_windows,
            max_symbols=args.max_symbols
        )
    
    # Print results
    print("\n" + "=" * 70)
    print("PREDICTION BENCHMARK RESULTS")
    print("=" * 70)
    
    agg = results.get('aggregate', {})
    print(f"\nAggregate ({agg.get('n_symbols', 0)} symbols, {agg.get('total_windows', 0)} windows):\n")
    
    if 'overall' in agg:
        o = agg['overall']
        print(f"  Regression Metrics:")
        print(f"    RMSE:  {o.get('rmse', {}).get('mean', 0):.4f} ± {o.get('rmse', {}).get('std', 0):.4f}")
        print(f"    MAE:   {o.get('mae', {}).get('mean', 0):.4f} ± {o.get('mae', {}).get('std', 0):.4f}")
        print(f"    R²:    {o.get('r2', {}).get('mean', 0):.4f} ± {o.get('r2', {}).get('std', 0):.4f}")
        print(f"    MAPE:  {o.get('mape', {}).get('mean', 0):.2f}% ± {o.get('mape', {}).get('std', 0):.2f}%")
        
        print(f"\n  Correlation:")
        print(f"    Pearson:  {o.get('pearson_correlation', {}).get('mean', 0):.4f} ± {o.get('pearson_correlation', {}).get('std', 0):.4f}")
        print(f"    Spearman: {o.get('spearman_correlation', {}).get('mean', 0):.4f} ± {o.get('spearman_correlation', {}).get('std', 0):.4f}")
        
        print(f"\n  Directional Accuracy:")
        print(f"    Overall:       {o.get('direction_accuracy', {}).get('mean', 0):.2%}")
        print(f"    Up Recall:     {o.get('up_recall', {}).get('mean', 0):.2%}")
        print(f"    Down Recall:   {o.get('down_recall', {}).get('mean', 0):.2%}")
        print(f"    Up Precision:  {o.get('up_precision', {}).get('mean', 0):.2%}")
        print(f"    Down Precision:{o.get('down_precision', {}).get('mean', 0):.2%}")
        print(f"    Up F1:         {o.get('up_f1', {}).get('mean', 0):.4f}")
        print(f"    Down F1:       {o.get('down_f1', {}).get('mean', 0):.4f}")
    
    # Save
    output_dir = args.output_dir or os.path.join(config.save_path, 'prediction_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'prediction_benchmark_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    if args.save_plots:
        print(f"Plots saved to: {engine.plots_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
