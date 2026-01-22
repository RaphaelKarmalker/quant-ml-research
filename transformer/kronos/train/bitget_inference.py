"""
Bitget Inference & Benchmarking Pipeline for Kronos Transformer

This script benchmarks the model by:
1. Encoding input windows to tokens (Tokenizer)
2. Decoding tokens back to features (Tokenizer decoder)
3. Comparing reconstructed features vs actual features

Metrics computed:
- RMSE, MAE, MSE per feature and aggregate
- Directional Accuracy (for return features)
- Correlation between predicted and actual
- Per-horizon breakdown

NO trading signals, NO backtest, NO extra head - pure token encode/decode benchmarking.

Usage:
    python bitget_inference.py                    # Benchmark on validation set
    python bitget_inference.py --symbol BTCUSDT   # Benchmark specific symbol
    python bitget_inference.py --max-symbols 20   # Limit symbols

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
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model.kronos import KronosTokenizer, Kronos, calc_time_stamps


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_regression_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics for feature predictions.
    """
    pred_flat = predictions.flatten()
    actual_flat = actuals.flatten()
    
    # Remove NaN
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(actual_flat))
    pred_flat = pred_flat[valid_mask]
    actual_flat = actual_flat[valid_mask]
    
    if len(pred_flat) == 0:
        return {'error': 'No valid predictions', 'n_samples': 0}
    
    mse = np.mean((pred_flat - actual_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - actual_flat))
    
    if np.std(pred_flat) > 1e-8 and np.std(actual_flat) > 1e-8:
        correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'n_samples': int(len(pred_flat))
    }


def compute_directional_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """
    Compute directional accuracy metrics (for return-like features).
    """
    pred_flat = predictions.flatten()
    actual_flat = actuals.flatten()
    
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(actual_flat))
    pred_flat = pred_flat[valid_mask]
    actual_flat = actual_flat[valid_mask]
    
    if len(pred_flat) == 0:
        return {'direction_accuracy': 0.0, 'up_accuracy': 0.0, 'down_accuracy': 0.0}
    
    # Sign agreement
    pred_sign = np.sign(pred_flat)
    actual_sign = np.sign(actual_flat)
    direction_accuracy = np.mean(pred_sign == actual_sign)
    
    # Separate up/down accuracy
    up_mask = actual_flat > 0
    down_mask = actual_flat < 0
    
    up_accuracy = np.mean(pred_sign[up_mask] == actual_sign[up_mask]) if np.sum(up_mask) > 0 else 0.0
    down_accuracy = np.mean(pred_sign[down_mask] == actual_sign[down_mask]) if np.sum(down_mask) > 0 else 0.0
    
    # Precision
    pred_up_mask = pred_flat > 0
    pred_down_mask = pred_flat < 0
    
    precision_up = np.mean(actual_flat[pred_up_mask] > 0) if np.sum(pred_up_mask) > 0 else 0.0
    precision_down = np.mean(actual_flat[pred_down_mask] < 0) if np.sum(pred_down_mask) > 0 else 0.0
    
    return {
        'direction_accuracy': float(direction_accuracy),
        'up_accuracy': float(up_accuracy),
        'down_accuracy': float(down_accuracy),
        'precision_up': float(precision_up),
        'precision_down': float(precision_down)
    }


# =============================================================================
# Benchmark Engine (Token encode/decode, no extra head)
# =============================================================================

class BitgetBenchmarkEngine:
    """
    Benchmark engine for Kronos model.
    Uses pure token encode -> decode pipeline (tokenizer only).
    """
    
    def __init__(self, config: Config = None, device: str = None):
        self.config = config or Config()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.val_data = None
        self.symbols = []
        
    def load_models(self):
        """Load tokenizer from checkpoint."""
        print("Loading tokenizer...")
        
        tokenizer_path = os.path.join(
            self.config.finetuned_tokenizer_path, 'tokenizer.pt'
        )
        
        if os.path.exists(tokenizer_path):
            print(f"  Loading from: {tokenizer_path}")
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
        print("Model loaded!\n")
    
    def load_data(self):
        """Load validation data."""
        val_path = os.path.join(self.config.dataset_path, 'val_data.pkl')
        
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        
        with open(val_path, 'rb') as f:
            self.val_data = pickle.load(f)
        
        self.symbols = list(self.val_data.keys())
        print(f"Loaded validation data: {len(self.symbols)} symbols")
    
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
        """Denormalize features back to original scale."""
        return x_norm * stds + means
    
    @torch.no_grad()
    def benchmark_window(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benchmark: encode -> decode a window.
        
        Args:
            x: Input features (seq_len, n_features)
        
        Returns:
            Tuple of (reconstructed, actual) both denormalized
        """
        # Normalize
        x_norm, means, stds = self._normalize_window(x)
        
        # To tensor
        x_t = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)
        
        # Forward through tokenizer (encode + quantize + decode)
        (z_pre, z), bsq_loss, quantized, z_indices = self.tokenizer(x_t)
        
        # z is the full reconstruction
        reconstructed_norm = z.squeeze(0).cpu().numpy()
        
        # Denormalize
        reconstructed = self._denormalize(reconstructed_norm, means, stds)
        actual = self._denormalize(x_norm, means, stds)
        
        return reconstructed, actual
    
    def benchmark_symbol(self, symbol: str, stride: int = 8, 
                         max_windows: int = None) -> Dict:
        """
        Benchmark a single symbol.
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
        
        # Collect predictions and actuals
        all_pred = []
        all_actual = []
        all_pred_future = []
        all_actual_future = []
        
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
            
            # Handle NaN
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Encode-decode benchmark
            reconstructed, actual = self.benchmark_window(x)
            
            all_pred.append(reconstructed)
            all_actual.append(actual)
            
            # Separate future portion (for prediction quality)
            all_pred_future.append(reconstructed[lookback:])
            all_actual_future.append(actual[lookback:])
        
        if not all_pred:
            return {'error': 'No valid windows'}
        
        # Stack results
        pred_all = np.vstack(all_pred)
        actual_all = np.vstack(all_actual)
        pred_future = np.vstack(all_pred_future)
        actual_future = np.vstack(all_actual_future)
        
        # Identify return features
        return_feature_indices = [i for i, f in enumerate(feature_list) if 'returns' in f.lower()]
        
        # Compute metrics
        metrics = {
            'symbol': symbol,
            'n_windows': len(all_pred),
        }
        
        # Overall reconstruction
        metrics['reconstruction'] = compute_regression_metrics(pred_all, actual_all)
        
        # Future portion only (most important for prediction)
        metrics['future_reconstruction'] = compute_regression_metrics(pred_future, actual_future)
        
        # Per-feature metrics
        metrics['per_feature'] = {}
        for i, feat in enumerate(feature_list):
            if i < pred_future.shape[1]:
                feat_metrics = compute_regression_metrics(
                    pred_future[:, i], actual_future[:, i]
                )
                
                if i in return_feature_indices:
                    dir_metrics = compute_directional_metrics(
                        pred_future[:, i], actual_future[:, i]
                    )
                    feat_metrics.update(dir_metrics)
                
                metrics['per_feature'][feat] = feat_metrics
        
        # Aggregate directional metrics for all return features
        if return_feature_indices:
            pred_returns = pred_future[:, return_feature_indices]
            actual_returns = actual_future[:, return_feature_indices]
            metrics['returns_directional'] = compute_directional_metrics(pred_returns, actual_returns)
        
        return metrics
    
    def run_benchmark(self, symbols: List[str] = None, stride: int = 8, 
                      max_windows_per_symbol: int = 100,
                      max_symbols: int = None) -> Dict:
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
        
        print(f"\nBenchmarking {len(symbols)} symbols...")
        print(f"  Stride: {stride}, Max windows/symbol: {max_windows_per_symbol}\n")
        
        all_metrics = []
        symbol_metrics = {}
        
        for symbol in tqdm(symbols, desc="Benchmarking"):
            try:
                metrics = self.benchmark_symbol(
                    symbol, 
                    stride=stride, 
                    max_windows=max_windows_per_symbol
                )
                
                if 'error' not in metrics:
                    all_metrics.append(metrics)
                    symbol_metrics[symbol] = metrics
                    
            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")
        
        # Aggregate metrics
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
                'feature_list': self.config.feature_list,
            }
        }
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across symbols."""
        aggregate = {
            'n_symbols': len(all_metrics),
            'total_windows': sum(m['n_windows'] for m in all_metrics),
        }
        
        # Aggregate reconstruction metrics
        for key in ['reconstruction', 'future_reconstruction']:
            values = {'rmse': [], 'mae': [], 'mse': [], 'correlation': []}
            for m in all_metrics:
                if key in m and 'rmse' in m[key]:
                    for metric in values.keys():
                        if metric in m[key]:
                            values[metric].append(m[key][metric])
            
            aggregate[key] = {
                metric: {
                    'mean': float(np.mean(v)) if v else 0,
                    'std': float(np.std(v)) if v else 0,
                }
                for metric, v in values.items() if v
            }
        
        # Aggregate directional metrics
        if 'returns_directional' in all_metrics[0]:
            dir_values = {
                'direction_accuracy': [],
                'up_accuracy': [],
                'down_accuracy': [],
                'precision_up': [],
                'precision_down': [],
            }
            for m in all_metrics:
                if 'returns_directional' in m:
                    for metric in dir_values.keys():
                        if metric in m['returns_directional']:
                            dir_values[metric].append(m['returns_directional'][metric])
            
            aggregate['returns_directional'] = {
                metric: {
                    'mean': float(np.mean(v)) if v else 0,
                    'std': float(np.std(v)) if v else 0,
                }
                for metric, v in dir_values.items() if v
            }
        
        return aggregate


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark Kronos tokenizer on Bitget data')
    parser.add_argument('--symbol', type=str, default=None, help='Specific symbol')
    parser.add_argument('--stride', type=int, default=8, help='Sliding window stride')
    parser.add_argument('--max-windows', type=int, default=100, help='Max windows per symbol')
    parser.add_argument('--max-symbols', type=int, default=None, help='Max symbols')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("KRONOS TOKENIZER BENCHMARK")
    print("(Encode -> Decode reconstruction quality)")
    print("=" * 60)
    
    config = Config()
    engine = BitgetBenchmarkEngine(config)
    
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
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    agg = results.get('aggregate', {})
    print(f"\nAggregate ({agg.get('n_symbols', 0)} symbols, {agg.get('total_windows', 0)} windows):")
    
    if 'future_reconstruction' in agg:
        fr = agg['future_reconstruction']
        print(f"\n  Future Reconstruction (predict window):")
        print(f"    RMSE:        {fr.get('rmse', {}).get('mean', 0):.4f} ± {fr.get('rmse', {}).get('std', 0):.4f}")
        print(f"    MAE:         {fr.get('mae', {}).get('mean', 0):.4f} ± {fr.get('mae', {}).get('std', 0):.4f}")
        print(f"    Correlation: {fr.get('correlation', {}).get('mean', 0):.4f} ± {fr.get('correlation', {}).get('std', 0):.4f}")
    
    if 'returns_directional' in agg:
        rd = agg['returns_directional']
        print(f"\n  Returns Directional Accuracy:")
        print(f"    Direction Accuracy: {rd.get('direction_accuracy', {}).get('mean', 0):.2%} ± {rd.get('direction_accuracy', {}).get('std', 0):.2%}")
        print(f"    Up Accuracy:        {rd.get('up_accuracy', {}).get('mean', 0):.2%}")
        print(f"    Down Accuracy:      {rd.get('down_accuracy', {}).get('mean', 0):.2%}")
        print(f"    Precision (Up):     {rd.get('precision_up', {}).get('mean', 0):.2%}")
        print(f"    Precision (Down):   {rd.get('precision_down', {}).get('mean', 0):.2%}")
    
    # Save results
    output_dir = args.output_dir or os.path.join(config.save_path, 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'benchmark_{timestamp}.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
