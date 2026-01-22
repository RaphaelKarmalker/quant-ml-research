"""
Bitget Inference Pipeline for Kronos Transformer

This script performs inference on the validation set and generates:
1. Return predictions for each symbol
2. Evaluation metrics (RMSE, MAE, directional accuracy, correlation)
3. Backtest-ready signals (long/short based on predicted returns)

Usage:
    python bitget_inference.py                    # Run inference on validation set
    python bitget_inference.py --symbol BTCUSDT   # Run for specific symbol
    python bitget_inference.py --export-signals   # Export trading signals

Author: Kronos ML Team
Date: 2026-01-21
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

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model.kronos import KronosTokenizer, Kronos, calc_time_stamps
from train_bitget import ReturnPredictionHead


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for return predictions.
    
    Args:
        predictions: Array of predicted returns (N, horizon)
        actuals: Array of actual returns (N, horizon)
    
    Returns:
        Dictionary of metrics
    """
    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    actual_flat = actuals.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(pred_flat) | np.isnan(actual_flat))
    pred_flat = pred_flat[valid_mask]
    actual_flat = actual_flat[valid_mask]
    
    if len(pred_flat) == 0:
        return {'error': 'No valid predictions'}
    
    # Basic regression metrics
    mse = np.mean((pred_flat - actual_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred_flat - actual_flat))
    
    # Correlation
    if np.std(pred_flat) > 1e-8 and np.std(actual_flat) > 1e-8:
        correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
    else:
        correlation = 0.0
    
    # Directional accuracy (sign agreement)
    pred_sign = np.sign(pred_flat)
    actual_sign = np.sign(actual_flat)
    direction_accuracy = np.mean(pred_sign == actual_sign)
    
    # Hit rate for non-zero moves
    nonzero_mask = actual_flat != 0
    if np.sum(nonzero_mask) > 0:
        hit_rate = np.mean(pred_sign[nonzero_mask] == actual_sign[nonzero_mask])
    else:
        hit_rate = 0.0
    
    # Information coefficient (IC) - rank correlation
    from scipy import stats
    ic, _ = stats.spearmanr(pred_flat, actual_flat)
    
    # Per-horizon metrics
    horizon_metrics = {}
    for h in range(predictions.shape[1]):
        pred_h = predictions[:, h]
        actual_h = actuals[:, h]
        valid = ~(np.isnan(pred_h) | np.isnan(actual_h))
        if np.sum(valid) > 0:
            horizon_metrics[f'h{h+1}_mae'] = np.mean(np.abs(pred_h[valid] - actual_h[valid]))
            horizon_metrics[f'h{h+1}_dir_acc'] = np.mean(np.sign(pred_h[valid]) == np.sign(actual_h[valid]))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'correlation': correlation,
        'direction_accuracy': direction_accuracy,
        'hit_rate': hit_rate,
        'information_coefficient': ic if not np.isnan(ic) else 0.0,
        'n_samples': len(pred_flat),
        **horizon_metrics
    }


def compute_backtest_metrics(signals: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
    """
    Compute simple backtest metrics for signal-based strategy.
    
    Args:
        signals: Trading signals (-1, 0, 1)
        returns: Actual returns
    
    Returns:
        Dictionary of backtest metrics
    """
    # Strategy returns: signal * actual return
    strategy_returns = signals * returns
    
    # Cumulative returns
    cum_returns = np.cumsum(strategy_returns)
    
    # Basic metrics
    total_return = cum_returns[-1] if len(cum_returns) > 0 else 0
    mean_return = np.mean(strategy_returns)
    std_return = np.std(strategy_returns)
    
    # Sharpe ratio (annualized, assuming hourly data)
    hours_per_year = 24 * 365
    sharpe = (mean_return / (std_return + 1e-8)) * np.sqrt(hours_per_year)
    
    # Max drawdown
    peak = np.maximum.accumulate(cum_returns)
    drawdown = peak - cum_returns
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Win rate
    trades = strategy_returns[signals != 0]
    win_rate = np.mean(trades > 0) if len(trades) > 0 else 0
    
    return {
        'total_return': total_return,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': np.sum(signals != 0)
    }


# =============================================================================
# Inference Engine
# =============================================================================

class BitgetInferenceEngine:
    """
    Inference engine for Kronos model on Bitget data.
    
    Handles:
    - Model loading
    - Sliding window inference
    - Return prediction
    - Metric computation
    - Signal generation
    """
    
    def __init__(self, config: Config = None, device: str = None):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration object (uses default if None)
            device: Device to run on ('cuda', 'cpu', or auto-detect)
        """
        self.config = config or Config()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Load validation data
        self._load_data()
    
    def _load_models(self):
        """Load tokenizer, predictor, and return head."""
        print("Loading models...")
        
        # Tokenizer
        tokenizer_path = self.config.finetuned_tokenizer_path
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.eval().to(self.device)
        print(f"  Tokenizer loaded from: {tokenizer_path}")
        
        # Predictor
        predictor_path = self.config.finetuned_predictor_path
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Predictor not found: {predictor_path}")
        
        self.predictor = Kronos.from_pretrained(predictor_path)
        self.predictor.eval().to(self.device)
        print(f"  Predictor loaded from: {predictor_path}")
        
        # Return head
        return_head_path = os.path.join(
            self.config.save_path, 
            self.config.predictor_save_folder_name,
            'checkpoints', 'return_head.pt'
        )
        
        self.return_head = ReturnPredictionHead(
            d_model=self.config.predictor_arch['d_model'],
            predict_horizon=self.config.predict_window,
            hidden_dim=self.config.predictor_arch['d_model'] // 2,
            dropout=0.0  # No dropout during inference
        )
        
        if os.path.exists(return_head_path):
            state_dict = torch.load(return_head_path, map_location=self.device)
            self.return_head.load_state_dict(state_dict)
            print(f"  Return head loaded from: {return_head_path}")
        else:
            print(f"  [WARN] Return head not found at {return_head_path}, using random init")
        
        self.return_head.eval().to(self.device)
    
    def _load_data(self):
        """Load validation data from pickle."""
        val_path = os.path.join(self.config.dataset_path, 'val_data.pkl')
        
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        
        with open(val_path, 'rb') as f:
            self.val_data = pickle.load(f)
        
        self.symbols = list(self.val_data.keys())
        print(f"Loaded validation data: {len(self.symbols)} symbols")
    
    def _normalize_features(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features per-window with robust scaling for returns.
        
        Returns:
            Tuple of (normalized features, means, stds)
        """
        x_norm = np.zeros_like(x)
        means = np.zeros(x.shape[1])
        stds = np.zeros(x.shape[1])
        
        return_features = [f for f in self.config.feature_list if 'returns' in f.lower()]
        n_return_features = len(return_features)
        
        for i in range(x.shape[1]):
            col = x[:, i]
            
            if i < n_return_features:
                # Robust scaling for returns
                median = np.median(col)
                q75, q25 = np.percentile(col, [75, 25])
                iqr = q75 - q25
                if iqr < 1e-8:
                    iqr = np.std(col) + 1e-8
                x_norm[:, i] = (col - median) / (iqr + 1e-8)
                means[i] = median
                stds[i] = iqr
            else:
                # Standard z-score
                mean = np.mean(col)
                std = np.std(col)
                x_norm[:, i] = (col - mean) / (std + 1e-5)
                means[i] = mean
                stds[i] = std
        
        x_norm = np.clip(x_norm, -self.config.clip, self.config.clip)
        
        return x_norm, means, stds
    
    @torch.no_grad()
    def predict_symbol(self, symbol: str, stride: int = 1) -> pd.DataFrame:
        """
        Run sliding window inference for a single symbol.
        
        Args:
            symbol: Symbol to predict
            stride: Step size for sliding window
        
        Returns:
            DataFrame with predictions and actuals
        """
        if symbol not in self.val_data:
            raise ValueError(f"Symbol {symbol} not in validation data")
        
        df = self.val_data[symbol]
        
        # Get feature columns
        available_features = [f for f in self.config.feature_list if f in df.columns]
        
        lookback = self.config.lookback_window
        predict_horizon = self.config.predict_window
        window = lookback + predict_horizon
        
        if len(df) < window:
            print(f"[WARN] {symbol}: Not enough data ({len(df)} < {window})")
            return pd.DataFrame()
        
        results = []
        
        for start_idx in range(0, len(df) - window + 1, stride):
            end_idx = start_idx + window
            window_df = df.iloc[start_idx:end_idx]
            
            # Extract features
            x = window_df[available_features].values.astype(np.float32)
            
            # Pad missing features
            if len(available_features) < len(self.config.feature_list):
                full_x = np.zeros((x.shape[0], len(self.config.feature_list)), dtype=np.float32)
                for i, f in enumerate(self.config.feature_list):
                    if f in available_features:
                        feat_idx = available_features.index(f)
                        full_x[:, i] = x[:, feat_idx]
                x = full_x
            
            # Handle NaN
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize
            x_norm, _, _ = self._normalize_features(x)
            
            # Get time features
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
            
            # Convert to tensors
            context_x = torch.from_numpy(x_norm[:lookback]).unsqueeze(0).to(self.device)
            context_stamp = torch.from_numpy(x_stamp[:lookback]).unsqueeze(0).to(self.device)
            
            # Tokenize
            token_seq_0, token_seq_1 = self.tokenizer.encode(context_x, half=True)
            
            # Get embeddings
            embeddings = self.predictor.embedding([token_seq_0, token_seq_1])
            
            # Predict returns
            return_pred = self.return_head(embeddings)
            return_pred = return_pred.cpu().numpy().flatten()
            
            # Get actual future returns (target index = 0 for returns_1h)
            target_idx = 0
            if 'returns_1h' in available_features:
                target_idx = available_features.index('returns_1h')
            actual_returns = x[lookback:lookback+predict_horizon, target_idx]
            
            # Store result
            result = {
                'timestamp': window_df.index[lookback],
                'symbol': symbol,
            }
            
            for h in range(predict_horizon):
                result[f'pred_h{h+1}'] = return_pred[h] if h < len(return_pred) else np.nan
                result[f'actual_h{h+1}'] = actual_returns[h] if h < len(actual_returns) else np.nan
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def run_inference(self, symbols: List[str] = None, stride: int = 1) -> Tuple[pd.DataFrame, Dict]:
        """
        Run inference on multiple symbols.
        
        Args:
            symbols: List of symbols (None = all)
            stride: Step size for sliding window
        
        Returns:
            Tuple of (predictions DataFrame, metrics dict)
        """
        if symbols is None:
            symbols = self.symbols
        
        all_results = []
        symbol_metrics = {}
        
        print(f"\nRunning inference on {len(symbols)} symbols...")
        
        for symbol in tqdm(symbols, desc="Inference"):
            try:
                df = self.predict_symbol(symbol, stride=stride)
                
                if len(df) > 0:
                    all_results.append(df)
                    
                    # Compute symbol-level metrics
                    pred_cols = [c for c in df.columns if c.startswith('pred_')]
                    actual_cols = [c for c in df.columns if c.startswith('actual_')]
                    
                    predictions = df[pred_cols].values
                    actuals = df[actual_cols].values
                    
                    metrics = compute_metrics(predictions, actuals)
                    symbol_metrics[symbol] = metrics
                    
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
        
        # Combine results
        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)
        else:
            results_df = pd.DataFrame()
        
        # Compute aggregate metrics
        if len(results_df) > 0:
            pred_cols = [c for c in results_df.columns if c.startswith('pred_')]
            actual_cols = [c for c in results_df.columns if c.startswith('actual_')]
            
            all_predictions = results_df[pred_cols].values
            all_actuals = results_df[actual_cols].values
            
            aggregate_metrics = compute_metrics(all_predictions, all_actuals)
        else:
            aggregate_metrics = {}
        
        return results_df, {
            'aggregate': aggregate_metrics,
            'by_symbol': symbol_metrics
        }
    
    def generate_signals(self, predictions_df: pd.DataFrame, 
                         threshold: float = 0.0) -> pd.DataFrame:
        """
        Generate trading signals from predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            threshold: Minimum return prediction to generate signal
        
        Returns:
            DataFrame with signals
        """
        signals_df = predictions_df.copy()
        
        # Use h1 (1-hour ahead) predictions for signals
        pred_col = 'pred_h1'
        
        if pred_col in signals_df.columns:
            signals_df['signal'] = np.where(
                signals_df[pred_col] > threshold, 1,
                np.where(signals_df[pred_col] < -threshold, -1, 0)
            )
            
            # Confidence based on prediction magnitude
            signals_df['confidence'] = np.abs(signals_df[pred_col])
        
        return signals_df


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Kronos inference on Bitget data')
    parser.add_argument('--symbol', type=str, default=None, help='Specific symbol to predict')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window stride')
    parser.add_argument('--export-signals', action='store_true', help='Export trading signals')
    parser.add_argument('--output-dir', type=str, default='./data/inference_outputs', 
                        help='Output directory')
    args = parser.parse_args()
    
    print("="*60)
    print("BITGET INFERENCE PIPELINE")
    print("="*60)
    
    # Initialize engine
    engine = BitgetInferenceEngine()
    
    # Run inference
    symbols = [args.symbol] if args.symbol else None
    predictions_df, metrics = engine.run_inference(symbols=symbols, stride=args.stride)
    
    if len(predictions_df) == 0:
        print("No predictions generated!")
        return
    
    # Print metrics
    print("\n" + "="*60)
    print("AGGREGATE METRICS")
    print("="*60)
    for k, v in metrics['aggregate'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    pred_path = os.path.join(args.output_dir, f'predictions_{timestamp}.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to: {pred_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(metrics), f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    # Export signals if requested
    if args.export_signals:
        signals_df = engine.generate_signals(predictions_df)
        signals_path = os.path.join(args.output_dir, f'signals_{timestamp}.csv')
        signals_df.to_csv(signals_path, index=False)
        print(f"Signals saved to: {signals_path}")
        
        # Compute simple backtest metrics
        if 'signal' in signals_df.columns and 'actual_h1' in signals_df.columns:
            backtest_metrics = compute_backtest_metrics(
                signals_df['signal'].values,
                signals_df['actual_h1'].values
            )
            print("\n" + "="*60)
            print("BACKTEST METRICS (Simple Signal Strategy)")
            print("="*60)
            for k, v in backtest_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
