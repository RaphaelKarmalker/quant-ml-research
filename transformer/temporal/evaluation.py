"""
Evaluation Module for Temporal Transformer

This module provides comprehensive evaluation metrics for:
1. Regression tasks (log returns)
2. Multi-class classification (direction: -1, 0, +1)
3. Binary classification (strong pump/dump)

Includes:
- Per-horizon breakdowns
- Per-instrument analysis
- Confidence calibration
- Trading-relevant metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import warnings


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Mean Absolute Error."""
    if mask is not None:
        valid = mask.astype(bool).flatten()
        y_true = y_true.flatten()[valid]
        y_pred = y_pred.flatten()[valid]
    
    if len(y_true) == 0:
        return float('nan')
    
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute Root Mean Squared Error."""
    if mask is not None:
        valid = mask.astype(bool).flatten()
        y_true = y_true.flatten()[valid]
        y_pred = y_pred.flatten()[valid]
    
    if len(y_true) == 0:
        return float('nan')
    
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error.
    
    SMAPE = 100 * mean(|y_true - y_pred| / (|y_true| + |y_pred| + eps))
    """
    if mask is not None:
        valid = mask.astype(bool).flatten()
        y_true = y_true.flatten()[valid]
        y_pred = y_pred.flatten()[valid]
    
    if len(y_true) == 0:
        return float('nan')
    
    eps = 1e-8
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    smape = 100 * np.mean(np.abs(y_true - y_pred) / denominator)
    
    return float(smape)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Only computed where y_true != 0.
    """
    if mask is not None:
        valid = mask.astype(bool).flatten()
        y_true = y_true.flatten()[valid]
        y_pred = y_pred.flatten()[valid]
    
    # Filter out zeros
    nonzero = np.abs(y_true) > 1e-8
    if nonzero.sum() == 0:
        return float('nan')
    
    y_true = y_true[nonzero]
    y_pred = y_pred[nonzero]
    
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    
    return float(mape)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Compute R-squared (coefficient of determination)."""
    if mask is not None:
        valid = mask.astype(bool).flatten()
        y_true = y_true.flatten()[valid]
        y_pred = y_pred.flatten()[valid]
    
    if len(y_true) < 2:
        return float('nan')
    
    return float(r2_score(y_true, y_pred))


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute directional accuracy for regression predictions.
    
    Measures how often the predicted direction (sign) matches actual direction.
    Important for trading: even if magnitude is wrong, direction matters.
    """
    if mask is not None:
        valid = mask.astype(bool).flatten()
        y_true = y_true.flatten()[valid]
        y_pred = y_pred.flatten()[valid]
    
    if len(y_true) == 0:
        return float('nan')
    
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    
    return float(np.mean(true_direction == pred_direction))


def compute_regression_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all regression metrics.
    
    Args:
        y_pred: Predictions (batch, n_targets) or (batch,)
        y_true: Targets (batch, n_targets) or (batch,)
        mask: Valid mask (batch, n_targets) or (batch,)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "mae": compute_mae(y_true, y_pred, mask),
        "rmse": compute_rmse(y_true, y_pred, mask),
        "smape": compute_smape(y_true, y_pred, mask),
        "r2": compute_r2(y_true, y_pred, mask),
        "directional_accuracy": compute_directional_accuracy(y_true, y_pred, mask),
    }
    
    return metrics


# =============================================================================
# CLASSIFICATION METRICS (Direction: 3-class)
# =============================================================================

def compute_direction_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute metrics for direction classification (3-class: down, neutral, up).
    
    Args:
        logits: (batch, n_horizons, 3) logits
        targets: (batch, n_horizons) targets in {-1, 0, 1}
        mask: (batch, n_horizons) valid mask
        
    Returns:
        Dictionary of metrics
    """
    # Handle dimensions
    if logits.ndim == 3:
        # Flatten across horizons
        batch, n_horizons, n_classes = logits.shape
        logits = logits.reshape(-1, n_classes)
        targets = targets.reshape(-1)
        if mask is not None:
            mask = mask.reshape(-1)
    
    # Apply mask
    if mask is not None:
        valid = mask.astype(bool)
        logits = logits[valid]
        targets = targets[valid]
    
    if len(targets) == 0:
        return {"accuracy": float('nan'), "f1_macro": float('nan')}
    
    # Convert targets from {-1, 0, 1} to {0, 1, 2}
    targets_idx = (targets + 1).astype(int)
    
    # Get predictions
    preds = np.argmax(logits, axis=-1)
    
    metrics = {
        "accuracy": float(accuracy_score(targets_idx, preds)),
        "f1_macro": float(f1_score(targets_idx, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(targets_idx, preds, average="weighted", zero_division=0)),
    }
    
    # Per-class metrics
    for i, label in enumerate(["down", "neutral", "up"]):
        binary_true = (targets_idx == i).astype(int)
        binary_pred = (preds == i).astype(int)
        
        metrics[f"precision_{label}"] = float(precision_score(binary_true, binary_pred, zero_division=0))
        metrics[f"recall_{label}"] = float(recall_score(binary_true, binary_pred, zero_division=0))
        metrics[f"f1_{label}"] = float(f1_score(binary_true, binary_pred, zero_division=0))
    
    return metrics


# =============================================================================
# BINARY CLASSIFICATION METRICS (Pump/Dump)
# =============================================================================

def compute_binary_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics for binary classification (pump/dump detection).
    
    Args:
        logits: (batch, n_targets) raw logits
        targets: (batch, n_targets) binary targets {0, 1}
        mask: (batch, n_targets) valid mask
        threshold: Decision threshold for predictions
        
    Returns:
        Dictionary of metrics
    """
    # Flatten
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    if mask is not None:
        mask = mask.reshape(-1)
        valid = mask.astype(bool)
        logits = logits[valid]
        targets = targets[valid]
    
    if len(targets) == 0:
        return {"accuracy": float('nan'), "f1": float('nan'), "auc": float('nan')}
    
    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    # Get predictions
    preds = (probs > threshold).astype(int)
    targets = targets.astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
    }
    
    # AUC (only if both classes present)
    if len(np.unique(targets)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(targets, probs))
        except:
            metrics["auc"] = float('nan')
    else:
        metrics["auc"] = float('nan')
    
    # Class balance
    metrics["positive_rate"] = float(np.mean(targets))
    metrics["pred_positive_rate"] = float(np.mean(preds))
    
    return metrics


# =============================================================================
# PER-HORIZON METRICS
# =============================================================================

def compute_per_horizon_metrics(
    predictions: Dict[str, np.ndarray],
    targets: np.ndarray,
    target_mask: np.ndarray,
    regression_indices: List[int],
    direction_indices: List[int],
    binary_indices: List[int],
    horizons: List[str] = ["15min", "1h", "4h", "12h"],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by prediction horizon.

    Args:
        predictions: Dict with regression, direction_logits, binary_logits
        targets: (batch, n_targets) all targets
        target_mask: (batch, n_targets) valid mask
        regression_indices: Indices of regression targets in targets array
        direction_indices: Indices of direction targets in targets array
        binary_indices: Indices of binary targets in targets array
        horizons: List of horizon names

    Returns:
        Nested dict: {horizon: {metric: value}}
    """
    results = {}

    for i, horizon in enumerate(horizons):
        results[horizon] = {}

        # Regression (one per horizon)
        if "regression" in predictions and i < predictions["regression"].shape[-1]:
            reg_pred = predictions["regression"][:, i]

            # Use correct index from regression_indices
            if i < len(regression_indices):
                reg_idx = regression_indices[i]
                reg_true = targets[:, reg_idx]
                reg_mask = target_mask[:, reg_idx]

                reg_metrics = compute_regression_metrics(reg_pred, reg_true, reg_mask)
                for k, v in reg_metrics.items():
                    results[horizon][f"regression_{k}"] = v

        # Direction (one per horizon)
        if "direction_logits" in predictions:
            dir_logits = predictions["direction_logits"]
            n_horizons = dir_logits.shape[1] if dir_logits.ndim == 3 else 1

            if i < n_horizons and i < len(direction_indices):
                if dir_logits.ndim == 3:
                    horizon_logits = dir_logits[:, i, :]
                else:
                    horizon_logits = dir_logits

                # Use correct index from direction_indices
                dir_idx = direction_indices[i]
                dir_true = targets[:, dir_idx]
                dir_mask = target_mask[:, dir_idx]

                dir_metrics = compute_direction_metrics(
                    horizon_logits[:, np.newaxis, :],
                    dir_true[:, np.newaxis],
                    dir_mask[:, np.newaxis],
                )
                for k, v in dir_metrics.items():
                    results[horizon][f"direction_{k}"] = v

        # Binary (pump/dump per horizon)
        if "binary_logits" in predictions:
            # Each horizon has 2 binary targets (pump, dump)
            pump_idx_in_binary = i * 2
            dump_idx_in_binary = i * 2 + 1

            if pump_idx_in_binary < len(binary_indices):
                pump_idx = binary_indices[pump_idx_in_binary]
                pump_pred = predictions["binary_logits"][:, pump_idx_in_binary]
                pump_true = targets[:, pump_idx]
                pump_mask = target_mask[:, pump_idx]

                pump_metrics = compute_binary_metrics(
                    pump_pred[:, np.newaxis],
                    pump_true[:, np.newaxis],
                    pump_mask[:, np.newaxis],
                )
                for k, v in pump_metrics.items():
                    results[horizon][f"pump_{k}"] = v

            if dump_idx_in_binary < len(binary_indices):
                dump_idx = binary_indices[dump_idx_in_binary]
                dump_pred = predictions["binary_logits"][:, dump_idx_in_binary]
                dump_true = targets[:, dump_idx]
                dump_mask = target_mask[:, dump_idx]

                dump_metrics = compute_binary_metrics(
                    dump_pred[:, np.newaxis],
                    dump_true[:, np.newaxis],
                    dump_mask[:, np.newaxis],
                )
                for k, v in dump_metrics.items():
                    results[horizon][f"dump_{k}"] = v

    return results


# =============================================================================
# TRADING-SPECIFIC METRICS
# =============================================================================

def compute_trading_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
    volatility: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute trading-relevant metrics for regression predictions.

    Args:
        y_pred: Predicted log returns
        y_true: Actual log returns
        mask: Valid mask
        volatility: Predicted volatility (optional)
        confidence: Prediction confidence scores (optional)

    Returns:
        Trading metrics dictionary
    """
    if mask is not None:
        valid = mask.astype(bool).flatten()
        y_true = y_true.flatten()[valid]
        y_pred = y_pred.flatten()[valid]
        if volatility is not None:
            volatility = volatility.flatten()[valid]
        if confidence is not None:
            confidence = confidence.flatten()[valid]

    if len(y_true) == 0:
        return {}

    metrics = {}

    # 1. Directional accuracy
    metrics["directional_accuracy"] = compute_directional_accuracy(
        y_true.reshape(-1), y_pred.reshape(-1)
    )

    # 2. Profitable trades (if we traded in predicted direction)
    pred_direction = np.sign(y_pred)
    actual_return = y_true * pred_direction  # PnL if we follow prediction

    metrics["profitable_trade_rate"] = float(np.mean(actual_return > 0))
    metrics["mean_trade_return"] = float(np.mean(actual_return))

    # 3. Sharpe ratio (annualized, assuming 15min intervals)
    # NOTE: This assumes 15min data (35040 periods/year).
    # For other horizons, this should be adjusted in per-horizon metrics.
    # 15min: 4*24*365 = 35040, 1h: 24*365 = 8760, 4h: 6*365 = 2190, 12h: 2*365 = 730
    if np.std(actual_return) > 0:
        sharpe = np.mean(actual_return) / np.std(actual_return)
        metrics["sharpe_ratio"] = float(sharpe * np.sqrt(35040))  # Annualized for 15min
    else:
        metrics["sharpe_ratio"] = 0.0

    # 4. Sortino ratio (downside deviation only)
    downside_returns = actual_return[actual_return < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino = np.mean(actual_return) / np.std(downside_returns)
        metrics["sortino_ratio"] = float(sortino * np.sqrt(35040))  # Annualized for 15min
    else:
        metrics["sortino_ratio"] = 0.0

    # 5. Win rate and avg win/loss
    wins = actual_return > 0
    losses = actual_return < 0
    metrics["win_rate"] = float(np.mean(wins))
    if wins.sum() > 0:
        metrics["avg_win"] = float(np.mean(actual_return[wins]))
    if losses.sum() > 0:
        metrics["avg_loss"] = float(np.mean(actual_return[losses]))

    # 6. Profit factor (gross profit / gross loss)
    total_profit = np.sum(actual_return[actual_return > 0])
    total_loss = np.abs(np.sum(actual_return[actual_return < 0]))
    if total_loss > 0:
        metrics["profit_factor"] = float(total_profit / total_loss)
    else:
        metrics["profit_factor"] = float('inf') if total_profit > 0 else 0.0

    # 7. Maximum drawdown (consecutive losses)
    cumulative_returns = np.cumsum(actual_return)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = running_max - cumulative_returns
    metrics["max_drawdown"] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # 8. Calmar ratio (return / max drawdown)
    if metrics["max_drawdown"] > 0:
        annualized_return = np.mean(actual_return) * 35040  # Assumes 15min data
        metrics["calmar_ratio"] = float(annualized_return / metrics["max_drawdown"])
    else:
        metrics["calmar_ratio"] = 0.0

    # 9. Hit rate for large moves (> 1% return)
    large_move_mask = np.abs(y_true) > 0.01
    if large_move_mask.sum() > 0:
        large_true = y_true[large_move_mask]
        large_pred = y_pred[large_move_mask]
        metrics["large_move_directional_accuracy"] = float(
            np.mean(np.sign(large_true) == np.sign(large_pred))
        )

    # 10. Confidence calibration (if confidence scores provided)
    if confidence is not None:
        # High confidence predictions should be more accurate
        high_conf_mask = confidence > np.median(confidence)
        if high_conf_mask.sum() > 10:
            high_conf_acc = np.mean(
                np.sign(y_true[high_conf_mask]) == np.sign(y_pred[high_conf_mask])
            )
            metrics["high_confidence_accuracy"] = float(high_conf_acc)

        # Confidence should correlate with accuracy
        correct = (np.sign(y_true) == np.sign(y_pred)).astype(float)
        if len(correct) > 1 and np.std(confidence) > 0:
            corr = np.corrcoef(confidence, correct)[0, 1]
            metrics["confidence_calibration_corr"] = float(corr)

    # 11. Volatility prediction quality (if volatility provided)
    if volatility is not None:
        # Compare predicted volatility to realized volatility
        realized_vol = np.abs(y_true)
        vol_mae = np.mean(np.abs(volatility - realized_vol))
        metrics["volatility_mae"] = float(vol_mae)

        # Correlation between predicted and realized volatility
        if len(realized_vol) > 1 and np.std(volatility) > 0:
            vol_corr = np.corrcoef(volatility, realized_vol)[0, 1]
            metrics["volatility_correlation"] = float(vol_corr)

    # 12. Prediction confidence calibration by bins
    pred_abs = np.abs(y_pred)
    percentiles = [50, 75, 90]
    thresholds = np.percentile(pred_abs, percentiles)

    for i, pct in enumerate(percentiles):
        mask_high = pred_abs >= thresholds[i]
        if mask_high.sum() > 10:
            high_acc = np.mean(
                np.sign(y_true[mask_high]) == np.sign(y_pred[mask_high])
            )
            metrics[f"accuracy_top_{100-pct}pct"] = float(high_acc)

    return metrics


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

@dataclass
class EvaluationResult:
    """Container for comprehensive evaluation results."""
    
    regression_metrics: Dict[str, float]
    direction_metrics: Dict[str, float]
    binary_metrics: Dict[str, float]
    per_horizon_metrics: Dict[str, Dict[str, float]]
    trading_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, float]:
        """Flatten to single dict for logging."""
        result = {}
        
        for k, v in self.regression_metrics.items():
            result[f"regression/{k}"] = v
        
        for k, v in self.direction_metrics.items():
            result[f"direction/{k}"] = v
        
        for k, v in self.binary_metrics.items():
            result[f"binary/{k}"] = v
        
        for horizon, metrics in self.per_horizon_metrics.items():
            for k, v in metrics.items():
                result[f"horizon/{horizon}/{k}"] = v
        
        for k, v in self.trading_metrics.items():
            result[f"trading/{k}"] = v
        
        return result
    
    def print_summary(self):
        """Print formatted evaluation summary."""
        print("\n" + "=" * 80)
        print(" EVALUATION RESULTS")
        print("=" * 80)
        
        print("\n📈 REGRESSION METRICS:")
        for k, v in self.regression_metrics.items():
            print(f"   {k}: {v:.4f}")
        
        print("\n🎯 DIRECTION METRICS:")
        for k, v in self.direction_metrics.items():
            print(f"   {k}: {v:.4f}")
        
        print("\n🚀 BINARY METRICS:")
        for k, v in self.binary_metrics.items():
            print(f"   {k}: {v:.4f}")
        
        print("\n💹 TRADING METRICS:")
        for k, v in self.trading_metrics.items():
            print(f"   {k}: {v:.4f}")
        
        print("\n⏰ PER-HORIZON BREAKDOWN:")
        for horizon, metrics in self.per_horizon_metrics.items():
            print(f"\n   {horizon}:")
            for k, v in metrics.items():
                print(f"      {k}: {v:.4f}")
        
        print("=" * 80)


def compute_all_metrics(
    predictions: Dict[str, np.ndarray],
    targets: np.ndarray,
    target_mask: np.ndarray,
    regression_indices: List[int],
    direction_indices: List[int],
    binary_indices: List[int],
    horizons: List[str] = ["15min", "1h", "4h", "12h"],
) -> EvaluationResult:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Dict with regression, direction_logits, binary_logits
        targets: (batch, n_targets) all targets
        target_mask: (batch, n_targets) valid mask
        regression_indices: Indices of regression targets
        direction_indices: Indices of direction targets
        binary_indices: Indices of binary targets
        horizons: List of horizon names
        
    Returns:
        EvaluationResult with all metrics
    """
    # Regression metrics
    regression_metrics = {}
    if "regression" in predictions and len(regression_indices) > 0:
        reg_pred = predictions["regression"]
        reg_true = targets[:, regression_indices]
        reg_mask = target_mask[:, regression_indices]
        
        regression_metrics = compute_regression_metrics(reg_pred, reg_true, reg_mask)
    
    # Direction metrics
    direction_metrics = {}
    if "direction_logits" in predictions and len(direction_indices) > 0:
        dir_logits = predictions["direction_logits"]
        dir_true = targets[:, direction_indices]
        dir_mask = target_mask[:, direction_indices]
        
        direction_metrics = compute_direction_metrics(dir_logits, dir_true, dir_mask)
    
    # Binary metrics
    binary_metrics = {}
    if "binary_logits" in predictions and len(binary_indices) > 0:
        bin_logits = predictions["binary_logits"]
        bin_true = targets[:, binary_indices]
        bin_mask = target_mask[:, binary_indices]
        
        binary_metrics = compute_binary_metrics(bin_logits, bin_true, bin_mask)
    
    # Per-horizon metrics
    per_horizon_metrics = compute_per_horizon_metrics(
        predictions, targets, target_mask,
        regression_indices, direction_indices, binary_indices,
        horizons
    )
    
    # Trading metrics
    trading_metrics = {}
    if "regression" in predictions and len(regression_indices) > 0:
        reg_pred = predictions["regression"]
        reg_true = targets[:, regression_indices]
        reg_mask = target_mask[:, regression_indices]

        # Extract volatility and confidence if available
        volatility = predictions.get("volatility", None)
        confidence = predictions.get("confidence", None)

        trading_metrics = compute_trading_metrics(
            reg_pred, reg_true, reg_mask,
            volatility=volatility,
            confidence=confidence
        )
    
    return EvaluationResult(
        regression_metrics=regression_metrics,
        direction_metrics=direction_metrics,
        binary_metrics=binary_metrics,
        per_horizon_metrics=per_horizon_metrics,
        trading_metrics=trading_metrics,
    )


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_confusion_matrix_data(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    labels: List[str] = ["Down", "Neutral", "Up"],
) -> np.ndarray:
    """
    Create confusion matrix for direction classification.
    
    Args:
        logits: (batch, n_horizons, 3) or (batch, 3)
        targets: (batch, n_horizons) or (batch,) in {-1, 0, 1}
        mask: Optional valid mask
        labels: Class labels
        
    Returns:
        Confusion matrix (3, 3)
    """
    # Flatten
    if logits.ndim == 3:
        logits = logits.reshape(-1, 3)
        targets = targets.reshape(-1)
        if mask is not None:
            mask = mask.reshape(-1)
    
    # Apply mask
    if mask is not None:
        valid = mask.astype(bool)
        logits = logits[valid]
        targets = targets[valid]
    
    # Convert to class indices
    targets_idx = (targets + 1).astype(int)
    preds = np.argmax(logits, axis=-1)
    
    return confusion_matrix(targets_idx, preds, labels=[0, 1, 2])


def create_prediction_distribution_data(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create histogram data for prediction vs actual distribution.
    
    Returns:
        Tuple of (pred_hist, target_hist, bin_edges)
    """
    if mask is not None:
        valid = mask.astype(bool).flatten()
        predictions = predictions.flatten()[valid]
        targets = targets.flatten()[valid]
    
    # Determine bin edges
    all_values = np.concatenate([predictions, targets])
    min_val, max_val = np.percentile(all_values, [1, 99])
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    
    pred_hist, _ = np.histogram(predictions, bins=bin_edges)
    target_hist, _ = np.histogram(targets, bins=bin_edges)
    
    return pred_hist, target_hist, bin_edges


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test metrics computation
    np.random.seed(42)
    
    batch_size = 100
    n_horizons = 4
    
    # Simulate predictions
    predictions = {
        "regression": np.random.randn(batch_size, n_horizons) * 0.02,
        "direction_logits": np.random.randn(batch_size, n_horizons, 3),
        "binary_logits": np.random.randn(batch_size, n_horizons * 2),  # pump + dump
    }
    
    # Simulate targets
    n_targets = n_horizons + n_horizons + n_horizons * 2  # reg + dir + binary
    targets = np.random.randn(batch_size, n_targets) * 0.02
    targets[:, n_horizons:2*n_horizons] = np.random.choice([-1, 0, 1], size=(batch_size, n_horizons))
    targets[:, 2*n_horizons:] = np.random.binomial(1, 0.1, size=(batch_size, n_horizons * 2))
    
    # Mask (mostly valid)
    target_mask = np.random.binomial(1, 0.9, size=(batch_size, n_targets)).astype(bool)
    
    # Compute metrics
    result = compute_all_metrics(
        predictions=predictions,
        targets=targets,
        target_mask=target_mask,
        regression_indices=list(range(n_horizons)),
        direction_indices=list(range(n_horizons, 2*n_horizons)),
        binary_indices=list(range(2*n_horizons, n_targets)),
        horizons=["15min", "1h", "4h", "12h"],
    )
    
    result.print_summary()
