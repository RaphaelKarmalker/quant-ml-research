"""
Training Pipeline for Temporal Transformer

This module implements:
1. PyTorch Lightning training module
2. Learning rate scheduling with warmup
3. Multi-task loss handling
4. Checkpointing and logging with WandB
5. Gradient accumulation and clipping

Uses evaluation.py for metrics computation.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Optional, Any, List
from pathlib import Path
import warnings
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config, get_config
from model import TemporalTransformer, MultiTaskLoss, create_model
from data_loading import CryptoDataModule, create_data_module
from evaluation import (
    compute_regression_metrics,
    compute_direction_metrics,
    compute_binary_metrics,
    compute_all_metrics,
)


# =============================================================================
# LIGHTNING MODULE
# =============================================================================

class TemporalTransformerLightning(pl.LightningModule):
    """
    PyTorch Lightning module wrapping the Temporal Transformer.
    
    Handles:
    - Training, validation, test steps
    - Loss computation
    - Metrics logging
    - Optimizer and scheduler configuration
    """
    
    def __init__(
        self,
        config: Config,
        feature_dim: Optional[int] = None,
        instrument_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        
        self.config = config
        self.instrument_ids = instrument_ids
        
        # Override input dim if provided (from actual data)
        if feature_dim is not None:
            config.features._actual_feature_dim = feature_dim
        
        # Create model
        self.model = create_model(config)
        
        # Create loss function
        self.loss_fn = MultiTaskLoss(config)
        
        # Metrics accumulators for epoch-level logging
        self.train_losses = []
        self.val_losses = []
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        return self.model(x, mask)
    
    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Shared step for train/val/test.
        
        Args:
            batch: Batch from dataloader
            stage: 'train', 'val', or 'test'
            
        Returns:
            Dict with loss and predictions
        """
        features = batch["features"]
        targets = batch["targets"]
        feature_mask = batch["feature_mask"]
        target_mask = batch["target_mask"]
        
        # Forward pass
        predictions = self.model(features, feature_mask)
        
        # Compute loss
        total_loss, loss_dict = self.loss_fn(predictions, targets, target_mask)
        
        # Store for metrics computation
        result = {
            "loss": total_loss,
            "predictions": predictions,
            "targets": targets,
            "target_mask": target_mask,
            **{f"{stage}_{k}": v for k, v in loss_dict.items()},
        }
        
        return result
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        result = self._shared_step(batch, "train")
        
        # Log losses to WandB
        self.log("train/loss", result["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/regression_loss", result.get("train_regression_loss", 0), on_step=True, on_epoch=True)
        self.log("train/direction_loss", result.get("train_direction_loss", 0), on_step=True, on_epoch=True)
        self.log("train/binary_loss", result.get("train_binary_loss", 0), on_step=True, on_epoch=True)
        
        # Log learning rate
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log("train/learning_rate", current_lr, on_step=True)
        
        # Log gradient norm every N steps
        if batch_idx % 50 == 0:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.log("train/grad_norm", total_norm, on_step=True)
        
        return result["loss"]
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        result = self._shared_step(batch, "val")
        
        # Log losses
        self.log("val/loss", result["loss"], prog_bar=True, on_epoch=True)
        self.log("val/regression_loss", result.get("val_regression_loss", 0), on_epoch=True)
        self.log("val/direction_loss", result.get("val_direction_loss", 0), on_epoch=True)
        self.log("val/binary_loss", result.get("val_binary_loss", 0), on_epoch=True)
        
        # Store for epoch-end metrics
        self.val_losses.append({
            "predictions": {k: v.detach().cpu() for k, v in result["predictions"].items() if isinstance(v, torch.Tensor)},
            "targets": result["targets"].detach().cpu(),
            "target_mask": result["target_mask"].detach().cpu(),
            "instrument_idx": batch["instrument_idx"].detach().cpu(),
            "time_idx_end": batch["time_idx_end"].detach().cpu(),
        })
        
        return result
    
    def on_validation_epoch_end(self):
        """Compute and log epoch-level metrics."""
        if not self.val_losses:
            return
        
        # Aggregate predictions and targets
        all_regression_preds = []
        all_direction_preds = []
        all_binary_preds = []
        all_targets = []
        all_masks = []
        all_inst_idx = []
        all_time_idx = []
        
        for batch_data in self.val_losses:
            preds = batch_data["predictions"]
            if "regression" in preds:
                all_regression_preds.append(preds["regression"])
            if "direction_logits" in preds:
                all_direction_preds.append(preds["direction_logits"])
            if "binary_logits" in preds:
                all_binary_preds.append(preds["binary_logits"])
            all_targets.append(batch_data["targets"])
            all_masks.append(batch_data["target_mask"])
            all_inst_idx.append(batch_data["instrument_idx"])
            all_time_idx.append(batch_data["time_idx_end"])
        
        # Concatenate
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0)
        inst_idx = torch.cat(all_inst_idx, dim=0)
        time_idx = torch.cat(all_time_idx, dim=0)
        
        # Compute metrics using evaluation module
        try:
            # Get target indices
            target_indices = self.loss_fn.regression_indices
            
            # Regression metrics
            if all_regression_preds:
                reg_preds = torch.cat(all_regression_preds, dim=0)
                reg_targets = targets[:, target_indices]
                reg_masks = masks[:, target_indices]

                reg_metrics = compute_regression_metrics(
                    reg_preds.numpy(),
                    reg_targets.numpy(),
                    reg_masks.numpy(),
                )

                for name, value in reg_metrics.items():
                    self.log(f"val/regression/{name}", value, on_epoch=True)

                # Log volatility and confidence metrics if available
                all_volatility = [b["predictions"].get("volatility") for b in self.val_losses if "volatility" in b["predictions"]]
                all_confidence = [b["predictions"].get("confidence") for b in self.val_losses if "confidence" in b["predictions"]]

                if all_volatility and all_volatility[0] is not None:
                    volatility = torch.cat(all_volatility, dim=0).numpy()
                    realized_vol = np.abs(reg_targets.numpy())
                    vol_mae = np.mean(np.abs(volatility - realized_vol)[reg_masks.numpy()])
                    self.log("val/volatility/mae", vol_mae, on_epoch=True)

                if all_confidence and all_confidence[0] is not None:
                    confidence = torch.cat(all_confidence, dim=0).numpy()
                    mean_confidence = np.mean(confidence[reg_masks.numpy()])
                    self.log("val/confidence/mean", mean_confidence, on_epoch=True)
            
            # Direction metrics
            if all_direction_preds:
                dir_preds = torch.cat(all_direction_preds, dim=0)
                dir_targets = targets[:, self.loss_fn.direction_indices]
                dir_masks = masks[:, self.loss_fn.direction_indices]
                
                dir_metrics = compute_direction_metrics(
                    dir_preds.numpy(),
                    dir_targets.numpy(),
                    dir_masks.numpy(),
                )
                
                for name, value in dir_metrics.items():
                    self.log(f"val/direction/{name}", value, on_epoch=True)
            
            # Binary metrics
            if all_binary_preds:
                bin_preds = torch.cat(all_binary_preds, dim=0)
                bin_targets = targets[:, self.loss_fn.binary_indices]
                bin_masks = masks[:, self.loss_fn.binary_indices]
                
                bin_metrics = compute_binary_metrics(
                    bin_preds.numpy(),
                    bin_targets.numpy(),
                    bin_masks.numpy(),
                )
                
                for name, value in bin_metrics.items():
                    self.log(f"val/binary/{name}", value, on_epoch=True)
        
        except Exception as e:
            warnings.warn(f"Error computing validation metrics: {e}")
        
        # Plot sliding windows for a quick visual check
        try:
            if all_regression_preds:
                reg_preds = torch.cat(all_regression_preds, dim=0)
                self._log_validation_windows(
                    reg_preds=reg_preds,
                    targets=targets,
                    target_mask=masks,
                    instrument_idx=inst_idx,
                    time_idx_end=time_idx,
                )
        except Exception as e:
            warnings.warn(f"Error creating validation plots: {e}")
        
        # Clear accumulated data
        self.val_losses = []

    @staticmethod
    def _select_window_starts(
        n_points: int,
        window_size: int,
        num_windows: int,
        min_stride: int,
    ) -> List[int]:
        if n_points < window_size or num_windows <= 0:
            return []
        
        max_start = n_points - window_size
        if num_windows == 1:
            return [max_start // 2]
        
        candidates = np.linspace(0, max_start, num_windows).round().astype(int)
        starts: List[int] = []
        last = -10**9
        for s in candidates:
            if s - last >= min_stride:
                starts.append(int(s))
                last = int(s)
        
        if len(starts) < num_windows:
            s = 0
            while len(starts) < num_windows and s <= max_start:
                if all(abs(s - x) >= min_stride for x in starts):
                    starts.append(int(s))
                s += max(1, min_stride)
        
        return sorted(starts)[:num_windows]

    def _log_validation_windows(
        self,
        reg_preds: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        instrument_idx: torch.Tensor,
        time_idx_end: torch.Tensor,
    ) -> None:
        cfg = self.config.training
        if not cfg.plot_val_windows:
            return
        
        if self.instrument_ids is None:
            warnings.warn("No instrument_ids provided; skipping validation plots.")
            return
        
        regression_cols = self.config.targets.get_regression_targets()
        target_col = f"target_log_return_{cfg.plot_val_horizon}"
        if target_col not in regression_cols:
            warnings.warn(f"Requested horizon {cfg.plot_val_horizon} not in regression targets.")
            return
        
        reg_output_idx = regression_cols.index(target_col)
        if reg_output_idx >= len(self.loss_fn.regression_indices):
            warnings.warn("Regression output index out of range; skipping plots.")
            return
        
        target_idx = self.loss_fn.regression_indices[reg_output_idx]
        
        pred = reg_preds[:, reg_output_idx].detach().cpu().numpy()
        true = targets[:, target_idx].detach().cpu().numpy()
        mask = target_mask[:, target_idx].detach().cpu().numpy().astype(bool)
        inst_idx = instrument_idx.detach().cpu().numpy()
        time_idx = time_idx_end.detach().cpu().numpy()
        
        valid = mask.astype(bool)
        pred = pred[valid]
        true = true[valid]
        inst_idx = inst_idx[valid]
        time_idx = time_idx[valid]
        
        if pred.size == 0:
            return
        
        unique_inst = np.unique(inst_idx)
        if cfg.plot_val_max_instruments and cfg.plot_val_max_instruments > 0:
            counts = {i: int((inst_idx == i).sum()) for i in unique_inst}
            unique_inst = np.array(
                sorted(unique_inst, key=lambda i: counts.get(int(i), 0), reverse=True)[:cfg.plot_val_max_instruments]
            )
        
        out_dir = Path(self.config.log_dir) / "plots" / f"val_epoch_{self.current_epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        wandb_logger = None
        if cfg.plot_val_log_to_wandb:
            for logger in getattr(self, "loggers", []) or []:
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break
        
        images_logged = 0
        
        for inst in unique_inst:
            inst_mask = inst_idx == inst
            inst_pred = pred[inst_mask]
            inst_true = true[inst_mask]
            inst_time = time_idx[inst_mask]
            
            window_size = min(cfg.plot_val_window_size, inst_pred.size)
            if window_size < 8:
                continue
            
            order = np.argsort(inst_time)
            inst_pred = inst_pred[order]
            inst_true = inst_true[order]
            inst_time = inst_time[order]
            
            min_stride = max(1, cfg.plot_val_min_stride)
            starts = self._select_window_starts(
                n_points=len(inst_pred),
                window_size=window_size,
                num_windows=cfg.plot_val_windows_per_instrument,
                min_stride=min_stride,
            )
            if not starts:
                continue
            
            n_rows = len(starts)
            fig, axes = plt.subplots(
                n_rows,
                1,
                figsize=(10, 2.4 * n_rows),
                sharex=False,
            )
            if n_rows == 1:
                axes = [axes]
            
            inst_id = self.instrument_ids[int(inst)] if int(inst) < len(self.instrument_ids) else f"inst_{int(inst)}"
            
            for i, start in enumerate(starts):
                end = start + window_size
                x = inst_time[start:end]
                y_true = inst_true[start:end]
                y_pred = inst_pred[start:end]
                
                mae = float(np.mean(np.abs(y_true - y_pred))) if y_true.size else float("nan")
                corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
                
                ax = axes[i]
                ax.plot(x, y_true, label="actual", color="black", linewidth=1.0)
                ax.plot(x, y_pred, label="pred", color="tab:blue", linewidth=1.0)
                ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
                ax.set_title(
                    f"{inst_id} | window {i + 1} | idx {int(x[0])}-{int(x[-1])} | MAE {mae:.4f} | Corr {corr:.2f}"
                )
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)
            
            fig.tight_layout()
            
            safe_inst = "".join(c if c.isalnum() or c in "-_." else "_" for c in inst_id)
            out_path = out_dir / f"{safe_inst}_h{cfg.plot_val_horizon}_e{self.current_epoch:03d}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            
            if wandb_logger is not None and images_logged < cfg.plot_val_log_max_images:
                wandb_logger.experiment.log(
                    {f"val/sliding_windows/{safe_inst}": wandb.Image(str(out_path))},
                    step=self.global_step,
                )
                images_logged += 1
    
    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        result = self._shared_step(batch, "test")
        
        self.log("test/loss", result["loss"], on_epoch=True)
        
        return result
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        cfg = self.config.training
        
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        
        # Learning rate scheduler with warmup
        if cfg.lr_scheduler == "cosine_warmup":
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=cfg.warmup_start_lr / cfg.learning_rate,
                total_iters=cfg.warmup_epochs,
            )
            
            # Cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg.max_epochs - cfg.warmup_epochs,
                eta_min=cfg.min_learning_rate,
            )
            
            # Combined scheduler
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[cfg.warmup_epochs],
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_callbacks(config: Config) -> List[pl.Callback]:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=f"{config.experiment_name}" + "-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",  # Fixed: must match logged metric name
        mode=config.training.checkpoint_mode,
        save_top_k=config.training.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=config.training.early_stopping_min_delta,
        patience=config.training.early_stopping_patience,
        mode="min",
        verbose=True,
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks


def create_loggers(config: Config) -> List:
    """Create training loggers with WandB."""
    loggers = []
    
    # WandB logger with comprehensive config logging
    wandb_logger = WandbLogger(
        project="temporal-transformer-crypto",
        name=config.experiment_name,
        save_dir=config.log_dir,
        log_model=True,  # Log model checkpoints to WandB
        config={
            # Data config
            "dataset_path": str(config.dataset_path),
            "train_start": config.dates.train_start,
            "train_end": config.dates.train_end,
            "test_start": config.dates.test_start,
            "test_end": config.dates.test_end,
            "val_ratio": config.dates.val_ratio,
            # Feature config
            "num_features": config.features.get_total_input_dim(),
            "num_continuous": config.features.get_num_continuous_features(),
            "num_static": config.features.get_num_static_features(),
            "ohlcv_features": len(config.features.ohlcv_features),
            "depth_features": len(config.features.depth_features),
            "lunar_features": len(config.features.lunar_features),
            "major_coin_features": len(config.features.major_coin_features),
            "cluster_features": len(config.features.cluster_features),
            # Target config
            "horizons": config.targets.horizons,
            "num_regression_targets": config.targets.get_num_regression_targets(),
            "num_direction_targets": config.targets.get_num_direction_targets(),
            "num_binary_targets": config.targets.get_num_binary_targets(),
            # Model config
            "d_model": config.model.d_model,
            "n_heads": config.model.n_heads,
            "n_encoder_layers": config.model.n_encoder_layers,
            "d_ff": config.model.d_ff,
            "dropout": config.model.dropout,
            "seq_length": config.model.seq_length,
            "use_revin": config.model.use_revin,
            "activation": config.model.activation,
            "estimated_params": config.model.get_estimated_params(),
            # Training config
            "batch_size": config.training.batch_size,
            "effective_batch_size": config.training.batch_size * config.training.accumulate_grad_batches,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "max_epochs": config.training.max_epochs,
            "warmup_epochs": config.training.warmup_epochs,
            "gradient_clip_val": config.training.gradient_clip_val,
            "precision": config.training.precision,
        },
        tags=["temporal-transformer", "crypto", "multi-horizon", "revin"],
    )
    loggers.append(wandb_logger)
    
    # CSV logger as backup
    csv_logger = CSVLogger(
        save_dir=config.log_dir,
        name=config.experiment_name,
    )
    loggers.append(csv_logger)
    
    return loggers


def train(
    config: Optional[Config] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> TemporalTransformerLightning:
    """
    Main training function.
    
    Args:
        config: Configuration object (uses default if None)
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Trained Lightning module
    """
    if config is None:
        config = get_config()
    
    # Print config
    config.print_summary()
    
    # Set up data
    print("\n" + "=" * 80)
    print(" SETTING UP DATA")
    print("=" * 80)
    data_module = create_data_module(config)
    
    # Get actual feature dimension from data
    feature_dim = data_module.get_feature_dim()
    print(f"\nActual feature dimension: {feature_dim}")
    
    # Create model
    print("\n" + "=" * 80)
    print(" CREATING MODEL")
    print("=" * 80)
    model = TemporalTransformerLightning(
        config,
        feature_dim=feature_dim,
        instrument_ids=data_module.val_dataset.instruments if data_module.val_dataset else None,
    )
    
    # Create callbacks and loggers
    callbacks = create_callbacks(config)
    loggers = create_loggers(config)
    
    # Check GPU availability
    print("\n" + "=" * 80)
    print(" HARDWARE")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ No GPU available, using CPU")
    
    # Create trainer
    print("\n" + "=" * 80)
    print(" TRAINING")
    print("=" * 80)
    
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        min_epochs=config.training.min_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
    )
    
    # Train
    train_loader = data_module.train_dataloader(use_instrument_sampler=True)
    val_loader = data_module.val_dataloader()
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint,
    )
    
    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    
    # Finish WandB run
    wandb.finish()
    
    return model


def test(
    config: Optional[Config] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Test the model on test set.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint to load
        
    Returns:
        Test metrics dictionary
    """
    if config is None:
        config = get_config()
    
    # Set up data
    data_module = create_data_module(config)
    
    # Load model
    if checkpoint_path:
        model = TemporalTransformerLightning.load_from_checkpoint(
            checkpoint_path,
            config=config,
            feature_dim=data_module.get_feature_dim(),
        )
    else:
        raise ValueError("checkpoint_path required for testing")
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        logger=False,
    )
    
    # Test
    test_loader = data_module.test_dataloader()
    results = trainer.test(model, dataloaders=test_loader)
    
    return results[0] if results else {}


# =============================================================================
# INFERENCE
# =============================================================================

class Predictor:
    """
    Inference wrapper for trained model.
    
    Handles loading checkpoint and making predictions.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: str = "auto",
    ):
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = TemporalTransformerLightning.load_from_checkpoint(
            checkpoint_path,
            config=config,
        )
        self.model.eval()
        self.model.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions.
        
        Args:
            features: (batch, seq_len, n_features) input features
            mask: Optional mask
            
        Returns:
            Predictions dictionary
        """
        features = features.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        
        outputs = self.model(features, mask)
        
        # Move to CPU
        return {k: v.cpu() for k, v in outputs.items() if isinstance(v, torch.Tensor)}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Temporal Transformer")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--test", type=str, default=None, help="Test checkpoint")
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.test:
        # Test mode
        results = test(config, checkpoint_path=args.test)
        print("\nTest results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    else:
        # Train mode
        train(config, resume_from_checkpoint=args.resume)
