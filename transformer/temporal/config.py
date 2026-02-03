"""
Configuration for Temporal Transformer Training Pipeline

This module defines all hyperparameters, paths, feature lists, and target configurations
for training a large-scale temporal transformer on crypto time series data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "data_storage_bitget"
OUTPUT_DIR = DATA_DIR / "output" / "intermediate"

# Dataset paths
DATASET_PATH = OUTPUT_DIR / "step_6" / "all_matched_data_with_targets.csv"

# Model output paths
MODEL_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CHECKPOINT_DIR = MODEL_OUTPUT_DIR / "checkpoints"
LOG_DIR = MODEL_OUTPUT_DIR / "logs"


# =============================================================================
# DATE CONFIGURATION - Train/Test Split
# =============================================================================

@dataclass
class DateConfig:
    """Date-based train/test split configuration."""
    
    # Training period: Aug 1, 2024 to Sep 1, 2025 (13 months)
    train_start: str = "2024-08-01"
    train_end: str = "2025-09-01"
    
    # Test period: Sep 1, 2025 to Jan 20, 2026 (~4.5 months)
    test_start: str = "2025-09-01"
    test_end: str = "2026-01-20"
    
    # Validation split from training data (last N% of training period)
    val_ratio: float = 0.1  # 10% of training data for validation
    
    def get_train_start_dt(self) -> datetime:
        return datetime.strptime(self.train_start, "%Y-%m-%d")
    
    def get_train_end_dt(self) -> datetime:
        return datetime.strptime(self.train_end, "%Y-%m-%d")
    
    def get_test_start_dt(self) -> datetime:
        return datetime.strptime(self.test_start, "%Y-%m-%d")
    
    def get_test_end_dt(self) -> datetime:
        return datetime.strptime(self.test_end, "%Y-%m-%d")


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Feature groups for the model."""
    
    # Core OHLCV features (time-varying, continuous)
    ohlcv_features: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume",
    ])
    
    # Depth/Order book features (time-varying, continuous)
    depth_features: List[str] = field(default_factory=lambda: [
        "depth_ask_price", "depth_bid_price",
        "depth_ask_volume", "depth_bid_volume",
        "depth_spread", "depth_mid_price",
    ])
    
    # Lunar/Social metrics (time-varying, continuous)
    lunar_features: List[str] = field(default_factory=lambda: [
        "lunar_contributors_active", "lunar_contributors_created",
        "lunar_interactions", "lunar_posts_active", "lunar_posts_created",
        "lunar_sentiment", "lunar_spam",
        "lunar_alt_rank", "lunar_circulating_supply",
        "lunar_galaxy_score", "lunar_market_cap",
        "lunar_market_dominance", "lunar_social_dominance",
        "lunar_volume_24h",
    ])
    
    # Major coin reference features (time-varying, continuous)
    # OPTION C: Core features only - close, volume + key lunar signals
    # Removed: alt_rank (always top5), circulating_supply (static), 
    #          market_cap (redundant), volume_24h (redundant),
    #          contributors_created, posts_created, spam (low signal)
    major_coin_features: List[str] = field(default_factory=lambda: [
        # BTC - Price + Core Lunar (7 features)
        "btc_close", "btc_volume",
        "btc_lunar_sentiment", "btc_lunar_galaxy_score",
        "btc_lunar_social_dominance", "btc_lunar_interactions",
        "btc_lunar_market_dominance",
        # ETH - Price + Core Lunar (7 features)
        "eth_close", "eth_volume",
        "eth_lunar_sentiment", "eth_lunar_galaxy_score",
        "eth_lunar_social_dominance", "eth_lunar_interactions",
        "eth_lunar_market_dominance",
        # DOGE - Price + Core Lunar (7 features)
        "doge_close", "doge_volume",
        "doge_lunar_sentiment", "doge_lunar_galaxy_score",
        "doge_lunar_social_dominance", "doge_lunar_interactions",
        "doge_lunar_market_dominance",
        # SOL - Price + Core Lunar (7 features)
        "sol_close", "sol_volume",
        "sol_lunar_sentiment", "sol_lunar_galaxy_score",
        "sol_lunar_interactions", "sol_lunar_social_dominance",
        "sol_lunar_market_dominance",
        # Fear & Greed Index
        "fng",
    ])
    
    # Behavioral cluster features (static per instrument, categorical/binary)
    cluster_features: List[str] = field(default_factory=lambda: [
        "cluster_extreme_fade",
        "cluster_hype_tech_fade",
        "cluster_gaming_retail_fade",
        "cluster_nft_cycle_fade",
        "cluster_dao_soft_fade",
        "cluster_defi_extended_fade",
        "cluster_defi_core_resilient",
        "cluster_solana_ecosystem",
        "cluster_base_ecosystem",
        "cluster_arbitrum_infra_ecosystem",
        "cluster_layer1_mixed",
        "cluster_layer2_speculative",
        "cluster_infra_slow_fade",
        "cluster_data_identity_low_beta",
        "cluster_long_tail_death",
        "cluster_reg_us_narrative",
    ])
    
    # Time features (to be engineered)
    time_features: List[str] = field(default_factory=lambda: [
        "hour_sin", "hour_cos",
        "day_of_week_sin", "day_of_week_cos",
        "day_of_month_sin", "day_of_month_cos",
        "month_sin", "month_cos",
    ])
    
    def get_all_continuous_features(self) -> List[str]:
        """Get all continuous input features for the model."""
        return (
            self.ohlcv_features +
            self.depth_features +
            self.lunar_features +
            self.major_coin_features
        )
    
    def get_static_features(self) -> List[str]:
        """Get static (per-instrument) features."""
        return self.cluster_features
    
    def get_all_features(self) -> List[str]:
        """Get all features including engineered time features."""
        return (
            self.get_all_continuous_features() +
            self.cluster_features +
            self.time_features
        )
    
    def get_num_continuous_features(self) -> int:
        """Number of continuous features (for RevIN)."""
        return len(self.get_all_continuous_features())
    
    def get_num_static_features(self) -> int:
        """Number of static/categorical features."""
        return len(self.cluster_features)
    
    def get_total_input_dim(self) -> int:
        """Total input dimension for the model."""
        return len(self.get_all_features())


# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

@dataclass
class TargetConfig:
    """Configurable target selection for multi-task learning."""
    
    # Available prediction horizons
    horizons: List[str] = field(default_factory=lambda: [
        "15min", "1h", "4h", "12h"
    ])
    
    # Horizon to number of candles mapping
    horizon_steps: Dict[str, int] = field(default_factory=lambda: {
        "15min": 1,
        "1h": 4,
        "4h": 16,
        "12h": 48,
    })
    
    # Target types
    # - log_return: Continuous regression target
    # - direction: 3-class classification (-1, 0, +1)
    # - strong_pump: Binary classification (>3% move up)
    # - strong_dump: Binary classification (>3% move down)
    
    # ACTIVE TARGETS: Configure which targets to train on
    # Set to True to include in training, False to exclude
    active_targets: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        "15min": {
            "log_return": True,
            "direction": True,
            "strong_pump": True,
            "strong_dump": True,
        },
        "1h": {
            "log_return": True,
            "direction": True,
            "strong_pump": True,
            "strong_dump": True,
        },
        "4h": {
            "log_return": True,
            "direction": True,
            "strong_pump": True,
            "strong_dump": True,
        },
        "12h": {
            "log_return": True,
            "direction": True,
            "strong_pump": True,
            "strong_dump": True,
        },
    })
    
    # Loss weights for multi-task learning
    # Higher weight = more importance in combined loss
    target_weights: Dict[str, float] = field(default_factory=lambda: {
        "log_return": 1.0,      # Primary regression task
        "direction": 0.5,       # 3-class direction
        "strong_pump": 0.3,     # Binary pump detection
        "strong_dump": 0.3,     # Binary dump detection
    })
    
    # Horizon weights (optional: weight different horizons differently)
    horizon_weights: Dict[str, float] = field(default_factory=lambda: {
        "15min": 1.0,
        "1h": 1.0,
        "4h": 1.0,
        "12h": 1.0,
    })
    
    def get_active_target_columns(self) -> List[str]:
        """Get list of all active target column names."""
        columns = []
        for horizon in self.horizons:
            for target_type, is_active in self.active_targets[horizon].items():
                if is_active:
                    columns.append(f"target_{target_type}_{horizon}")
        return columns
    
    def get_regression_targets(self) -> List[str]:
        """Get regression target column names."""
        columns = []
        for horizon in self.horizons:
            if self.active_targets[horizon].get("log_return", False):
                columns.append(f"target_log_return_{horizon}")
        return columns
    
    def get_classification_targets(self) -> List[str]:
        """Get classification target column names (direction, pump, dump)."""
        columns = []
        for horizon in self.horizons:
            for target_type in ["direction", "strong_pump", "strong_dump"]:
                if self.active_targets[horizon].get(target_type, False):
                    columns.append(f"target_{target_type}_{horizon}")
        return columns
    
    def get_num_regression_targets(self) -> int:
        """Number of regression outputs."""
        return len(self.get_regression_targets())
    
    def get_num_direction_targets(self) -> int:
        """Number of direction classification outputs (3-class each)."""
        count = 0
        for horizon in self.horizons:
            if self.active_targets[horizon].get("direction", False):
                count += 1
        return count
    
    def get_num_binary_targets(self) -> int:
        """Number of binary classification outputs."""
        count = 0
        for horizon in self.horizons:
            for target_type in ["strong_pump", "strong_dump"]:
                if self.active_targets[horizon].get(target_type, False):
                    count += 1
        return count


# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION - LARGE SCALE
# =============================================================================

@dataclass
class ModelConfig:
    """
    Large-scale Temporal Transformer architecture configuration.
    
    Designed for intensive training with significant capacity:
    - ~50M parameters
    - Deep architecture for complex temporal patterns
    - Wide hidden dimensions for rich representations
    """
    
    # Sequence configuration
    seq_length: int = 96              # 24 hours of 15-min candles (encoder length)
    max_seq_length: int = 192         # Maximum sequence length for positional encoding
    
    # Transformer architecture - LARGE SCALE
    d_model: int = 512                # Hidden dimension (wide)
    n_heads: int = 8                  # Attention heads
    n_encoder_layers: int = 12        # Deep encoder (12 layers)
    d_ff: int = 2048                  # Feed-forward dimension (4x d_model)
    
    # Regularization
    dropout: float = 0.1              # Dropout rate
    attention_dropout: float = 0.1    # Attention-specific dropout
    ff_dropout: float = 0.1           # Feed-forward dropout
    
    # RevIN configuration (Reversible Instance Normalization)
    use_revin: bool = True            # Enable RevIN for distribution shift
    revin_affine: bool = True         # Learnable affine parameters in RevIN
    revin_eps: float = 1e-5           # Epsilon for numerical stability
    
    # Input projection
    input_projection_layers: int = 2  # MLP layers for input projection
    
    # Output heads configuration
    regression_head_layers: int = 2   # MLP layers for regression head
    classification_head_layers: int = 2  # MLP layers for classification heads
    
    # Activation
    activation: str = "gelu"          # GELU activation (better than ReLU for transformers)
    
    # Layer normalization
    norm_first: bool = True           # Pre-norm (more stable training)
    
    # Static feature handling
    static_embedding_dim: int = 64    # Embedding dim for cluster features
    
    def get_estimated_params(self) -> int:
        """Estimate total number of parameters."""
        # Rough estimation
        # Transformer layer: ~4 * d_model^2 (attention) + 2 * d_model * d_ff (FFN)
        transformer_params = self.n_encoder_layers * (
            4 * self.d_model ** 2 +  # Multi-head attention
            2 * self.d_model * self.d_ff  # FFN
        )
        # Input/output projections
        io_params = self.d_model * 1000  # Rough estimate
        return transformer_params + io_params


# =============================================================================
# TRAINING CONFIGURATION - INTENSIVE
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training hyperparameters for intensive training.
    
    Designed for thorough training with proper regularization.
    """
    
    # Batch configuration
    batch_size: int = 64              # Batch size per GPU
    accumulate_grad_batches: int = 4  # Effective batch size = 64 * 4 = 256
    
    # Training duration
    max_epochs: int = 100             # Maximum epochs
    min_epochs: int = 10              # Minimum epochs before early stopping
    
    # Learning rate
    learning_rate: float = 1e-4       # Initial learning rate
    min_learning_rate: float = 1e-6  # Minimum LR for scheduler
    
    # Learning rate schedule
    lr_scheduler: str = "cosine_warmup"  # cosine with warmup
    warmup_epochs: int = 5            # Warmup epochs
    warmup_start_lr: float = 1e-7     # Starting LR for warmup
    
    # Gradient clipping (important for transformers)
    gradient_clip_val: float = 1.0    # Clip gradients
    gradient_clip_algorithm: str = "norm"  # Clip by norm
    
    # Early stopping
    early_stopping_patience: int = 15  # Epochs without improvement
    early_stopping_min_delta: float = 1e-4  # Minimum improvement
    
    # Optimizer
    optimizer: str = "adamw"          # AdamW optimizer
    weight_decay: float = 0.01        # L2 regularization
    betas: tuple = (0.9, 0.999)       # Adam betas
    
    # Multi-task learning
    loss_combination: str = "weighted_sum"  # How to combine losses
    
    # Checkpointing
    save_top_k: int = 3               # Keep top K checkpoints
    checkpoint_metric: str = "val/loss"  # Metric for checkpoint selection (must match logged name)
    checkpoint_mode: str = "min"      # Minimize the metric
    
    # Logging
    log_every_n_steps: int = 50       # Log frequency
    val_check_interval: float = 0.25  # Validate 4x per epoch
    
    # Validation plotting
    plot_val_windows: bool = True         # Create sliding window plots on val epoch end
    plot_val_windows_per_instrument: int = 4  # How many windows per instrument
    plot_val_window_size: int = 64        # Points per window (validation samples)
    plot_val_min_stride: int = 48         # Minimum stride between windows (reduce overlap)
    plot_val_horizon: str = "15min"       # Horizon name to visualize
    plot_val_max_instruments: int = 0     # 0 = all instruments, otherwise cap
    plot_val_log_to_wandb: bool = True    # Log a subset of plots to WandB
    plot_val_log_max_images: int = 8      # Max images to log to WandB per epoch
    
    # Hardware
    precision: str = "16-mixed"       # Mixed precision training (faster)
    num_workers: int = 4              # DataLoader workers
    pin_memory: bool = True           # Pin memory for GPU transfer
    
    # Reproducibility
    seed: int = 42                    # Random seed


# =============================================================================
# COMBINED CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    dates: DateConfig = field(default_factory=DateConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment name
    experiment_name: str = "temporal_transformer_v1"
    
    # Paths
    dataset_path: Path = DATASET_PATH
    checkpoint_dir: Path = CHECKPOINT_DIR
    log_dir: Path = LOG_DIR
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print(" TEMPORAL TRANSFORMER CONFIGURATION")
        print("=" * 80)
        print(f"\n📁 Dataset: {self.dataset_path}")
        print(f"\n📅 Date Splits:")
        print(f"   Train: {self.dates.train_start} to {self.dates.train_end}")
        print(f"   Test:  {self.dates.test_start} to {self.dates.test_end}")
        print(f"\n📊 Features:")
        print(f"   Continuous: {self.features.get_num_continuous_features()}")
        print(f"   Static: {self.features.get_num_static_features()}")
        print(f"   Time: {len(self.features.time_features)}")
        print(f"   Total: {self.features.get_total_input_dim()}")
        print(f"\n🎯 Targets:")
        print(f"   Regression: {self.targets.get_num_regression_targets()}")
        print(f"   Direction (3-class): {self.targets.get_num_direction_targets()}")
        print(f"   Binary: {self.targets.get_num_binary_targets()}")
        print(f"   Active columns: {len(self.targets.get_active_target_columns())}")
        print(f"\n🧠 Model Architecture:")
        print(f"   d_model: {self.model.d_model}")
        print(f"   n_heads: {self.model.n_heads}")
        print(f"   n_layers: {self.model.n_encoder_layers}")
        print(f"   d_ff: {self.model.d_ff}")
        print(f"   Estimated params: ~{self.model.get_estimated_params() / 1e6:.1f}M")
        print(f"   RevIN: {'Yes' if self.model.use_revin else 'No'}")
        print(f"\n🏋️ Training:")
        print(f"   Batch size: {self.training.batch_size} x {self.training.accumulate_grad_batches} = {self.training.batch_size * self.training.accumulate_grad_batches}")
        print(f"   Max epochs: {self.training.max_epochs}")
        print(f"   LR: {self.training.learning_rate} (warmup: {self.training.warmup_epochs} epochs)")
        print(f"   Precision: {self.training.precision}")
        print("=" * 80)


# =============================================================================
# DEFAULT CONFIG INSTANCE
# =============================================================================

def get_config() -> Config:
    """Get default configuration."""
    return Config()


if __name__ == "__main__":
    # Print configuration summary when run directly
    config = get_config()
    config.print_summary()
