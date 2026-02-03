"""
Temporal Transformer Model with RevIN

This module implements:
1. RevIN (Reversible Instance Normalization) for distribution shift
2. Large-scale Temporal Transformer encoder
3. Multi-task prediction heads (regression + classification)
4. Positional encoding for temporal sequences

Architecture optimized for crypto time series forecasting.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import Config, get_config


# =============================================================================
# REVERSIBLE INSTANCE NORMALIZATION (RevIN)
# =============================================================================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for time series.
    
    From: "Reversible Instance Normalization for Accurate Time-Series 
           Forecasting against Distribution Shift" (ICLR 2022)
    
    Key idea: Normalize input at instance level (per sample, per feature),
    store statistics for denormalization of outputs.
    
    This handles distribution shift between train and test time by:
    1. Removing instance-specific statistics during forward pass
    2. Re-injecting them for the output (especially for regression targets)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Args:
            num_features: Number of features to normalize
            eps: Epsilon for numerical stability
            affine: Whether to use learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            # Learnable scale and shift (shared across instances)
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # Storage for normalization statistics (set during forward)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = "norm",
        target_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Apply normalization or denormalization.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features) for norm
               or (batch, num_outputs) for denorm
            mode: 'norm' for normalization, 'denorm' for denormalization
            target_indices: For denorm, which feature indices correspond to outputs
            
        Returns:
            Normalized/denormalized tensor
        """
        if mode == "norm":
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x, target_indices)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input sequence.
        
        Args:
            x: (batch, seq_len, features)
            
        Returns:
            Normalized tensor with stored mean/std
        """
        # Compute instance statistics across time dimension
        # Shape: (batch, 1, features)
        self.mean = x.mean(dim=1, keepdim=True)
        self.std = x.std(dim=1, keepdim=True) + self.eps
        
        # Normalize
        x_norm = (x - self.mean) / self.std
        
        # Apply learnable affine transformation
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
        
        return x_norm
    
    def _denormalize(
        self,
        x: torch.Tensor,
        target_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Denormalize output predictions.
        
        Used for regression targets to restore original scale.
        
        Args:
            x: (batch, num_outputs)
            target_indices: Which feature indices the outputs correspond to
            
        Returns:
            Denormalized tensor
        """
        if self.mean is None or self.std is None:
            raise ValueError("Must call forward with mode='norm' first!")
        
        # Remove affine transformation if present
        if self.affine:
            x = (x - self.affine_bias[target_indices]) / (self.affine_weight[target_indices] + self.eps)
        
        # Select the relevant statistics
        # self.mean, self.std: (batch, 1, features)
        # We need: (batch, num_outputs)
        if target_indices is not None:
            mean = self.mean[:, 0, target_indices]  # (batch, num_outputs)
            std = self.std[:, 0, target_indices]    # (batch, num_outputs)
        else:
            mean = self.mean[:, 0, :x.shape[-1]]
            std = self.std[:, 0, :x.shape[-1]]
        
        # Denormalize
        x_denorm = x * std + mean
        
        return x_denorm


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    Uses sin/cos functions at different frequencies to encode position.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            Position-encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Often works better than sinusoidal for shorter sequences.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# TRANSFORMER ENCODER BLOCK
# =============================================================================

class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with pre-norm architecture.
    
    Pre-norm is more stable for deep transformers.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_mask: Optional attention mask
            key_padding_mask: Optional padding mask (batch, seq_len)
            use_causal_mask: Whether to apply causal masking (prevent looking ahead)

        Returns:
            Output tensor and attention weights
        """
        # Create causal mask if needed (CRITICAL for financial forecasting)
        if use_causal_mask and attn_mask is None:
            seq_len = x.size(1)
            # Upper triangular mask: True = do NOT attend (future positions)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )
            attn_mask = causal_mask

        if self.norm_first:
            # Pre-norm: Norm -> Attention -> Residual
            x_norm = self.norm1(x)
            attn_out, attn_weights = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout(attn_out)

            # Pre-norm: Norm -> FFN -> Residual
            x_norm = self.norm2(x)
            x = x + self.ff(x_norm)
        else:
            # Post-norm: Attention -> Residual -> Norm
            attn_out, attn_weights = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.ff(x))

        return x, attn_weights


# =============================================================================
# MULTI-TASK PREDICTION HEAD
# =============================================================================

class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for combined regression and classification.

    Handles:
    1. Regression targets (log returns) - MSE loss
    2. Direction classification (3-class: -1, 0, +1) - CrossEntropy
    3. Binary classification (strong pump/dump) - BCE
    4. Volatility prediction (for risk management)
    5. Prediction confidence (epistemic uncertainty)
    """

    def __init__(
        self,
        d_model: int,
        config: Config,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config

        if hidden_dim is None:
            hidden_dim = d_model // 2

        # Get target counts
        self.n_regression = config.targets.get_num_regression_targets()
        self.n_direction = config.targets.get_num_direction_targets()
        self.n_binary = config.targets.get_num_binary_targets()

        # Build heads
        # Regression head (outputs continuous values)
        if self.n_regression > 0:
            self.regression_head = self._build_head(
                d_model, hidden_dim, self.n_regression, num_layers, dropout
            )

        # Direction head (outputs 3 logits per horizon)
        if self.n_direction > 0:
            self.direction_head = self._build_head(
                d_model, hidden_dim, self.n_direction * 3, num_layers, dropout
            )

        # Binary classification head (outputs logits for pump/dump)
        if self.n_binary > 0:
            self.binary_head = self._build_head(
                d_model, hidden_dim, self.n_binary, num_layers, dropout
            )

        # Volatility prediction head (predict future volatility for risk management)
        # One volatility prediction per regression target
        if self.n_regression > 0:
            self.volatility_head = self._build_head(
                d_model, hidden_dim, self.n_regression, num_layers, dropout
            )

        # Confidence head (predict prediction uncertainty - epistemic)
        # One confidence score per regression target
        if self.n_regression > 0:
            self.confidence_head = self._build_head(
                d_model, hidden_dim, self.n_regression, num_layers, dropout
            )
        
        # Store target column ordering for proper loss computation
        self.target_cols = config.targets.get_active_target_columns()
        self._build_target_indices()
    
    def _build_head(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Build an MLP head."""
        layers = []
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        return nn.Sequential(*layers)
    
    def _build_target_indices(self):
        """Build mapping from target columns to output indices."""
        self.regression_indices = []
        self.direction_indices = []
        self.binary_indices = []
        
        for i, col in enumerate(self.target_cols):
            if "log_return" in col:
                self.regression_indices.append(i)
            elif "direction" in col:
                self.direction_indices.append(i)
            elif "strong_pump" in col or "strong_dump" in col:
                self.binary_indices.append(i)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all heads.

        Args:
            x: (batch, d_model) - pooled sequence representation

        Returns:
            Dictionary with predictions:
            - regression: (batch, n_regression)
            - direction_logits: (batch, n_direction, 3)
            - binary_logits: (batch, n_binary)
            - volatility: (batch, n_regression) - predicted volatility (always positive)
            - confidence: (batch, n_regression) - prediction confidence (0-1)
        """
        outputs = {}

        if self.n_regression > 0:
            outputs["regression"] = self.regression_head(x)

            # Volatility prediction (use softplus to ensure positive)
            volatility_raw = self.volatility_head(x)
            outputs["volatility"] = F.softplus(volatility_raw)

            # Confidence prediction (use sigmoid to get 0-1 range)
            confidence_raw = self.confidence_head(x)
            outputs["confidence"] = torch.sigmoid(confidence_raw)

        if self.n_direction > 0:
            direction_logits = self.direction_head(x)  # (batch, n_direction * 3)
            outputs["direction_logits"] = direction_logits.view(-1, self.n_direction, 3)

        if self.n_binary > 0:
            outputs["binary_logits"] = self.binary_head(x)

        return outputs
    
    def get_target_indices(self) -> Dict[str, List[int]]:
        """Get indices mapping for each task type."""
        return {
            "regression": self.regression_indices,
            "direction": self.direction_indices,
            "binary": self.binary_indices,
        }


# =============================================================================
# STATIC FEATURE ENCODER
# =============================================================================

class StaticFeatureEncoder(nn.Module):
    """
    Encode static features (cluster memberships) into embeddings.
    
    Projects binary cluster features into a dense representation
    that's added to the sequence encoding.
    """
    
    def __init__(
        self,
        num_clusters: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Project cluster features to model dimension
        self.projection = nn.Sequential(
            nn.Linear(num_clusters, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, cluster_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cluster_features: (batch, num_clusters) binary features
            
        Returns:
            (batch, d_model) static embedding
        """
        return self.projection(cluster_features)


# =============================================================================
# MAIN TEMPORAL TRANSFORMER MODEL
# =============================================================================

class TemporalTransformer(nn.Module):
    """
    Large-scale Temporal Transformer for multi-horizon crypto prediction.
    
    Architecture:
    1. Input projection (MLP to d_model)
    2. RevIN for distribution shift handling
    3. Positional encoding
    4. N transformer encoder layers
    5. Sequence pooling (mean or last token)
    6. Multi-task prediction heads
    
    Supports:
    - Multi-horizon prediction (15min, 1h, 4h, 12h)
    - Regression + classification outputs
    - RevIN normalization at model level
    - Static feature conditioning
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        model_cfg = config.model
        
        # Store dimensions
        self.d_model = model_cfg.d_model
        self.seq_length = model_cfg.seq_length
        
        # Calculate input dimension
        # This will be set properly when we know the actual feature count
        self.input_dim = config.features.get_total_input_dim()
        
        # 1. Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, model_cfg.d_model),
            nn.GELU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.d_model, model_cfg.d_model),
        )
        
        # 2. RevIN (on continuous features only, NOT static/cluster features)
        self.use_revin = model_cfg.use_revin
        if self.use_revin:
            # Only normalize continuous features (OHLCV, depth, lunar, major coins)
            # NOT cluster features or time features
            self.n_continuous = config.features.get_num_continuous_features()
            self.revin = RevIN(
                num_features=self.n_continuous,
                eps=model_cfg.revin_eps,
                affine=model_cfg.revin_affine,
            )
        
        # 3. Positional encoding
        self.pos_encoder = LearnablePositionalEncoding(
            d_model=model_cfg.d_model,
            max_len=model_cfg.max_seq_length,
            dropout=model_cfg.dropout,
        )
        
        # 4. Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=model_cfg.d_model,
                n_heads=model_cfg.n_heads,
                d_ff=model_cfg.d_ff,
                dropout=model_cfg.dropout,
                attention_dropout=model_cfg.attention_dropout,
                activation=model_cfg.activation,
                norm_first=model_cfg.norm_first,
            )
            for _ in range(model_cfg.n_encoder_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(model_cfg.d_model)

        # Attention-based pooling for better sequence aggregation
        self.pooling_attention = nn.Linear(model_cfg.d_model, 1)

        # 5. Multi-task prediction head
        self.prediction_head = MultiTaskHead(
            d_model=model_cfg.d_model,
            config=config,
            num_layers=model_cfg.regression_head_layers,
            dropout=model_cfg.dropout,
        )
        
        # Initialize weights
        self._init_weights()
        
        # Print model summary
        self._print_summary()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _print_summary(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'=' * 60}")
        print(f" TEMPORAL TRANSFORMER MODEL")
        print(f"{'=' * 60}")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Model dim: {self.d_model}")
        print(f"  Heads: {self.config.model.n_heads}")
        print(f"  Layers: {self.config.model.n_encoder_layers}")
        print(f"  FFN dim: {self.config.model.d_ff}")
        print(f"  RevIN: {'Yes' if self.use_revin else 'No'}")
        print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable: {trainable_params:,}")
        print(f"{'=' * 60}\n")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, seq_len, input_dim)
            mask: Optional padding mask (batch, seq_len)
                  CONVENTION: True = valid position, False = padded/invalid
                  (Will be inverted for PyTorch's key_padding_mask where True = ignore)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # 1. Apply RevIN normalization (only on continuous features)
        if self.use_revin:
            # Split input into continuous and non-continuous parts
            x_continuous = x[:, :, :self.n_continuous]  # First N features are continuous
            x_other = x[:, :, self.n_continuous:]  # Rest are static/time features

            # Normalize only continuous features
            x_continuous = self.revin(x_continuous, mode="norm")

            # Concatenate back
            x = torch.cat([x_continuous, x_other], dim=-1)

        # 2. Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 3. Positional encoding
        x = self.pos_encoder(x)
        
        # 4. Transformer encoder
        encoder_attention_weights = []  # Renamed to avoid conflict

        # Convert mask for attention
        # MASK CONVENTION:
        #   - Our mask: True = valid, False = padded
        #   - PyTorch key_padding_mask: True = ignore, False = attend
        #   → We need to INVERT our mask for PyTorch
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask  # Invert: valid→attend, padded→ignore

        for layer in self.encoder_layers:
            x, attn_w = layer(x, key_padding_mask=key_padding_mask, use_causal_mask=True)
            if return_attention:
                encoder_attention_weights.append(attn_w)

        # 5. Final norm
        x = self.final_norm(x)

        # 6. Sequence pooling - attention-weighted pooling (better than last-token)
        # Compute attention scores for each position
        attn_scores = self.pooling_attention(x)  # (batch, seq_len, 1)

        # Apply mask if provided (set invalid positions to -inf)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        # Softmax to get weights
        pooling_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        pooled = (x * pooling_weights).sum(dim=1)  # (batch, d_model)

        # 7. Multi-task predictions
        predictions = self.prediction_head(pooled)

        # Optionally denormalize regression outputs
        # (disabled by default as targets are already in log-return space)

        if return_attention:
            predictions["encoder_attention_weights"] = encoder_attention_weights
            predictions["pooling_weights"] = pooling_weights

        return predictions
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    Handles:
    1. MSE for regression (log returns)
    2. CrossEntropy for direction (3-class)
    3. BCE for binary classification (pump/dump)
    4. Volatility prediction (auxiliary task - no ground truth needed)
    5. Confidence calibration (auxiliary task - no ground truth needed)

    With configurable weights per task and horizon.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Target counts
        self.n_regression = config.targets.get_num_regression_targets()
        self.n_direction = config.targets.get_num_direction_targets()
        self.n_binary = config.targets.get_num_binary_targets()

        # Loss functions
        self.mse_loss = nn.MSELoss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        # Auxiliary loss weights
        self.volatility_loss_weight = 0.1  # Small weight for auxiliary task
        self.confidence_loss_weight = 0.1  # Small weight for auxiliary task

        # Build target index mapping
        self._build_target_mapping()
    
    def _build_target_mapping(self):
        """Build mapping from target columns to task types and horizons."""
        self.target_cols = self.config.targets.get_active_target_columns()
        
        self.regression_indices = []
        self.direction_indices = []
        self.binary_indices = []
        
        for i, col in enumerate(self.target_cols):
            if "log_return" in col:
                self.regression_indices.append(i)
            elif "direction" in col:
                self.direction_indices.append(i)
            else:
                self.binary_indices.append(i)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dict with regression, direction_logits, binary_logits
            targets: (batch, n_targets) target values
            target_mask: (batch, n_targets) True = valid target
            
        Returns:
            Total loss and dict of individual losses
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=targets.device)
        
        # 1. Regression loss (MSE)
        if self.n_regression > 0 and "regression" in predictions:
            reg_pred = predictions["regression"]  # (batch, n_regression)
            reg_targets = targets[:, self.regression_indices]
            reg_mask = target_mask[:, self.regression_indices]
            
            reg_loss = self.mse_loss(reg_pred, reg_targets)
            reg_loss = (reg_loss * reg_mask.float()).sum() / (reg_mask.sum() + 1e-8)
            
            losses["regression_loss"] = reg_loss
            total_loss = total_loss + reg_loss * self.config.targets.target_weights["log_return"]
        
        # 2. Direction loss (CrossEntropy)
        if self.n_direction > 0 and "direction_logits" in predictions:
            dir_logits = predictions["direction_logits"]  # (batch, n_direction, 3)
            dir_targets = targets[:, self.direction_indices]  # (batch, n_direction)
            dir_mask = target_mask[:, self.direction_indices]
            
            # Convert targets from {-1, 0, 1} to {0, 1, 2}
            dir_targets_idx = (dir_targets + 1).long()  # {0, 1, 2}
            
            # Compute loss per horizon
            batch_size = dir_logits.shape[0]
            dir_loss = torch.tensor(0.0, device=targets.device)
            
            for i in range(self.n_direction):
                horizon_logits = dir_logits[:, i, :]  # (batch, 3)
                horizon_targets = dir_targets_idx[:, i]  # (batch,)
                horizon_mask = dir_mask[:, i]  # (batch,)
                
                loss_i = self.ce_loss(horizon_logits, horizon_targets)
                dir_loss = dir_loss + (loss_i * horizon_mask.float()).sum() / (horizon_mask.sum() + 1e-8)
            
            dir_loss = dir_loss / self.n_direction
            losses["direction_loss"] = dir_loss
            total_loss = total_loss + dir_loss * self.config.targets.target_weights["direction"]
        
        # 3. Binary classification loss (BCE)
        if self.n_binary > 0 and "binary_logits" in predictions:
            bin_logits = predictions["binary_logits"]  # (batch, n_binary)
            bin_targets = targets[:, self.binary_indices]
            bin_mask = target_mask[:, self.binary_indices]
            
            bin_loss = self.bce_loss(bin_logits, bin_targets)
            bin_loss = (bin_loss * bin_mask.float()).sum() / (bin_mask.sum() + 1e-8)
            
            losses["binary_loss"] = bin_loss
            # Average weight for pump and dump
            pump_weight = self.config.targets.target_weights["strong_pump"]
            dump_weight = self.config.targets.target_weights["strong_dump"]
            total_loss = total_loss + bin_loss * (pump_weight + dump_weight) / 2
        
        # 4. Volatility auxiliary loss (predict realized volatility)
        if self.n_regression > 0 and "volatility" in predictions:
            vol_pred = predictions["volatility"]  # (batch, n_regression)
            reg_targets = targets[:, self.regression_indices]
            reg_mask = target_mask[:, self.regression_indices]

            # Realized volatility = absolute value of actual returns
            realized_vol = torch.abs(reg_targets)

            # MSE between predicted and realized volatility
            vol_loss = self.mse_loss(vol_pred, realized_vol)
            vol_loss = (vol_loss * reg_mask.float()).sum() / (reg_mask.sum() + 1e-8)

            losses["volatility_loss"] = vol_loss
            total_loss = total_loss + vol_loss * self.volatility_loss_weight

        # 5. Confidence auxiliary loss (calibration)
        if self.n_regression > 0 and "confidence" in predictions:
            conf_pred = predictions["confidence"]  # (batch, n_regression)
            reg_pred = predictions["regression"]
            reg_targets = targets[:, self.regression_indices]
            reg_mask = target_mask[:, self.regression_indices]

            # Ground truth: High confidence when prediction is accurate
            # Accuracy = 1 - normalized absolute error
            prediction_error = torch.abs(reg_pred - reg_targets)
            # Normalize error to [0, 1] range (clip at max error of 0.1 = 10% log return)
            max_error = 0.1
            normalized_error = torch.clamp(prediction_error / max_error, 0, 1)
            accuracy_score = 1 - normalized_error  # 1 = perfect, 0 = very wrong

            # BCE loss: confidence should match accuracy
            conf_loss = self.mse_loss(conf_pred, accuracy_score)
            conf_loss = (conf_loss * reg_mask.float()).sum() / (reg_mask.sum() + 1e-8)

            losses["confidence_loss"] = conf_loss
            total_loss = total_loss + conf_loss * self.confidence_loss_weight

        losses["total_loss"] = total_loss

        return total_loss, losses


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_model(config: Optional[Config] = None) -> TemporalTransformer:
    """Create model from config."""
    if config is None:
        config = get_config()
    
    return TemporalTransformer(config)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Test model creation
    config = get_config()
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = config.model.seq_length
    input_dim = config.features.get_total_input_dim()
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, mask, return_attention=True)
    
    print("\nTest forward pass:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} tensors")
    
    # Count parameters
    params = count_parameters(model)
    print(f"\nParameters: {params['total']:,} ({params['total']/1e6:.2f}M)")
