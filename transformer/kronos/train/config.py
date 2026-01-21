import os

class Config:
    """
    Configuration class for the entire project.
    
    This configuration is optimized for Bitget cryptocurrency data with
    return-based feature engineering for Kronos transformer training.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # Use absolute path based on this file's location
        _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.qlib_data_path = os.path.join(_base_dir, "data", "processed_datasets")
        self.instrument = None  # multi-symbol mode: not required

        # Overall time range for data loading
        self.dataset_begin_time = "2024-01-01"
        self.dataset_end_time = '2026-01-20'

        # Sliding window parameters for creating samples
        # Using 168 hours (7 days) lookback for crypto hourly data
        self.lookback_window = 168  # 7 days of hourly data for context
        self.predict_window = 8     # Predict next 8 hours for return targets
        self.max_context = 256      # Maximum context length for the model

        # =================================================================
        # Major Coins Configuration (selectable)
        # =================================================================
        # Options: ['btc', 'eth', 'sol', 'doge']
        # Currently using BTC and ETH only for dimensionality control
        self.major_coins = ['btc', 'eth']
        
        # =================================================================
        # Engineered Feature List (return-based)
        # =================================================================
        # These features are created by bitget_preprocessing.py
        self.feature_list = [
            # --- Return Features (from close price) ---
            "returns_1h",       # 1-hour return
            "returns_4h",       # 4-hour return
            "returns_24h",      # 24-hour return
            
            # --- Relative OHLC Features ---
            "hl_range",         # (high - low) / close - intrabar volatility
            "oc_range",         # (close - open) / close - bar direction
            "high_rel",         # (high - close) / close - upper wick
            "low_rel",          # (close - low) / close - lower wick
            
            # --- Volatility Features ---
            "volatility_24h",   # Rolling 24h volatility of returns
            
            # --- Volume Features ---
            "volume_rel",       # volume / MA(volume, 24h) - relative volume
            "volume_pct_change", # Volume percent change
            
            # --- BTC Reference Features ---
            "btc_returns_1h",
            "btc_returns_4h",
            "btc_returns_24h",
            "btc_volatility_24h",
            "btc_volume_rel",
            
            # --- ETH Reference Features ---
            "eth_returns_1h",
            "eth_returns_4h",
            "eth_returns_24h",
            "eth_volatility_24h",
            "eth_volume_rel",
            
            # --- Lifecycle ---
            "ts_since_listing_log",  # log(ts_since_listing + 1) - compressed scale
            
            # --- Sentiment ---
            "fng_norm",         # FNG normalized to [-1, 1] range
        ]
        
        # Target feature for return prediction
        self.target_feature = "returns_1h"  # Predict next-step 1h return

        # =================================================================
        # Fine-Tuning Options
        # =================================================================
        # If True: train CloseHead with MSE on "close" feature, backbone with tiny LR
        self.close_head_only = False              # Disabled for return prediction
        self.close_head_lr = 1e-3
        self.backbone_lr_scale = 0.01
        
        # Return prediction head configuration
        self.use_return_head = True               # Enable return prediction head
        self.return_head_lr = 1e-3                # LR for return prediction head
        self.return_loss_weight = 0.3             # Weight for return MSE in combined loss
        self.token_loss_weight = 0.7              # Weight for token prediction loss


        # Time-based features to be generated
        # Note: ts_since_listing is now a numeric feature, not a temporal embedding
        self.time_feature_list = ['hour', 'weekday']

        # Train from scratch (do not load Hugging Face models)
        self.init_tokenizer_from_pretrained = False
        self.init_predictor_from_pretrained = False

        # === Tokenizer Architecture (used if init_tokenizer_from_pretrained=False)
        self.tokenizer_arch = dict(
            d_in=len(self.feature_list),  # 22 features for Bitget data
            d_model=256,      # Reduced from 512 for faster training
            n_heads=8,        # Reduced from 16
            ff_dim=1024,      # Reduced from 2048
            n_enc_layers=4,   # Reduced from 6
            n_dec_layers=4,   # Reduced from 6
            ffn_dropout_p=0.05,
            attn_dropout_p=0.05,
            resid_dropout_p=0.05,
            s1_bits=12,       # vocab 4096 (codebook_dim=24, divisible by group_size=8)
            s2_bits=12,       # vocab 4096
            beta=0.25,
            gamma0=0.10,
            gamma=0.10,
            zeta=0.05,
            group_size=8
        )

        # === Predictor Architecture (used if init_predictor_from_pretrained=False)
        self.predictor_arch = dict(
            s1_bits=self.tokenizer_arch['s1_bits'],
            s2_bits=self.tokenizer_arch['s2_bits'],
            n_layers=6,       # Reduced from 10
            d_model=256,      # Reduced from 512
            n_heads=8,        # Reduced from 16
            ff_dim=1024,      # Reduced from 2048
            ffn_dropout_p=0.2,
            attn_dropout_p=0.05,
            resid_dropout_p=0.2,
            token_dropout_p=0.05,
            learn_te=True
        )


        # CSV input (multi-symbol enforced)
        self.use_raw_csv = True
        self.raw_csv_path = "./data/dataset/bitget_engineered.csv"  # Engineered features CSV
        self.csv_datetime_col = "timestamp"
        self.csv_symbol_col = "instrument_id"

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Train: 2024-01-01 to 2025-08-31 (~20 months)
        # Val: 2025-09-01 to 2026-01-20 (~5 months)
        self.train_time_range = ["2024-01-01", "2025-08-31"]
        self.val_time_range = ["2025-09-01", "2026-01-20"]
        self.test_time_range = ["2025-09-01", "2026-01-20"]  # Same as val for now

        # Directory to save the processed, pickled datasets (absolute path)
        self.dataset_path = os.path.join(_base_dir, "data", "processed_datasets")

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.epochs = 5
        self.log_interval = 200  # Log training status every N batches.
        self.batch_size = 20  # Batch size per GPU.

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size

        # Learning rates for different model components.
        self.tokenizer_learning_rate = 1e-4
        self.predictor_learning_rate = 1e-5

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 8

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.97
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = 100  # Global random seed for reproducibility.

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        # Weights & Biases logging (replace former Comet settings)
        self.use_wandb = True  # Enable to activate wandb logging
        self.wandb_config = {
            "project": "Kronos-ML",
            "entity": "raphaelkarmalker_development"  # TODO: set your W&B entity (team/user)
        }
        self.wandb_tag = 'training'
        self.wandb_name = 'training'

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = "./data/models"
        self.tokenizer_save_folder_name = 'tokenizer_model'
        self.predictor_save_folder_name = 'finetune_model'

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # TODO: Update these paths to your pretrained model locations.
        # These can be local paths or Hugging Face Hub model identifiers.
        #self.pretrained_tokenizer_path = f"NeoQuasar/Kronos-Tokenizer-base"
        #self.pretrained_predictor_path = f"NeoQuasar/Kronos-base"
        self.pretrained_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.pretrained_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        #self.finetuned_tokenizer_path = f"NeoQuasar/Kronos-Tokenizer-base"
       # self.finetuned_predictor_path = f"NeoQuasar/Kronos-base"


        # =================================================================
        # Inference Parameters
        # =================================================================

        self.inference_T = 0.4
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 4
        self.backtest_batch_size = 20
        # Sliding stride for evaluation/backtest windows (number of steps to shift each slide)
        self.slide_step = 1000
        # Visualization
        self.vis_plot_amount = 10
        # NEW: allow using processed pickle (test_data.pkl) instead of raw CSV
        self.inference_use_test_pickle = True          # set True to use test_data.pkl
        self.processed_test_pickle_name = "test_data.pkl"  # override if different
        # --- NEW: live plot after each inference batch ---
        self.live_plot_sliding = True
        # --- NEW: live plot handling mode & output dir ---
        self.inference_live_plot_mode = "save"          # "save" or "show"
        self.inference_plot_output_dir = "./data/inference_outputs"
        self.inference_save_analysis_plots = True  # <- NEW

        # Run full analysis every N batches during sliding inference (0 disables)
        self.inference_analyze_every_n_batches = 1  # <- NEW

        # W&B logging for inference analysis
        self.inference_analysis_wandb_log = self.use_wandb  # <- NEW: enable wandb logging of analysis if wandb is enabled
        self.inference_analysis_wandb_prefix = "analysis/"  # <- NEW: stable key prefix for overlay in W&B
        self.analysis_target_col = "close"  # <- NEW: default target for analysis

        # Dedicated W&B naming for inference runs (do not reuse training)
        self.inference_wandb_name = "inference"   # <- NEW
        self.inference_wandb_tag = "inference"    # <- NEW

        # =================================================================
        # Backtest
        # =================================================================
        self.backtest_output_dir = "./backtest_results"
        self.backtest_corr_min = 0.30          # trade horizons with corr >= threshold
        self.backtest_top_horizons = 0         # if >0 and threshold selects none, pick top-K horizons
        self.backtest_commission_bps = 0.0     # per-step commission in basis points
        # Live W&B logging during backtest
        self.backtest_use_wandb = self.use_wandb
        self.backtest_wandb_name = "backtest"
        self.backtest_wandb_tag = "backtest"
        self.backtest_log_every = 1            # log every N windows
        self.backtest_log_images = True        # include equity images in logs
        # NEW: strategy mode and stride
        self.backtest_mode = "hold"            # "hold" (hold-to-horizon) or "step" (bar-by-bar)
        self.backtest_auto_stride = True       # if True and mode=="hold": use pred_len as slide step

