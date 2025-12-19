import os
import sys
from typing import List, Tuple
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <- ensure headless, prevents blank interactive windows
import matplotlib.pyplot as plt
import random  # <- NEW
from time import gmtime, strftime  # <- NEW
# Optional Weights & Biases (for analysis logging)
try:
    import wandb  # <- NEW
except Exception:
    wandb = None

# Ensure project root is importable
sys.path.append("../")

# Use model API directly
from model.kronos import (
    Kronos,
    KronosTokenizer,
    KronosPredictor,
    calc_time_stamps,  # reuse existing util to build time features
)
from config import Config
from utils.analyser import SlidingWindowAnalyzer


def plot_prediction(kline_df, pred_df, symbol: str = None, show: bool = True, save_path: str = None):
    # ...existing code...
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    if symbol:
        fig.suptitle(f"Symbol: {symbol}", fontsize=14)
        plt.subplots_adjust(top=0.90)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    elif show and matplotlib.get_backend().lower() != "agg":  # <- avoid show() on Agg
        plt.show()
    else:
        plt.close(fig)


class CSVWindowInference:
    """
    Single-window CSV inference runner using KronosPredictor.generate.

    - Reads one CSV (single series).
    - Builds one context window and one prediction window (no sliding).
    - Auto-creates time features for x and y timestamps.
    - Normalizes x, runs AR inference, denormalizes preds, and plots.
    """

    def __init__(self, csv_path: str, datetime_col: str = "datetime", feature_cols: List[str] = None, time_cols: List[str] = None, device: str = "cuda:0",
                 tokenizer_id: str = None, model_id: str = None, max_context: int = 512, clip: float = 5.0,
                 use_test_pickle: bool = False, processed_test_path: str = None, test_symbol: str = None):
        self.csv_path = csv_path
        self.datetime_col = datetime_col
        # Use provided features or default to Config().feature_list
        self.feature_cols = feature_cols
        # Use provided time cols or default to Config().time_feature_list
        self.time_cols = time_cols 
        self.device = device
        self.tokenizer_id = tokenizer_id
        self.model_id = model_id
        self.max_context = max_context
        self.clip = clip
        self.use_test_pickle = use_test_pickle
        self.processed_test_path = processed_test_path
        self.test_symbol = test_symbol

        self._tokenizer = None
        self._model = None
        self._predictor = None

    def load_models(self):
        self._tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        self._model = Kronos.from_pretrained(self.model_id)
        self._predictor = KronosPredictor(
            self._model, self._tokenizer,
            device=self.device,
            max_context=self.max_context,
            clip=self.clip,
            feature_cols=self.feature_cols,
            time_cols=self.time_cols,
        )

    def _read_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if self.datetime_col not in df.columns:
            raise ValueError(f"datetime column '{self.datetime_col}' not found in CSV.")
        # ensure datetime index for plotting and stamp generation
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df = df.sort_values(self.datetime_col)
        df = df.set_index(self.datetime_col)
        # basic column check
        for c in self.feature_cols:
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' not found in CSV.")
        # keep only needed columns for model input (+ optional ts_since_listing if requested)
        keep_cols = list(dict.fromkeys(self.feature_cols))
        if self.time_cols and ('ts_since_listing' in self.time_cols) and ('ts_since_listing' in df.columns):
            if 'ts_since_listing' not in keep_cols:
                keep_cols.append('ts_since_listing')
        return df[keep_cols].copy()

    def _read_test_pickle(self) -> dict:
        """
        Load ALL symbols from processed test_data.pkl.
        Returns:
            dict[str, pd.DataFrame]
        """
        if not self.processed_test_path or not os.path.isfile(self.processed_test_path):
            raise FileNotFoundError(f"Processed test pickle not found: {self.processed_test_path}")
        with open(self.processed_test_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or not data:
            raise ValueError("Loaded pickle does not contain symbol dictionary.")

        out = {}
        for sym, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    print(f"[WARN] Could not convert index to datetime for symbol {sym}: {e}")
                    continue
            # Ensure all required feature columns exist
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                print(f"[WARN] Skipping symbol {sym}; missing features: {missing}")
                continue
            # keep feature_cols + optional ts_since_listing
            keep_cols = list(dict.fromkeys(self.feature_cols))
            if self.time_cols and ('ts_since_listing' in self.time_cols):
                if 'ts_since_listing' in df.columns:
                    keep_cols.append('ts_since_listing')
                else:
                    print(f"[WARN] Symbol {sym} missing 'ts_since_listing' while requested in time_cols")
            df = df[keep_cols].dropna()
            if df.empty:
                continue
            out[sym] = df
        if not out:
            raise RuntimeError("No valid symbols found in test pickle.")
        return out

    def _load_source(self) -> pd.DataFrame:
        """
        Unified loader: either CSV or processed pickle.
        """
        if self.use_test_pickle:
            return self._read_test_pickle()
        return self._read_csv()

    def _select_window(
        self, df: pd.DataFrame, lookback: int, pred_len: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.shape[0] < (lookback + pred_len):
            raise ValueError(f"Not enough rows ({df.shape[0]}) for lookback={lookback} + pred_len={pred_len}.")
        # take the last contiguous block: [-(lookback+pred_len):]
        block = df.iloc[-(lookback + pred_len):]
        x_df = block.iloc[:lookback]
        future_df = block.iloc[lookback:]  # used only for timestamps and plotting window
        return x_df, future_df

    def _iter_sliding_windows_by_dates(
        self, df: pd.DataFrame, lookback: int, pred_len: int, step: int = 1
    ):
        """
        Yields (x_df, future_df) sliding one step forward each time.
        Uses the actual available time range in df (ignores external start_date/end_date).
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        if step is None or int(step) < 1:
            raise ValueError("step must be a positive integer.")
        step = int(step)

        # Dynamisch: benutze echten Zeitraum aus df
        start_ts = df.index.min()
        end_ts = df.index.max()

        s0 = 0
        end_pos_target = len(df) - 1

        # e = s + lookback + pred_len - 1
        max_s = end_pos_target - (lookback + pred_len) + 1
        if max_s < s0:
            raise ValueError(
                f"Not enough data ({len(df)}) for lookback={lookback} + pred_len={pred_len}."
            )

        for s in range(s0, max_s + 1, step):
            x_start = s
            x_end = s + lookback
            y_end = x_end + pred_len
            x_df = df.iloc[x_start:x_end]
            future_df = df.iloc[x_end:y_end]
            yield x_df, future_df


    def run(self, lookback: int, pred_len: int, T: float = 1.0, top_k: int = 0, top_p: float = 0.9, sample_count: int = 1, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
            pred_df: predicted DataFrame with columns matching feature_cols
            kline_df: concatenated ground truth window for plotting (lookback + pred_len)
        """
        if self._predictor is None:
            self.load_models()

        df_or_dict = self._load_source()
        if isinstance(df_or_dict, dict):
            # multi-symbol: pick first symbol deterministically
            sym = sorted(df_or_dict.keys())[0]
            print(f"[INFO] Multiple symbols loaded ({len(df_or_dict)}). Using first symbol '{sym}' for single-window run().")
            df = df_or_dict[sym]
        else:
            df = df_or_dict
            sym = getattr(self, "test_symbol", None)

        x_df, future_df = self._select_window(df, lookback, pred_len)

        # Use predictor.predict which handles stamps and normalization internally
        pred_df = self._predictor.predict(
            df=x_df,  # pass x_df with optional ts_since_listing preserved
            x_timestamp=x_df.index,
            y_timestamp=future_df.index,
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        # full GT window for plotting
        kline_df = pd.concat([x_df[self.feature_cols], future_df[self.feature_cols]], axis=0)

        return pred_df, kline_df

    def run_sliding(
        self,
        lookback: int,
        pred_len: int,
        start_date: str,
        end_date: str,
        T: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
        batch_size: int = 64,
        step: int = 1,
        live_plot: bool = False,   # <- NEW
        live_plot_dir: str = None,          # <- NEW
        live_plot_mode: str = "save",       # <- NEW ("save" or "show")
        periodic_analysis_n: int | None = None  # <- NEW (run cumulative analysis every N batches)
    ):
        """
        Slide the context window by 1 step from start_date until the end of the prediction window equals end_date.
        Returns a list of dicts: [{'start': Timestamp, 'end': Timestamp, 'pred': pred_df, 'kline': kline_df}, ...]
        """
        if self._predictor is None:
            self.load_models()
        df_or_dict = self._load_source()

        # Prepare window collections
        x_df_list, x_ts_list, y_ts_list, meta = [], [], [], []

        if isinstance(df_or_dict, dict):
            # Iterate all symbols
            for sym, df in df_or_dict.items():
                try:
                    for x_df, future_df in self._iter_sliding_windows_by_dates(df, lookback, pred_len, step=step):
                        x_df_list.append(x_df)  # keep optional ts_since_listing
                        x_ts_list.append(x_df.index)
                        y_ts_list.append(future_df.index)
                        meta.append((sym, x_df.index[0], future_df.index[-1], x_df[self.feature_cols], future_df[self.feature_cols]))
                except ValueError as e:
                    print(f"[WARN] Symbol {sym} skipped: {e}")
        else:
            df = df_or_dict
            sym = getattr(self, "test_symbol", None)
            for x_df, future_df in self._iter_sliding_windows_by_dates(df, lookback, pred_len, step=step):
                x_df_list.append(x_df)  # keep optional ts_since_listing
                x_ts_list.append(x_df.index)
                y_ts_list.append(future_df.index)
                meta.append((sym, x_df.index[0], future_df.index[-1], x_df[self.feature_cols], future_df[self.feature_cols]))

        n = len(x_df_list)
        if n == 0:
            return []
        bs = max(1, min(batch_size, n))

        results = []

        # NEW: helper to run cumulative analysis and overwrite plots
        def _run_periodic_analysis(accum_results):
            if not accum_results:
                return
            try:
                cfg_local = Config()
                analysis_target_col = getattr(cfg_local, "analysis_target_col", "close")
                wandb_log_flag = getattr(cfg_local, "inference_analysis_wandb_log", False)
                wandb_prefix = getattr(cfg_local, "inference_analysis_wandb_prefix", "analysis/")
                analyzer = SlidingWindowAnalyzer(
                    target_col=analysis_target_col,
                    wandb_log=wandb_log_flag,
                    wandb_prefix=wandb_prefix
                )
                out_dir = os.path.join(live_plot_dir or ".", "analysis")
                os.makedirs(out_dir, exist_ok=True)

                mpae = analyzer.compute_mpae_bar_by_bar(accum_results)
                analyzer.plot_mpae_curve(mpae, show=False, save_path=os.path.join(out_dir, "mpae_curve.png"))

                da = analyzer.compute_directional_accuracy_bar_by_bar(accum_results)
                analyzer.plot_da_curve(da, show=False, save_path=os.path.join(out_dir, "directional_accuracy_curve.png"))

                rmse = analyzer.compute_rmse_bar_by_bar(accum_results)
                analyzer.plot_curve(rmse, ylabel="RMSE", title=f"RMSE per horizon ({analysis_target_col})",
                                    show=False, save_path=os.path.join(out_dir, "rmse_curve.png"))

                smape = analyzer.compute_smape_bar_by_bar(accum_results)
                analyzer.plot_curve(smape, ylabel="sMAPE", title=f"sMAPE per horizon ({analysis_target_col})",
                                    show=False, save_path=os.path.join(out_dir, "smape_curve.png"))

                ret_corr = analyzer.compute_return_corr_bar_by_bar(accum_results)
                analyzer.plot_curve(ret_corr, ylabel="Return correlation", title=f"Return Corr per horizon ({analysis_target_col})",
                                    show=False, save_path=os.path.join(out_dir, "return_corr_curve.png"))

                # --- NEW: R² per horizon (periodic) ---
                r2 = analyzer.compute_r2_bar_by_bar(accum_results)
                analyzer.plot_curve(r2, ylabel="R²", title=f"R² per horizon ({analysis_target_col})",
                                    show=False, save_path=os.path.join(out_dir, "r2_curve.png"))
                # --------------------------------------

                # --- NEW: MSPE per horizon (periodic) ---
                mspe = analyzer.compute_mspe_bar_by_bar(accum_results)
                analyzer.plot_curve(mspe, ylabel="VSE", title=f"VSE per horizon ({analysis_target_col})",
                                    show=False, save_path=os.path.join(out_dir, "vse_curve.png"))
                # ----------------------------------------

            except Exception as e:
                print(f"[WARN] Periodic analysis failed: {e}")

        # Batch inference
        def _sanitize(name: str):
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))

        for i in range(0, n, bs):
            j = min(i + bs, n)
            chunk_preds = self._predictor.predict_batch(
                df_list=x_df_list,  # pass DataFrames with optional ts_since_listing
                x_timestamp_list=x_ts_list[i:j],
                y_timestamp_list=y_ts_list[i:j],
                pred_len=pred_len,
                T=T,
                top_k=top_k,
                top_p=top_p,
                sample_count=sample_count,
                verbose=verbose,
            )
            # Assemble results
            batch_results = []  # <- NEW (collect this batch for optional live plot)
            for k, pred_df in enumerate(chunk_preds):
                sym, start_ts, end_ts, x_part, y_part = meta[i + k]
                kline_df = pd.concat([x_part, y_part], axis=0)
                item = {
                    'symbol': sym,
                    'start': start_ts,
                    'end': end_ts,
                    'pred': pred_df,
                    'kline': kline_df,
                }
                results.append(item)
                batch_results.append(item)
            if live_plot and batch_results:
                pick = random.choice(batch_results)
                if live_plot_mode == "save":
                    if live_plot_dir:
                        fn = f"{_sanitize(pick.get('symbol'))}_{_sanitize(pick['start'])}_{_sanitize(pick['end'])}.png"
                        save_path = os.path.join(live_plot_dir, fn)
                    else:
                        save_path = None
                    try:
                        plot_prediction(pick['kline'], pick['pred'],
                                        symbol=pick.get('symbol'),
                                        show=False,
                                        save_path=save_path)
                    except Exception as e:
                        print(f"[WARN] live_plot save failed: {e}")
                else:
                    try:
                        plot_prediction(pick['kline'], pick['pred'], symbol=pick.get('symbol'), show=True)
                    except Exception as e:
                        print(f"[WARN] live_plot show failed: {e}")

            # NEW: run cumulative analysis every N batches (overwrite plots)
            if periodic_analysis_n and periodic_analysis_n > 0:
                batch_idx = (i // bs)  # 0-based
                if ((batch_idx + 1) % periodic_analysis_n) == 0:
                    _run_periodic_analysis(results)

        return results

    def visualize(self, kline_df: pd.DataFrame, pred_df: pd.DataFrame, symbol: str = None):
        plot_prediction(kline_df, pred_df, symbol=symbol, show=True)

    def visualize_sliding(self, slide_results, amount: int):
        """
        Plot `amount` windows approximately evenly spaced across slide_results.
        """
        if not slide_results or amount is None or amount <= 0:
            return
        n = len(slide_results)
        amount = min(amount, n)
        # pick roughly even indices
        idxs = np.unique(np.round(np.linspace(0, n - 1, num=amount)).astype(int))
        for i in idxs:
            item = slide_results[i]
            self.visualize(item['kline'], item['pred'], symbol=item.get('symbol'))


def main():
    # ---- Edit these values directly ----
    cfg = Config()
    csv_path = cfg.raw_csv_path
    datetime_col = cfg.csv_datetime_col
    device = "cuda:0"
    lookback = cfg.lookback_window
    pred_len = cfg.predict_window
    max_context = cfg.max_context
    clip = cfg.clip
    T = cfg.inference_T
    top_k = cfg.inference_top_k
    top_p = cfg.inference_top_p
    sample_count = cfg.inference_sample_count
    SHOW_PLOT = True
    test_start, test_end = cfg.test_time_range
    plot_amount = getattr(cfg, "vis_plot_amount", 10)
    slide_batch_size = getattr(cfg, "backtest_batch_size", 64)
    slide_step = getattr(cfg, "slide_step", 1)
    analysis_target_col = getattr(cfg, "analysis_target_col", "close")
    live_plot = getattr(cfg, "live_plot_sliding", False)  # <- NEW
    live_plot_mode = getattr(cfg, "inference_live_plot_mode", "save")
    live_plot_dir = getattr(cfg, "inference_plot_output_dir", "./inference_outputs")
    save_analysis = getattr(cfg, "inference_save_analysis_plots", False)  # <- NEW
    periodic_analysis_n = getattr(cfg, "inference_analyze_every_n_batches", 0)  # <- NEW
    # ------------------------------------

    tokenizer_id = cfg.finetuned_tokenizer_path
    model_id = cfg.finetuned_predictor_path

    # Init W&B if analysis logging is enabled (use dedicated inference name/tag)
    wandb_run = None  # <- NEW
    if getattr(cfg, "inference_analysis_wandb_log", False) and getattr(cfg, "use_wandb", False) and (wandb is not None):
        try:
            if getattr(wandb, "run", None) is None:
                base_name = getattr(cfg, "inference_wandb_name", "inference")
                ts = strftime("%Y%m%d_%H%M%S", gmtime())
                run_name = f"{base_name}-{ts}"  # <- NEW: dynamic name
                wandb_run = wandb.init(
                    project=cfg.wandb_config.get("project"),
                    entity=cfg.wandb_config.get("entity"),
                    name=run_name,  # <- CHANGED
                    tags=[getattr(cfg, "inference_wandb_tag", "inference")]
                )
                # Push config for reproducibility
                wandb.config.update(cfg.__dict__, allow_val_change=True)
                print(f"Weights & Biases (inference) initialized: {run_name}")
        except Exception as e:
            print(f"[WARN] Failed to init W&B for inference: {e}")

    # Pre-create plotting directories so periodic plots appear immediately
    os.makedirs(live_plot_dir, exist_ok=True)  # <- NEW
    analysis_dir = os.path.join(live_plot_dir, "analysis")  # <- NEW
    os.makedirs(analysis_dir, exist_ok=True)  # <- NEW

    # Decide data source
    use_pickle = getattr(cfg, "inference_use_test_pickle", False)
    processed_test_path = os.path.join(cfg.dataset_path, getattr(cfg, "processed_test_pickle_name", "test_data.pkl"))
    runner = CSVWindowInference(
        csv_path=cfg.raw_csv_path,
        datetime_col=datetime_col,
        feature_cols=cfg.feature_list,
        time_cols=cfg.time_feature_list,
        device=device,
        tokenizer_id=tokenizer_id,
        model_id=model_id,
        max_context=max_context,
        clip=clip,
        use_test_pickle=use_pickle,
        processed_test_path=processed_test_path,
        test_symbol=None,  # multi-symbol mode: not needed
    )

    # Sliding over test range
    slide_results = runner.run_sliding(
        lookback=lookback,
        pred_len=pred_len,
        start_date=test_start,
        end_date=test_end,
        T=T,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=True,
        batch_size=slide_batch_size,
        step=slide_step,
        live_plot=live_plot,
        live_plot_dir=live_plot_dir,
        live_plot_mode=live_plot_mode,
        periodic_analysis_n=periodic_analysis_n if periodic_analysis_n > 0 else None  # <- NEW
    )
    print(f"Generated {len(slide_results)} sliding windows.")
    if slide_results and SHOW_PLOT:
        runner.visualize_sliding(slide_results, amount=plot_amount)
        analyzer = SlidingWindowAnalyzer(
            target_col=analysis_target_col,
            wandb_log=getattr(cfg, "inference_analysis_wandb_log", False),          # <- NEW
            wandb_prefix=getattr(cfg, "inference_analysis_wandb_prefix", "analysis/")  # <- NEW
        )

        # use the pre-created analysis_dir for final plots as well
        show_analysis = SHOW_PLOT and (matplotlib.get_backend().lower() != "agg")

        # MPAE
        mpae = analyzer.compute_mpae_bar_by_bar(slide_results)
        analyzer.plot_mpae_curve(
            mpae,
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "mpae_curve.png") if save_analysis else None
        )

        # Directional Accuracy
        da = analyzer.compute_directional_accuracy_bar_by_bar(slide_results)
        analyzer.plot_da_curve(
            da,
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "directional_accuracy_curve.png") if save_analysis else None
        )

        # RMSE
        rmse = analyzer.compute_rmse_bar_by_bar(slide_results)
        analyzer.plot_curve(
            rmse, ylabel="RMSE", title=f"RMSE per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "rmse_curve.png") if save_analysis else None
        )

        # sMAPE
        smape = analyzer.compute_smape_bar_by_bar(slide_results)
        analyzer.plot_curve(
            smape, ylabel="sMAPE", title=f"sMAPE per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "smape_curve.png") if save_analysis else None
        )

        # Return correlation
        ret_corr = analyzer.compute_return_corr_bar_by_bar(slide_results)
        analyzer.plot_curve(
            ret_corr, ylabel="Return correlation", title=f"Return Corr per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "return_corr_curve.png") if save_analysis else None
        )

        # R²
        r2 = analyzer.compute_r2_bar_by_bar(slide_results)
        analyzer.plot_curve(
            r2, ylabel="R²", title=f"R² per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "r2_curve.png") if save_analysis else None
        )

        # MSPE
        mspe = analyzer.compute_mspe_bar_by_bar(slide_results)
        analyzer.plot_curve(
            mspe, ylabel="VSE", title=f"VSE per horizon ({analysis_target_col})",
            show=show_analysis,
            save_path=os.path.join(analysis_dir, "vse_curve.png") if save_analysis else None
        )

    # Finish W&B run if we started it
    if wandb_run is not None:  # <- NEW
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()