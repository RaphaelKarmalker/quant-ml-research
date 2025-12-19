import os
import sys
import json
from time import gmtime, strftime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project root
sys.path.append("../")

from config import Config
from utils.analyser import SlidingWindowAnalyzer
from target_inference import CSVWindowInferenceTarget

try:
    import wandb
except Exception:
    wandb = None


def _pct_ret(curr, prev, eps=1e-12):
    prev = np.where(np.abs(prev) < eps, eps, prev)
    return (curr - prev) / prev


def _select_horizons(corr_series, corr_min: float | None, top_k: int | None):
    # Accept pd.Series or np.ndarray
    vals = np.asarray(corr_series, dtype=float).reshape(-1)
    idxs = np.arange(len(vals))
    if corr_min is not None:
        sel = idxs[vals >= corr_min]
        if sel.size > 0:
            return sel
    if top_k is not None and top_k > 0:
        order = np.argsort(-vals)  # descending
        sel = order[:min(top_k, len(order))]
        return np.sort(sel)
    return idxs


def _plot_equity(equity: np.ndarray, save_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(equity, label="Equity", color="black")
    plt.xlabel("Trade index")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def run_backtest():
    cfg = Config()
    # Settings
    lookback = cfg.lookback_window
    pred_len = cfg.predict_window
    device = "cuda:0"
    # thresholds for horizon selection
    corr_min = float(getattr(cfg, "backtest_corr_min", 0.30))
    top_k_h = int(getattr(cfg, "backtest_top_horizons", 0)) or None
    # trading costs
    commission_bps = float(getattr(cfg, "backtest_commission_bps", 0.0))
    # strategy mode
    mode = getattr(cfg, "backtest_mode", "hold")  # NEW: "hold" or "step"
    # inference params
    T = cfg.inference_T
    top_k = cfg.inference_top_k
    top_p = cfg.inference_top_p
    sample_count = cfg.inference_sample_count
    slide_step = getattr(cfg, "slide_step", 1)
    # use pred_len stride for hold-to-horizon to avoid overlap if enabled
    auto_stride = bool(getattr(cfg, "backtest_auto_stride", True))
    effective_step = (pred_len if (mode == "hold" and auto_stride) else slide_step)  # NEW
    # data/source
    tokenizer_id = cfg.finetuned_tokenizer_path
    model_id = getattr(cfg, "finetuned_target_predictor_path", cfg.finetuned_predictor_path)
    use_pickle = getattr(cfg, "inference_use_test_pickle", True)
    processed_test_path = os.path.join(cfg.dataset_path, getattr(cfg, "processed_test_pickle_name", "test_data.pkl"))
    # columns
    feature_cols = cfg.feature_list
    time_cols = cfg.time_feature_list
    target_col = getattr(cfg, "analysis_target_col", "close")

    # Output dir
    ts = strftime("%Y%m%d_%H%M%S", gmtime())
    out_dir = os.path.abspath(getattr(cfg, "backtest_output_dir", "./backtest_results"))
    run_dir = os.path.join(out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Optional W&B init for backtest live logs
    wandb_run = None
    if getattr(cfg, "backtest_use_wandb", False) and getattr(cfg, "use_wandb", False) and (wandb is not None):
        try:
            wandb_run = wandb.init(
                project=cfg.wandb_config.get("project"),
                entity=cfg.wandb_config.get("entity"),
                name=f"{getattr(cfg, 'backtest_wandb_name', 'backtest')}-{ts}",
                tags=[getattr(cfg, "backtest_wandb_tag", "backtest")]
            )
            wandb.config.update({
                "lookback": lookback, "pred_len": pred_len,
                "corr_min": corr_min, "top_k_h": top_k_h,
                "commission_bps": commission_bps,
                "slide_step": slide_step
            }, allow_val_change=True)
        except Exception as e:
            print(f"[WARN] W&B init failed for backtest: {e}")
            wandb_run = None

    # 1) Inference: slide over the entire test set (all symbols if provided)
    runner = CSVWindowInferenceTarget(
        csv_path=cfg.raw_csv_path,
        datetime_col=cfg.csv_datetime_col,
        feature_cols=feature_cols,
        time_cols=time_cols,
        device=device,
        tokenizer_id=tokenizer_id,
        model_id=model_id,
        max_context=cfg.max_context,
        clip=cfg.clip,
        use_test_pickle=use_pickle,
        processed_test_path=processed_test_path,
        test_symbol=None,
        target_col=target_col,
    )

    slide_results = runner.run_sliding(
        lookback=lookback,
        pred_len=pred_len,
        start_date=cfg.test_time_range[0],
        end_date=cfg.test_time_range[1],
        T=T,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
        batch_size=getattr(cfg, "backtest_batch_size", 64),
        step=effective_step,  # CHANGED: use effective_step
        live_plot=False,
        live_plot_dir=None,
        live_plot_mode="save",
        periodic_analysis_n=None
    )
    if not slide_results:
        print("[Backtest] No sliding results generated.")
        return

    # 2) Correlation by horizon
    analyzer = SlidingWindowAnalyzer(
        target_col=target_col,
        wandb_log=False,
        wandb_prefix="analysis/"
    )
    corr_ser = analyzer.compute_return_corr_bar_by_bar(slide_results)  # index 1..H
    corr_csv = os.path.join(run_dir, "return_corr_by_horizon.csv")
    corr_ser.to_csv(corr_csv)
    # Optional: log correlation as a W&B table (no image)
    try:
        if wandb_run is not None:
            corr_table = wandb.Table(columns=["horizon", "return_corr"])
            for h, v in zip(corr_ser.index.astype(int), corr_ser.values.astype(float)):
                corr_table.add_data(int(h), float(v))
            wandb.log({"analysis/return_corr_by_horizon": corr_table})
    except Exception as e:
        print(f"[WARN] Could not log correlation table: {e}")

    # Select 0-based horizons to trade
    sel_horizons = _select_horizons(corr_ser.values, corr_min=corr_min, top_k=top_k_h)
    if sel_horizons.size == 0:
        print("[Backtest] No horizons selected by correlation; falling back to all horizons.")
        sel_horizons = np.arange(pred_len)

    # For hold mode: choose a single best horizon h* among selected horizons
    hold_h_star = None  # NEW
    if mode == "hold":
        corr_vals = np.asarray(corr_ser.values, dtype=float).reshape(-1)
        # pick argmax among selected horizons (ignore NaNs safely)
        valid_mask = ~np.isnan(corr_vals[sel_horizons])
        if valid_mask.any():
            best_idx = np.argmax(corr_vals[sel_horizons][valid_mask])
            hold_h_star = int(sel_horizons[valid_mask][best_idx])
        else:
            hold_h_star = int(np.nanargmax(corr_vals))  # fallback

    # save selection (+ mode and h*)
    with open(os.path.join(run_dir, "horizon_selection.json"), "w") as f:
        json.dump({
            "mode": mode,
            "corr_min": corr_min, "top_k_horizons": top_k_h,
            "selected_horizons_0based": sel_horizons.tolist(),
            "hold_h_star_0based": hold_h_star,
            "corr_values": corr_ser.values.tolist()
        }, f, indent=2)
    if wandb_run is not None:
        payload = {
            "analysis/mode": mode,
            "analysis/selected_horizons_0based": sel_horizons.tolist(),
            "analysis/corr_values": corr_ser.values.tolist()
        }
        if hold_h_star is not None:
            payload["analysis/hold_h_star_0based"] = hold_h_star
        wandb.log(payload)

    # 3) Strategy
    trades = []
    bps2frac = commission_bps / 10000.0
    global_equity = [1.0]
    prev_pos = 0  # used only in step mode

    log_every = int(getattr(cfg, "backtest_log_every", 1))
    for idx_item, item in enumerate(slide_results):
        sym = item.get("symbol", "UNKNOWN")
        kline: pd.DataFrame = item["kline"]
        if target_col not in kline.columns:
            continue
        full = kline[target_col].values.astype(float)
        if len(full) < (lookback + pred_len):
            continue
        p0 = float(full[-pred_len - 1])      # anchor (entry price)
        gt_y = full[-pred_len:]              # GT future closes
        pred_y = item["pred"][target_col].values.astype(float)
        if len(pred_y) != pred_len:
            continue

        if mode == "hold":
            # Trade a single horizon h* per window
            h = int(hold_h_star if hold_h_star is not None else 0)
            h = max(0, min(pred_len - 1, h))
            pred_ret = _pct_ret(pred_y[h], p0)  # returns vs entry
            gt_ret = _pct_ret(gt_y[h], p0)

            # optional: skip tiny signals below cost
            signal = 1 if pred_ret > 0 else (-1 if pred_ret < 0 else 0)
            if abs(pred_ret) <= bps2frac:
                signal = 0

            # round-trip cost (entry + exit)
            cost = (2.0 * bps2frac) if signal != 0 else 0.0
            pnl = signal * gt_ret - cost

            trades.append({
                "mode": mode,
                "symbol": sym,
                "window_start": item["start"],
                "window_end": item["end"],
                "horizon": h + 1,
                "pred": float(pred_y[h]),
                "gt": float(gt_y[h]),
                "pred_ret": float(pred_ret),
                "gt_ret": float(gt_ret),
                "signal": int(signal),
                "cost_frac": float(cost),
                "pnl": float(pnl),
            })
            global_equity.append(global_equity[-1] * (1.0 + pnl))
            # remain flat after each window
            prev_pos = 0

        else:
            # step mode: bar-by-bar trading (existing behavior)
            pred_prev = p0
            gt_prev = p0
            for h in range(pred_len):
                if h not in set(sel_horizons):
                    pred_prev = pred_y[h]
                    gt_prev = gt_y[h]
                    continue

                pred_ret = _pct_ret(pred_y[h], pred_prev)
                gt_ret = _pct_ret(gt_y[h], gt_prev)

                if pred_ret > 0:
                    signal = 1
                elif pred_ret < 0:
                    signal = -1
                else:
                    signal = 0

                turnover = abs(signal - prev_pos) / 2.0
                cost = turnover * bps2frac
                pnl = signal * gt_ret - cost

                trades.append({
                    "mode": mode,
                    "symbol": sym,
                    "window_start": item["start"],
                    "window_end": item["end"],
                    "horizon": h + 1,
                    "pred": float(pred_y[h]),
                    "gt": float(gt_y[h]),
                    "pred_ret": float(pred_ret),
                    "gt_ret": float(gt_ret),
                    "signal": int(signal),
                    "turnover": float(turnover),
                    "cost_frac": float(cost),
                    "pnl": float(pnl),
                })
                global_equity.append(global_equity[-1] * (1.0 + pnl))
                prev_pos = signal
                pred_prev = pred_y[h]
                gt_prev = gt_y[h]
            prev_pos = 0  # reset between windows

        # Live W&B logging per window (table)
        if wandb_run is not None and ((idx_item + 1) % max(1, log_every) == 0):
            try:
                equity = np.asarray(global_equity, dtype=float)
                pnl_arr = np.array([t["pnl"] for t in trades], dtype=float) if trades else np.array([0.0])

                eq_table = wandb.Table(columns=["trade_idx", "equity"])
                for ti, ev in enumerate(equity):
                    eq_table.add_data(int(ti), float(ev))

                payload = {
                    "backtest/step": idx_item + 1,
                    "backtest/mode": mode,
                    "backtest/n_trades": len(trades),
                    "backtest/equity": float(equity[-1]),
                    "backtest/total_return": float(equity[-1] - 1.0),
                    "backtest/mean_pnl": float(pnl_arr.mean()),
                    "backtest/equity_table": eq_table
                }
                if mode == "hold" and hold_h_star is not None:
                    payload["backtest/hold_h_star_0based"] = hold_h_star
                wandb.log(payload, step=idx_item + 1)
            except Exception as e:
                print(f"[WARN] W&B live table log failed: {e}")

    if not trades:
        print("[Backtest] No trades generated.")
        if wandb_run is not None:
            try: wandb_run.finish()
            except Exception: pass
        return

    trades_df = pd.DataFrame(trades)
    trades_csv = os.path.join(run_dir, "trades.csv")
    trades_df.to_csv(trades_csv, index=False)

    equity = np.asarray(global_equity, dtype=float)
    final_eq_path = os.path.join(run_dir, "equity.png")
    _plot_equity(equity, final_eq_path)  # keep local file if you want it on disk

    # 4) Summary
    total_return = equity[-1] - 1.0
    pnl_arr = trades_df["pnl"].values.astype(float)
    mean = pnl_arr.mean()
    std = pnl_arr.std(ddof=1) if len(pnl_arr) > 1 else 0.0
    sharpe = (mean / std) * np.sqrt(252) if std > 0 else float("nan")
    hit_rate = float((np.sign(trades_df["pnl"]) == np.sign(trades_df["gt_ret"])).mean())

    summary = {
        "n_trades": int(len(trades_df)),
        "total_return": float(total_return),
        "mean_pnl": float(mean),
        "std_pnl": float(std),
        "sharpe_like": float(sharpe),
        "hit_rate": hit_rate,
        "commission_bps": commission_bps,
        "selected_horizons_0based": sel_horizons.tolist(),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Final W&B logging (tables only; no images)
    if wandb_run is not None:
        try:
            final_eq_table = wandb.Table(columns=["trade_idx", "equity"])
            for ti, ev in enumerate(equity):
                final_eq_table.add_data(int(ti), float(ev))

            # Optional: cap number of rows to avoid huge tables
            max_tr_rows = int(getattr(cfg, "backtest_max_trades_table_rows", 2000))
            tr_cols = list(trades_df.columns)
            trades_table = wandb.Table(columns=tr_cols)
            for _, row in trades_df.head(max_tr_rows).iterrows():
                trades_table.add_data(*[row[c] for c in tr_cols])

            wandb.log({
                "backtest/final_equity": float(equity[-1]),
                "backtest/final_total_return": float(total_return),
                "backtest/final_sharpe_like": float(sharpe),
                "backtest/final_hit_rate": float(hit_rate),
                "backtest/final_equity_table": final_eq_table,
                "backtest/trades_table": trades_table
            })
            # attach artifacts (CSV and local plot file kept as artifacts only)
            art = wandb.Artifact(f"backtest_run_{ts}", type="backtest")
            art.add_file(trades_csv)
            art.add_file(corr_csv)
            art.add_file(final_eq_path)
            wandb.log_artifact(art)
            wandb_run.finish()
        except Exception as e:
            print(f"[WARN] W&B finalize (tables) failed: {e}")

    print(f"[Backtest] Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    run_backtest()
