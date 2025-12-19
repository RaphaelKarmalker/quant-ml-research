import os
import sys
import json
import time
from time import gmtime, strftime
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

# Ensure project root is importable
sys.path.append("../")

from config import Config
from dataset import QlibDataset
from model.kronos import Kronos, KronosTokenizer
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
    safe_barrier,
)


class CloseHead(nn.Module):
    """Simple linear head to predict a single target (close) from hidden states."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, d_model] -> [B, T]
        return self.proj(h).squeeze(-1)


class KronosCloseFinetuner(nn.Module):
    """
    Wrap a Kronos predictor and add a linear head to regress the 'close' value.
    Uses Kronos embedding + time embedding + transformer + norm as backbone.
    """
    def __init__(self, base: Kronos, head: CloseHead):
        super().__init__()
        self.base = base
        self.head = head

    def forward(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: torch.Tensor = None,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Embed tokens + time, run backbone, output hidden states
        x = self.base.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.base.time_emb(stamp)
        x = self.base.token_drop(x)
        for layer in self.base.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.base.norm(x)
        # Linear projection to single target
        return self.head(x)  # [B, T]

    def backbone_parameters(self):
        return self.base.parameters()

    def close_head_parameters(self):
        return self.head.parameters()


def create_dataloaders(config: dict, rank: int, world_size: int):
    print(f"[Rank {rank}] Creating distributed dataloaders...")
    train_dataset = QlibDataset('train')
    valid_dataset = QlibDataset('val')
    print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=False
    )
    print(f"[Rank {rank}] Dataloaders created. Train steps/epoch: {len(train_loader)}, Val steps: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset


def _build_model_and_tokenizer(config: dict, device: torch.device, rank: int):
    # Tokenizer for on-the-fly tokenization
    tokenizer = KronosTokenizer.from_pretrained(config['finetuned_tokenizer_path']).to(device)
    tokenizer.eval()

    # Kronos predictor backbone
    if config.get('init_predictor_from_pretrained', True):
        if rank == 0:
            print("[Predictor Init] Loading pretrained predictor:", config['pretrained_predictor_path'])
        base_model = Kronos.from_pretrained(config['pretrained_predictor_path'])
    else:
        arch = config['predictor_arch']
        if rank == 0:
            print("[Predictor Init] Initializing predictor from scratch with arch:", arch)
        base_model = Kronos(
            arch['s1_bits'], arch['s2_bits'], arch['n_layers'], arch['d_model'],
            arch['n_heads'], arch['ff_dim'],
            arch['ffn_dropout_p'], arch['attn_dropout_p'], arch['resid_dropout_p'],
            arch['token_dropout_p'], arch['learn_te'],
            time_feature_list=config.get('time_feature_list')
        )
    base_model = base_model.to(device)

    # Mode flags (from Config with fallback to config dict)
    cfg = Config()
    close_head_only = bool(getattr(cfg, "close_head_only", config.get("close_head_only", False)))

    if close_head_only:
        # Wrap with CloseHead
        close_head = CloseHead(base_model.d_model).to(device)
        model = KronosCloseFinetuner(base_model, close_head)
        mode = "close_head_only"
    else:
        model = base_model
        mode = "full_token_prediction"

    return tokenizer, model, mode


def _build_optimizer_and_scheduler(
    model: nn.Module,
    mode: str,
    config: dict,
    steps_per_epoch: int
):
    cfg = Config()
    # Defaults as required
    close_head_lr = float(getattr(cfg, "close_head_lr", config.get("close_head_lr", 1e-3)))
    backbone_lr_scale = float(getattr(cfg, "backbone_lr_scale", config.get("backbone_lr_scale", 0.01)))

    if mode == "close_head_only":
        # Resolve module reference (DDP or not)
        head_ref = model.module.head if hasattr(model, "module") else model.head
        base_ref = model.module.base if hasattr(model, "module") else model.base

        params_head = list(head_ref.parameters())
        params_backbone = list(base_ref.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": params_head, "lr": close_head_lr},
                {"params": params_backbone, "lr": close_head_lr * backbone_lr_scale},
            ],
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.95)),
            weight_decay=config.get('adam_weight_decay', 0.01),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=[close_head_lr, close_head_lr * backbone_lr_scale],
            steps_per_epoch=steps_per_epoch,
            epochs=config['epochs'],
            pct_start=0.03,
            div_factor=10,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['predictor_learning_rate'],
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.95)),
            weight_decay=config.get('adam_weight_decay', 0.01),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config['predictor_learning_rate'],
            steps_per_epoch=steps_per_epoch,
            epochs=config['epochs'],
            pct_start=0.03,
            div_factor=10,
        )

    return optimizer, scheduler


def train_model(
    model: nn.Module,
    tokenizer: KronosTokenizer,
    device: torch.device,
    config: dict,
    save_dir: str,
    logger,
    rank: int,
    world_size: int,
    mode: str
):
    """
    Two modes:
    - close_head_only: MSE on next-step 'close' using new CloseHead while updating backbone with small LR.
    - full_token_prediction: original CE over both token heads.
    """
    start_time = time.time()
    if rank == 0:
        accumulation_steps = int(config.get('accumulation_steps', 1))
        effective_bs = config['batch_size'] * world_size * accumulation_steps
        print(f"[Rank {rank}] Mode: {mode}")
        print(f"[Rank {rank}] BATCHSIZE (per GPU): {config['batch_size']}, Accumulation: {accumulation_steps}, Effective total: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)
    ddp_active = torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1

    optimizer, scheduler = _build_optimizer_and_scheduler(model, mode, config, steps_per_epoch=len(train_loader))

    cfg = Config()
    accumulation_steps = int(config.get('accumulation_steps', 1))
    feature_list: List[str] = getattr(cfg, "feature_list", config.get("feature_list"))
    close_col_name = getattr(cfg, "target_close_col", "close")
    if feature_list is None:
        raise ValueError("feature_list must be defined in Config or passed via config.")
    if close_col_name not in feature_list:
        raise ValueError(f"close column '{close_col_name}' not found in feature_list {feature_list}.")
    close_idx = feature_list.index(close_col_name)

    best_val_loss = float('inf')
    batch_idx_global = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch_idx)

        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)

        for i, (ori_batch_x, ori_batch_x_stamp) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
            ori_batch_x_stamp = ori_batch_x_stamp.squeeze(0).to(device, non_blocking=True)

            current_batch_total_loss = 0.0
            for j in range(accumulation_steps):
                start_idx = j * (ori_batch_x.shape[0] // accumulation_steps)
                end_idx = (j + 1) * (ori_batch_x.shape[0] // accumulation_steps)
                batch_x = ori_batch_x[start_idx:end_idx]
                batch_x_stamp = ori_batch_x_stamp[start_idx:end_idx]

                with torch.no_grad():
                    token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

                # Teacher-forced next-step setup
                token_in_0 = token_seq_0[:, :-1]
                token_in_1 = token_seq_1[:, :-1]
                token_out_0 = token_seq_0[:, 1:]
                token_out_1 = token_seq_1[:, 1:]
                stamp_in = batch_x_stamp[:, :-1, :]

                if mode == "close_head_only":
                    preds = model(token_in_0, token_in_1, stamp_in)  # [B, T-1]
                    target_close = batch_x[:, 1:, close_idx]         # [B, T-1]
                    loss = F.mse_loss(preds, target_close)
                else:
                    logits = model(token_in_0, token_in_1, stamp_in)
                    head_ref = model.module.head if hasattr(model, "module") else model.head
                    loss, s1_loss, s2_loss = head_ref.compute_loss(logits[0], logits[1], token_out_0, token_out_1)

                loss_scaled = loss / accumulation_steps
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if rank == 0 and (batch_idx_global + 1) % config['log_interval'] == 0:
                lrs = [pg['lr'] for pg in optimizer.param_groups]
                lr_str = ", ".join(f"{lr:.6f}" for lr in lrs)
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LRs [{lr_str}], Loss: {current_batch_total_loss/accumulation_steps:.4f}"
                )
            if rank == 0 and logger:
                log_dict = {
                    'train_loss_batch': current_batch_total_loss / accumulation_steps,
                    'lr_group_0': optimizer.param_groups[0]['lr'],
                }
                if len(optimizer.param_groups) > 1:
                    log_dict['lr_group_1'] = optimizer.param_groups[1]['lr']
                wandb.log(log_dict, step=batch_idx_global)

            batch_idx_global += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in_0 = token_seq_0[:, :-1]
                token_in_1 = token_seq_1[:, :-1]
                token_out_0 = token_seq_0[:, 1:]
                token_out_1 = token_seq_1[:, 1:]
                stamp_in = batch_x_stamp[:, :-1, :]

                if mode == "close_head_only":
                    preds = model(token_in_0, token_in_1, stamp_in)
                    target_close = batch_x[:, 1:, close_idx]
                    val_loss = F.mse_loss(preds, target_close)
                else:
                    logits = model(token_in_0, token_in_1, stamp_in)
                    head_ref = model.module.head if hasattr(model, "module") else model.head
                    val_loss, _, _ = head_ref.compute_loss(logits[0], logits[1], token_out_0, token_out_1)

                tot_val_loss_sum_rank += val_loss.item()
                val_batches_processed_rank += 1

        if ddp_active:
            val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
            val_batches_tensor = torch.tensor(val_batches_processed_rank, device=device)
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_sum_tensor.item() / max(1, val_batches_tensor.item())
        else:
            avg_val_loss = tot_val_loss_sum_rank / max(val_batches_processed_rank, 1)

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                wandb.log({'val_loss_epoch': avg_val_loss, 'epoch': epoch_idx})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                os.makedirs(save_path, exist_ok=True)
                # Save base Kronos
                base_to_save = (model.module.base if hasattr(model, "module") and hasattr(model.module, "base")
                                else model.base if hasattr(model, "base")
                                else model.module if hasattr(model, "module") else model)
                base_to_save.save_pretrained(save_path)

                # Save CloseHead if in close_head_only mode
                if mode == "close_head_only":
                    head_ref = model.module.head if hasattr(model, "module") else model.head
                    torch.save(head_ref.state_dict(), os.path.join(save_path, "close_head.pt"))
                    meta = {
                        "feature_list": feature_list,
                        "close_col": close_col_name,
                        "close_index": close_idx,
                        "d_model": base_to_save.d_model,
                        "mode": mode,
                    }
                    with open(os.path.join(save_path, "close_head_meta.json"), "w") as f:
                        json.dump(meta, f, indent=2)

                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                if logger:
                    artifact_name = "best_model_close_finetune" if mode == "close_head_only" else "best_model_predictor_full"
                    artifact = wandb.Artifact(artifact_name, type="model")
                    artifact.add_dir(save_path)
                    wandb.log_artifact(artifact)

        safe_barrier()

    return {'best_val_loss': best_val_loss}


def main(config: dict):
    # DDP setup
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'], rank)

    # Flags with defaults
    cfg = Config()
    close_head_only = bool(getattr(cfg, "close_head_only", config.get("close_head_only", False)))
    close_head_lr = float(getattr(cfg, "close_head_lr", config.get("close_head_lr", 1e-3)))
    backbone_lr_scale = float(getattr(cfg, "backbone_lr_scale", config.get("backbone_lr_scale", 0.01)))

    save_dir = os.path.join(config['save_path'], config.get('predictor_save_folder_name', 'predictor_finetune'))
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Logger and summary (master only)
    wandb_run, master_summary = None, {}
    if rank == 0:
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
            'mode': 'close_head_only' if close_head_only else 'full_token_prediction',
            'close_head_lr': close_head_lr,
            'backbone_lr_scale': backbone_lr_scale,
        }
        if config.get('use_wandb', False):
            wandb_run = wandb.init(
                project=config['wandb_config']['project'],
                entity=config['wandb_config'].get('entity'),
                name=config.get('wandb_name', 'kronos-ft-close'),
                tags=[config.get('wandb_tag', 'finetune')]
            )
            wandb.config.update({**config, "close_head_only": close_head_only, "close_head_lr": close_head_lr, "backbone_lr_scale": backbone_lr_scale})
            print("Weights & Biases Logger Initialized.")
    safe_barrier()

    # Build models
    tokenizer, model, mode = _build_model_and_tokenizer(config, device, rank)

    # DDP wrap if needed
    ddp_active = torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1
    if ddp_active:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Log model size (backbone)
    base_ref = (model.module.base if hasattr(model, "module") and hasattr(model.module, "base")
                else model.module if hasattr(model, "module")
                else model)
    if rank == 0:
        print(f"Backbone Model Size: {get_model_size(base_ref)}")

    # Train
    dt_result = train_model(model, tokenizer, device, config, save_dir, wandb_run, rank, world_size, mode)

    # Finalize (master only)
    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Fine-tuning finished. Summary file saved.')
        if wandb_run:
            wandb_run.finish()

    cleanup_ddp()


if __name__ == '__main__':
    # Usage: torchrun --standalone --nproc_per_node=NUM_GPUS target_finetuning.py
    if "WORLD_SIZE" not in os.environ:
        # Single-GPU fallback
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        main(Config().__dict__)
    else:
        config_instance = Config()
        main(config_instance.__dict__)
