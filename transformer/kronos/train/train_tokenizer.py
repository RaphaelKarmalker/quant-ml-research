import os
import sys
import json
import time
from time import gmtime, strftime
import argparse
import datetime
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.training_utils import safe_barrier
import wandb

# Ensure project root is in path
sys.path.append("../")
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
)


def create_dataloaders(config: dict, rank: int, world_size: int):
    """
    Creates and returns distributed dataloaders for training and validation.

    Args:
        config (dict): A dictionary of configuration parameters.
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, train_dataset, valid_dataset).
    """
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
        shuffle=False,  # Shuffle is handled by the sampler
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True  # include all samples
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


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def _unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def apply_tokenizer_finetune_strategy(model, config, rank=0):
    mode = config.get("tokenizer_ft_mode", "full")
    if mode == "full":
        # train everything
        pass
    elif mode == "linear_only":
        _freeze_all(model)
        # Train only quantizer projections and final head
        _unfreeze(model.quant_embed)
        _unfreeze(model.post_quant_embed_pre)
        _unfreeze(model.post_quant_embed)
        _unfreeze(model.head)
    elif mode == "freeze_enc_dec":
        # Freeze encoder/decoder blocks, train embeddings and projections
        for blk in model.encoder:
            _freeze_all(blk)
        for blk in model.decoder:
            _freeze_all(blk)
        _unfreeze(model.embed)
        _unfreeze(model.quant_embed)
        _unfreeze(model.post_quant_embed_pre)
        _unfreeze(model.post_quant_embed)
        _unfreeze(model.head)
    elif mode == "last_n":
        n = int(config.get("tokenizer_unfreeze_last_n", 1))
        _freeze_all(model)
        # Unfreeze last n encoder/decoder blocks and necessary projections/head
        if len(model.encoder) > 0:
            for blk in model.encoder[-n:]:
                _unfreeze(blk)
        if len(model.decoder) > 0:
            for blk in model.decoder[-n:]:
                _unfreeze(blk)
        _unfreeze(model.quant_embed)
        _unfreeze(model.post_quant_embed_pre)
        _unfreeze(model.post_quant_embed)
        _unfreeze(model.head)
    else:
        raise ValueError(f"Unknown tokenizer_ft_mode: {mode}")

    total, trainable = _count_params(model)
    if rank == 0:
        print(f"[Tokenizer FT] Mode={mode}, Params trainable/total: {trainable}/{total} ({trainable/total:.2%})")


def train_model(model, device, config, save_dir, logger, rank, world_size):
    """
    The main training and validation loop for the tokenizer.
    """
    start_time = time.time()
    if rank == 0:
        effective_bs = config['batch_size'] * world_size * config['accumulation_steps']
        print(f"[Rank {rank}] BATCHSIZE (per GPU): {config['batch_size']}")
        print(f"[Rank {rank}] Effective total batch size: {effective_bs}")

    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config, rank, world_size)
    ddp_active = torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['tokenizer_learning_rate'],
        weight_decay=config['adam_weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config['tokenizer_learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs'],
        pct_start=0.03,
        div_factor=10
    )

    best_val_loss = float('inf')
    dt_result = {}
    batch_idx_global_train = 0

    for epoch_idx in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        # only if sampler supports it
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch_idx)

        # Set dataset seeds for reproducible sampling
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        valid_dataset.set_epoch_seed(0)  # Keep validation sampling consistent

        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)

            # --- Gradient Accumulation Loop ---
            current_batch_total_loss = 0.0
            for j in range(config['accumulation_steps']):
                start_idx = j * (ori_batch_x.shape[0] // config['accumulation_steps'])
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config['accumulation_steps'])
                batch_x = ori_batch_x[start_idx:end_idx]

                # Forward pass
                zs, bsq_loss, _, _ = model(batch_x)
                z_pre, z = zs

                # Loss calculation
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                recon_loss = recon_loss_pre + recon_loss_all
                loss = (recon_loss + bsq_loss) / 2  # Assuming w_1=w_2=1

                loss_scaled = loss / config['accumulation_steps']
                current_batch_total_loss += loss.item()
                loss_scaled.backward()

            # --- Optimizer Step after Accumulation ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # --- Logging (Master Process Only) ---
            if rank == 0 and (batch_idx_global_train + 1) % config['log_interval'] == 0:
                avg_loss = current_batch_total_loss / config['accumulation_steps']
                print(
                    f"[Rank {rank}, Epoch {epoch_idx + 1}/{config['epochs']}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )
            if rank == 0 and logger:
                avg_loss = current_batch_total_loss / config['accumulation_steps']
                wandb.log({
                    'train_tokenizer_loss_batch': avg_loss,
                    'train_vqvae_vq_loss_each_batch': bsq_loss.item(),
                    'train_recon_loss_pre_each_batch': recon_loss_pre.item(),
                    'train_recon_loss_each_batch': recon_loss_all.item(),
                    'tokenizer_learning_rate': optimizer.param_groups[0]["lr"],
                }, step=batch_idx_global_train)

            batch_idx_global_train += 1

        # --- Validation Loop ---
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                zs, _, _, _ = model(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)

                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)

        if ddp_active:
            # Reduce validation losses from all processes
            val_loss_sum_tensor = torch.tensor(tot_val_loss_sum_rank, device=device)
            val_count_tensor = torch.tensor(val_sample_count_rank, device=device)
            dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_sum_tensor.item() / val_count_tensor.item() if val_count_tensor.item() > 0 else 0
        else:
            # Single-process average
            avg_val_loss = tot_val_loss_sum_rank / val_sample_count_rank if val_sample_count_rank > 0 else 0

        # --- End of Epoch Summary & Checkpointing (Master Process Only) ---
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx + 1}/{config['epochs']} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            if logger:
                wandb.log({'val_tokenizer_loss_epoch': avg_val_loss, 'epoch': epoch_idx})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = f"{save_dir}/checkpoints/best_model"
                # handle both DDP and non-DDP
                (model.module if hasattr(model, "module") else model).save_pretrained(save_path)
                print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
                if logger:
                    artifact = wandb.Artifact("best_model_tokenizer", type="model")
                    artifact.add_dir(save_path)
                    wandb.log_artifact(artifact)

        

        safe_barrier()

        #dist.barrier()  # Ensure all processes finish the epoch before starting the next one.

    dt_result['best_val_loss'] = best_val_loss
    return model, dt_result


def main(config: dict):
    """
    Main function to orchestrate the DDP training process.
    """
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(config['seed'], rank)

    save_dir = os.path.join(config['save_path'], config['tokenizer_save_folder_name'])

    # Logger and summary setup (master process only)
    wandb_run, master_summary = None, {}
    if rank == 0:
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        master_summary = {
            'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            'save_directory': save_dir,
            'world_size': world_size,
        }
        if config['use_wandb']:
            wandb_run = wandb.init(
                project=config['wandb_config']['project'],
                entity=config['wandb_config'].get('entity'),
                name=config['wandb_name'],
                tags=[config['wandb_tag']]
            )
            wandb.config.update(config)
            print("Weights & Biases Logger Initialized.")
    safe_barrier()
    #dist.barrier()  # Ensure save directory is created before proceeding

    # Model Initialization
    if config.get('init_tokenizer_from_pretrained', True):
        if rank == 0:
            print("[Tokenizer Init] Loading pretrained tokenizer:", config['pretrained_tokenizer_path'])
        model = KronosTokenizer.from_pretrained(config['pretrained_tokenizer_path'])
    else:
        arch = config['tokenizer_arch']
        if rank == 0:
            print("[Tokenizer Init] Initializing from scratch with arch:", arch)
        model = KronosTokenizer(
            arch['d_in'], arch['d_model'], arch['n_heads'], arch['ff_dim'],
            arch['n_enc_layers'], arch['n_dec_layers'],
            arch['ffn_dropout_p'], arch['attn_dropout_p'], arch['resid_dropout_p'],
            arch['s1_bits'], arch['s2_bits'],
            arch['beta'], arch['gamma0'], arch['gamma'], arch['zeta'],
            arch['group_size']
        )

    # Optionally control which parts to train (kept "full" by default)
    apply_tokenizer_finetune_strategy(model, config, rank)

    model.to(device)

    # Wrap with DDP only if process group is initialized and multi-process
    ddp_active = torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1
    if ddp_active:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    base_model = model.module if hasattr(model, "module") else model
    if rank == 0:
        print(f"Model Size: {get_model_size(base_model)}")

    # Start Training
    _, dt_result = train_model(
        model, device, config, save_dir, wandb_run, rank, world_size
    )

    # Finalize and save summary (master process only)
    if rank == 0:
        master_summary['final_result'] = dt_result
        with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
            json.dump(master_summary, f, indent=4)
        print('Training finished. Summary file saved.')
        if wandb_run:
            wandb_run.finish()

    cleanup_ddp()


if __name__ == '__main__':
    # Usage: torchrun --standalone --nproc_per_node=NUM_GPUS train_tokenizer.py
    if "WORLD_SIZE" not in os.environ:
        # Fallback für Single-GPU Training auf Windows
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        main(Config().__dict__)
    else:
        config_instance = Config()
        main(config_instance.__dict__)
