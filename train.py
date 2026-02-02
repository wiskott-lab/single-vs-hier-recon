# train.py
# Single entry point that works BOTH locally and on HPC.
# Use `torchrun` everywhere (even for 1 GPU) so DDP just works.

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import get_config
from distributed import (
    ddp_setup, ddp_cleanup, is_distributed, is_main_process,
    reduce_mean, set_seed, get_world_size,
    get_rank, ddp_debug_print,
)
from logging_neptune import NeptuneRun, CheckpointManager
from data import build_dataloaders
from model_def import build_model
from checkpoints import save_checkpoint, install_preemption_handler
from vqvae.codebook_tracker import CodebookTracker
from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm

def main():
    cfg = get_config()
    ddp_setup()  # initialize DDP if WORLD_SIZE > 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True  # speed-up for fixed-size inputs

    # Neptune only init on rank 0 to avoid duplicate runs
    neptune = NeptuneRun()
    if is_main_process():
        neptune.init(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
        )
        for k, v in vars(cfg).items():
            neptune[f"params/{k}"] = v
        neptune["params/world_size"] = get_world_size()
    ckpt_mng = CheckpointManager(neptune.run if is_main_process() else None)
    train_loader, val_loader, train_sampler = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    if cfg.torch_compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    raw_model = model
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=False,
        )
        raw_model = model.module

    optimizer = optim.Adam(raw_model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # preemption handler
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    install_preemption_handler(
        raw_model,
        optimizer,
        run_dir,
        neptune_if_main=neptune if is_main_process() else None,
    )

    # log parameter count (rank 0 only)
    if is_main_process():
        total_params = sum(p.numel() for p in raw_model.parameters())
        neptune["params/total_params"].log(total_params)
        try:
            dummy = torch.randn(1, 3, cfg.input_size, cfg.input_size, device=device)
            flops = FlopCountAnalysis(raw_model, dummy).total()  # total FLOPs for one forward
            gflops = flops / 1e9
            neptune["compute/gflops_per_forward"].log(gflops)
        except Exception:
            pass

    # codebook usage trackers (handles N hierarchical levels)
    tracker_train = CodebookTracker(levels=cfg.num_levels)
    tracker_val = CodebookTracker(levels=cfg.num_levels)

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    train_start_time = time.time()

    # Training
    for epoch in range(cfg.epochs):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time.perf_counter()
        local_num_samples = 0

        if is_main_process():
            neptune["params/epoch"].log(epoch)

        running = 0.0
        iterator = train_loader
        if is_main_process():
            iterator = tqdm(iterator, desc=f"Epoch {epoch+1}/{cfg.epochs} [Training]")

        for images, _ in iterator:
            bsz = images.size(0)
            local_num_samples += bsz
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                reconstructed, quant_loss, *levels_indices = model(images)
                quant_loss = quant_loss.mean()
                rec_loss = criterion(reconstructed, images)
                loss = rec_loss + cfg.beta * quant_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            # metric reductions across ranks (mean), then log from rank 0
            mean_rec = reduce_mean(torch.tensor(rec_loss.item(), device=device))
            mean_quant = reduce_mean(torch.tensor(quant_loss.item(), device=device))
            if is_main_process():
                neptune["train/mse"].log(float(mean_rec))
                neptune["train/quant_loss"].log(float(mean_quant))

            # codebook usage logging per level (only rank 0)
            if is_main_process():
                for level, indices in enumerate(levels_indices):
                    batch_used, batch10_used = tracker_train.update_codebook_usage(
                        indices, level=level
                    )
                    neptune[f"train/cb_usage_batch_l_{level}"].log(int(batch_used))
                    neptune[f"train/cb_usage_10batch_l_{level}"].log(int(batch10_used))

        # epoch loss averaged across processes
        local_mean = torch.tensor(running / len(train_loader), device=device)
        train_loss = reduce_mean(local_mean).item()

        # compute global samples & epoch time, then throughput
        local_samples_tensor = torch.tensor(float(local_num_samples), device=device)
        mean_samples = reduce_mean(local_samples_tensor)  # mean over ranks
        global_samples = float(mean_samples) * get_world_size()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time_sec = time.perf_counter() - epoch_start
        epoch_time_min = epoch_time_sec / 60.0
        throughput = global_samples / epoch_time_sec if epoch_time_sec > 0 else 0.0

        if is_main_process():
            runtime_hours = (time.time() - train_start_time) / 3600.0
            neptune["train/loss"].log(train_loss)
            neptune["compute/epoch_time_min"].log(epoch_time_min)
            neptune["compute/throughput_imgs_per_sec"].log(throughput)
            neptune["compute/runtime_hours"].log(runtime_hours)
            best_train_loss = ckpt_mng.save_and_upload(
                raw_model, train_loss, best_train_loss, phase="train", epoch=epoch
            )

        # Validation
        model.eval()
        v_running = 0.0
        iterator = val_loader
        if is_main_process():
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Epoch {epoch+1}/{cfg.epochs} [Validation]")

        with torch.no_grad():
            for images, _ in iterator:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    vrec, vq, *val_levels_indices = model(images)
                    vq = vq.mean()
                    vrec_loss = criterion(vrec, images)
                    vloss = vrec_loss + cfg.beta * vq

                v_running += vloss.item()
                mean_vrec = reduce_mean(torch.tensor(vrec_loss.item(), device=device))
                mean_vq = reduce_mean(torch.tensor(vq.item(), device=device))
                if is_main_process():
                    neptune["val/mse"].log(float(mean_vrec))
                    neptune["val/quant_loss"].log(float(mean_vq))
                    for level, indices in enumerate(val_levels_indices):
                        b_used, b10_used = tracker_val.update_codebook_usage(
                            indices, level=level
                        )
                        neptune[f"val/cb_usage_batch_l_{level}"].log(int(b_used))
                        neptune[f"val/cb_usage_10batch_l_{level}"].log(int(b10_used))

        local_vmean = torch.tensor(v_running / len(val_loader), device=device)
        val_loss = reduce_mean(local_vmean).item()
        if is_main_process():
            neptune["val/loss"].log(val_loss)
            best_val_loss = ckpt_mng.save_and_upload(
                raw_model, val_loss, best_val_loss, phase="val", epoch=epoch
            )

    # finalize
    if is_main_process():
        neptune.stop()
        ckpt_mng.finalize()
    ddp_cleanup()


if __name__ == "__main__":
    main()
