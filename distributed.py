# distributed.py

import os
import random
import numpy as np
import torch
import torch.distributed as dist

def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def is_main_process() -> bool:
    """Rank 0 only (good for logging, checkpointing, etc.)."""
    return get_rank() == 0

def ddp_setup():
    """Initialize process group if launched by torchrun (env://)."""
    if is_distributed():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend="nccl", init_method="env://")

def ddp_cleanup():
    if is_distributed() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def all_reduce(tensor: torch.Tensor):
    """Sum-reduce in-place across ranks, no-op in single-process runs."""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """Return mean across ranks (copy)."""
    t = x.clone()
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    return t

def ddp_debug_print(tag: str = ""):
    """Debug helper: print rank/world_size once from each process."""
    msg = (
        f"[DDP DEBUG] {tag} "
        f"RANK={get_rank()} WORLD_SIZE={get_world_size()} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK', '0')}"
    )
    print(msg, flush=True)
