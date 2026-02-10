import argparse
import os
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


class TinyModel(nn.Module):
    """A tiny model for visualization purposes."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def build_buckets(grads: List[torch.Tensor], bucket_numel: int):
    """
    We bucket so instead of #(grads) all_reduce calls, we have #(grads) / bucket_numel all_reduce calls.
    Each bucket is a list of grad tensors of size bucket_numel.
    For correctness, the key is every rank must build identical buckets in the same order so that the all_reduce calls are identical.
    """
    buckets = []
    current = []
    current_numel = 0

    for grad in grads:
        flattened = grad.contiguous().view(-1)
        offset = 0
        remaining = flattened.numel()
        while remaining > 0:
            space_left = bucket_numel - current_numel
            space_taken = min(space_left, remaining) # take as much as we can from the remaining
            current.append((flattened, offset, space_taken)) # only saving a reference to the grad tensor, not the data, so its efficient memory-wise
            remaining -= space_taken
            offset += space_taken
            current_numel += space_taken
            if current_numel == bucket_numel:
                buckets.append(current)
                current = []
                current_numel = 0

    if current:
        buckets.append(current)

    return buckets


def sync_bucketed(model, world_size, bucket_numel):

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    buckets = build_buckets(grads, bucket_numel)
    
    for slices in buckets:
        total_len = sum(space_taken for _, _, space_taken in slices)
        bucket = torch.zeros(total_len, device=grads[0].device, dtype=grads[0].dtype)

        # pack the grad tensors into the bucket
        cursor = 0
        for flat, offset, space_taken in slices:
            bucket[cursor:cursor+space_taken] = flat[offset:offset+space_taken]
            cursor += space_taken

        # all_reduce the bucket
        dist.all_reduce(bucket, op=dist.ReduceOp.SUM)   
        bucket /= world_size

        # unpack the bucket into the grad tensors
        cursor = 0
        for flat, offset, space_taken in slices:
            flat[offset:offset+space_taken] = bucket[cursor:cursor+space_taken]
            cursor += space_taken


def sync_per_param(model, world_size):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def _run_step(model, device, local_input, local_target):
    outputs = model(local_input)
    loss = F.cross_entropy(outputs, local_target)
    model.zero_grad()
    loss.backward()
    return loss


def _time_sync(
    model,
    device,
    local_input,
    local_target,
    world_size,
    bucket_numel,
    iters,
    warmup,
    use_bucket,
):
    for _ in range(warmup):
        _run_step(model, device, local_input, local_target)
        if use_bucket:
            sync_bucketed(model, world_size, bucket_numel)
        else:
            sync_per_param(model, world_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        _run_step(model, device, local_input, local_target)
        if use_bucket:
            sync_bucketed(model, world_size, bucket_numel)
        else:
            sync_per_param(model, world_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    return (end - start) / iters

        
def bucketing_worker(rank, world_size, args):
    """Worker function for each process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29508"

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")


    # =========================================================================
    # Setup: Create identical models on all ranks
    # =========================================================================

    torch.manual_seed(42)
    model = TinyModel().to(device)
    # NOTE: dist.broadcast writes into each param in place, and these param require grad (model params)
    # Pytorch forbids in-place ops on such tensors, so we need to use torch.no_grad() to avoid this.
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(" GRADIENT BUCKETING ")
        print("=" * 60)
        print(f"\nWorld size: {world_size}")
        print(f"Model: {model}")


    # Create rank-specific data (simulating distributed batch)
    torch.manual_seed(42 + rank) # different seed for each rank

    local_input = torch.randn(8, 4, device=device) # batch of size 8, 4 features
    local_target = torch.randint(2, (8,), device=device) # batch of size 8, 2 classes

    dist.barrier()


    # =========================================================================
    # Forward and backward (compute LOCAL gradients)
    # =========================================================================
    if rank == 0:
        print("\n" + "-" * 60)
        print(" STEP 2: Compute gradients LOCALLY (before sync)")
        print("-" * 60)


    # Warmup + timing for baseline and bucketed sync
    baseline_time = _time_sync(
        model,
        device,
        local_input,
        local_target,
        world_size,
        args.bucket_size,
        args.iters,
        args.warmup,
        use_bucket=False,
    )
    bucket_time = _time_sync(
        model,
        device,
        local_input,
        local_target,
        world_size,
        args.bucket_size,
        args.iters,
        args.warmup,
        use_bucket=True,
    )

    dist.barrier()
    if rank == 0:
        baseline_ms = baseline_time * 1000
        bucket_ms = bucket_time * 1000
        speedup = baseline_time / bucket_time if bucket_time > 0 else 0.0
        print("\n" + "-" * 60)
        print(" THROUGHPUT COMPARISON (backward + sync)")
        print("-" * 60)
        print(f"Per-param all_reduce: {baseline_ms:.3f} ms/iter")
        print(f"Bucketed all_reduce : {bucket_ms:.3f} ms/iter")
        print(f"Speedup (baseline/bucketed): {speedup:.2f}x")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Gradient Bucketing")
    parser.add_argument("--world-size", "-w", type=int, default=4)
    parser.add_argument("--bucket-size", type=int, default=1024,
                        help="Bucket size in number of elements")
    parser.add_argument("--iters", type=int, default=40,
                        help="Iterations for timing")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (not timed)")
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║" + " GRADIENT BUCKETING ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    mp.spawn(
        bucketing_worker,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
