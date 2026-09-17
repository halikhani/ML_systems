"""
Gradient Bucketing

Per-parameter all_reduce (see simple_ddp.py's manual_gradient_sync) issues one
collective call per parameter tensor. For a model with thousands of small
parameters, the fixed per-call overhead (kernel launch, network round trip)
dominates the actual data transfer time.

Gradient bucketing fixes this by packing multiple gradients into a single
flat buffer and issuing one all_reduce per buffer ("bucket") instead of one
per parameter. This is the core optimization DDP uses internally.

Usage:
    python gradient_bucketing.py
    python gradient_bucketing.py --bucket-size 4096
"""

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
    Group gradient tensors into buckets of (roughly) bucket_numel elements each,
    so that sync_bucketed can issue one all_reduce per bucket instead of one
    all_reduce per gradient.

    A single gradient tensor may need to be split across bucket boundaries
    (e.g. a big tensor spans two buckets) or share a bucket with other small
    tensors, so each bucket should be represented as a list of
    (flattened_grad_tensor, offset, length) slices rather than whole tensors.

    Correctness requirement: every rank must build IDENTICAL buckets in the
    same order (same grads, same order, same bucket boundaries) since the
    all_reduce calls in sync_bucketed must line up 1:1 across ranks.

    Args:
        grads: list of gradient tensors (one per parameter with grad set).
        bucket_numel: target number of elements per bucket.

    Returns:
        list of buckets, where each bucket is a list of
        (flattened_grad, offset, length) tuples describing which slice of
        which gradient belongs in that bucket.
    """
    # TODO: flatten each grad tensor with .contiguous().view(-1)
    flattened_grads = [
        grad.contiguous().view(-1)
        for grad in grads
    ]
    # TODO: walk through each flattened grad, splitting it into chunks that
    #       fit in the current bucket (a single grad may span multiple buckets)
    buckets = []
    current_bucket = []
    current_bucket_size = 0
    for flattened_grad in flattened_grads:
        grad_offset = 0
        while grad_offset < flattened_grad.numel(): # this is basically checking we went over all the elems in the current flattened_grad
            space_left = bucket_numel - current_bucket_size
            grad_left = flattened_grad.numel() - grad_offset
            take = min(space_left, grad_left)

            current_bucket.append(
                (flattened_grad, grad_offset, take)
            )
            grad_offset += take
            current_bucket_size += take

            if current_bucket_size == bucket_numel:
                buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0

    if current_bucket:
        buckets.append(current_bucket)
    
    return buckets



def sync_bucketed(model, world_size, bucket_numel):
    """
    Synchronize gradients using bucketed all_reduce.

    For each bucket produced by build_buckets:
      1. Allocate a flat buffer of the right size ("pack" the bucket).
      2. Copy each grad slice into its spot in the buffer.
      3. all_reduce the buffer (SUM) and divide by world_size to average.
      4. Copy the averaged values back out of the buffer into the original
         gradient tensors ("unpack" the bucket), in place.
    """
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    buckets = build_buckets(grads, bucket_numel)

    # TODO: for each bucket:
    #   - allocate a zeroed flat tensor sized to the bucket's total element count
    #   - pack: copy each (flat, offset, length) slice into the buffer
    #   - dist.all_reduce the buffer, then divide by world_size
    #   - unpack: copy the averaged values back into each slice's original
    #     location (this mutates param.grad in place)
    # raise NotImplementedError("sync_bucketed: implement pack / all_reduce / unpack")

    for bucket in buckets:
        # Figure out buffer size
        total_numel = sum(
            length 
            for _, _, length in bucket
        )

        # Allocate communication buffer
        flat_grad, _, _ = bucket[0]
        buffer = torch.zeros(
            total_numel,
            device=flat_grad.device,
            dtype=flat_grad.dtype,
        )

        # -----------------
        # PACK
        # -----------------
        buffer_offset = 0
        for flat_grad, grad_offset, length in bucket:

            # NOTE: destination.copy_(source)
            buffer[
                buffer_offset:buffer_offset + length
            ].copy_(
                flat_grad[
                    grad_offset: grad_offset + length
                ]
            )

            buffer_offset += length
        
        # -----------------
        # COMMUNICATE
        # -----------------

        dist.all_reduce(
            buffer,
            op=dist.ReduceOp.SUM,
        )
        buffer /= world_size

        # -----------------
        # UNPACK
        # -----------------

        buffer_offset = 0
        for flat_grad, grad_offset, length in bucket:
            flat_grad[
                grad_offset: grad_offset + length
            ].copy_(
                buffer[
                    buffer_offset: buffer_offset + length
                ]
            )
            
            buffer_offset += length

        # NOTE:
        # grad_offset
        #     ↓
        # where inside the original gradient?

        # buffer_offset
        #     ↓
        # where inside the communication buffer?


def sync_per_param(model, world_size):
    """Baseline: one all_reduce per parameter gradient (no bucketing)."""
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
    torch.manual_seed(42 + rank)  # different seed for each rank

    local_input = torch.randn(8, 4, device=device)  # batch of size 8, 4 features
    local_target = torch.randint(2, (8,), device=device)  # batch of size 8, 2 classes

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
