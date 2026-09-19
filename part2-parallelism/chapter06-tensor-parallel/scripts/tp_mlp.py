#!/usr/bin/env python3
"""
Tensor-Parallel MLP Block

This script implements a complete tensor-parallel MLP block using
the Megatron-style column→row pattern for minimal communication.

Usage:
    python tp_mlp.py
    python tp_mlp.py --tp-size 4 --hidden-size 256
"""

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp



class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel MLP using Megatron-style column→row parallelism.

    Structure:
        Input → [Column-Parallel Linear] → GeLU → [Row-Parallel Linear] → Output

    Communication: 1 all_reduce per forward pass (after row-parallel)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        tp_rank: int,
        tp_group=None,
        profile: bool = False,
    ):
        super().__init__()
        self.profile = profile
        self.timings = {"matmul": 0.0, "gelu": 0.0, "wait": 0.0, "all_reduce": 0.0}

        assert intermediate_size % tp_size == 0

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group

        self.intermediate_local = intermediate_size // tp_size

        # Column-parallel: W1 shape [hidden, intermediate // tp_size]
        self.w1 = nn.Linear(hidden_size, self.intermediate_local, bias=False)

        # Row-parallel: W2 shape [intermediate // tp_size, hidden]
        self.w2 = nn.Linear(self.intermediate_local, hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling for TP."""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with minimal communication.

        Args:
            x: Input tensor of shape [batch, seq, hidden]

        Returns:
            Output tensor of shape [batch, seq, hidden]
        """
        if not self.profile:
            # Step 1: Column-parallel linear
            h = self.w1(x)

            # Step 2: GeLU activation (local)
            h = torch.nn.functional.gelu(h)

            # Step 3: Row-parallel linear
            y = self.w2(h)
            
            # Step 4: All-reduce across row-parallel dimension
            dist.all_reduce(y, op=dist.ReduceOp.SUM, group=self.tp_group)

            return y
        else:
            t0 = time.perf_counter()
            h = self.w1(x)                      # matmul 1 (column-parallel)
            t1 = time.perf_counter()
            h = torch.nn.functional.gelu(h)
            t2 = time.perf_counter()
            y = self.w2(h)                      # matmul 2 (row-parallel)
            t3 = time.perf_counter()
            dist.barrier(group=self.tp_group)   # isolate load-imbalance wait from comm
            t3b = time.perf_counter()
            dist.all_reduce(y, group=self.tp_group)
            t4 = time.perf_counter()

            self.timings["matmul"]     += (t1 - t0) + (t3 - t2)
            self.timings["gelu"]       += (t2 - t1)
            self.timings["wait"]       += (t3b - t3)
            self.timings["all_reduce"] += (t4 - t3b)
            return y

    def reset_timings(self):
        self.timings = {"matmul": 0.0, "gelu": 0.0, "wait": 0.0, "all_reduce": 0.0}

class NonParallelMLP(nn.Module):
    """Standard MLP for comparison."""
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.gelu(self.w1(x)))


def benchmark_tp_mlp(rank: int, world_size: int, hidden_size: int,
                     batch_size: int, seq_len: int, warmup: int = 10,
                     iterations: int = 100) -> Tuple[float, torch.Tensor, dict]:

    """Benchmark tensor-parallel MLP."""
    device = torch.device("cpu")
    intermediate_size = hidden_size * 4

    # Create TP MLP
    tp_mlp = TensorParallelMLP(
        hidden_size, intermediate_size, world_size, rank, profile=True
    ).to(device)

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Warmup
    for _ in range(warmup):
        _ = tp_mlp(x)
        dist.barrier()
    tp_mlp.reset_timings()
    # Benchmark
    dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        y = tp_mlp(x)
    total_time = time.perf_counter() - start

    timings = {k: v / iterations for k, v in tp_mlp.timings.items()}
    return total_time / iterations, y, timings


def verify_correctness(rank: int, world_size: int, hidden_size: int) -> None:
    """Verify TP MLP produces correct output."""
    device = torch.device("cpu")
    intermediate_size = hidden_size * 4

    if rank == 0:
        print("\n" + "=" * 60)
        print(" CORRECTNESS VERIFICATION")
        print("=" * 60)

    # Create test input (same on all ranks)
    torch.manual_seed(42)
    x = torch.randn(4, 8, hidden_size, device=device)

    # Create TP MLP with deterministic weights
    torch.manual_seed(100)
    tp_mlp = TensorParallelMLP(
        hidden_size, intermediate_size, world_size, rank
    ).to(device)


    # Forward pass
    y_tp = tp_mlp(x)

    # Gather TP weights to rank 0 for comparison
    # W1 (column-parallel)
    w1_local = tp_mlp.w1.weight.data.clone()
    w1_gathered = [torch.zeros_like(w1_local) for _ in range(world_size)]
    dist.all_gather(w1_gathered, w1_local, group=tp_mlp.tp_group)


    # W2 (row-parallel)
    w2_local = tp_mlp.w2.weight.data.clone()
    w2_gathered = [torch.zeros_like(w2_local) for _ in range(world_size)]
    dist.all_gather(w2_gathered, w2_local, group=tp_mlp.tp_group)

    if rank == 0:
        # Reconstruct full weights
        w1_full = torch.cat(w1_gathered, dim=0).T # w1 stored in torch nn.Linear as [intermediate // tp_size, hidden] -> final shape [hidden, intermediate]
        w2_full = torch.cat(w2_gathered, dim=1) #  final shape [intermediate, hidden] as saved in nn.Linear

        # Compute reference output
        h = torch.nn.functional.gelu(x @ w1_full.T)
        y_ref = h @ w2_full.T

        diff = (y_tp - y_ref).abs().max().item()
        print(f"\nInput shape: {x.shape}")
        print(f"Output shape: {y_tp.shape}")
        print(f"Max difference from reference: {diff:.2e}")
        print(f"Correct: {diff < 1e-5}")


def analyze_communication(rank: int, world_size: int,
                          hidden_size: int, batch_size: int, seq_len: int) -> None:
    """Analyze communication costs."""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print(" COMMUNICATION ANALYSIS")
    print("=" * 60)

    bytes_per_element = 4  # float32
    elements_per_allreduce = batch_size * seq_len * hidden_size
    bytes_per_allreduce = elements_per_allreduce * bytes_per_element

    # Ring all_reduce volume
    ring_volume = 2 * bytes_per_allreduce * (world_size - 1) / world_size

    print(f"""
Configuration:
  Hidden size: {hidden_size}
  Batch size: {batch_size}
  Sequence length: {seq_len}
  TP degree: {world_size}

Per forward pass:
  All-reduce calls: 1
  Elements per all-reduce: {elements_per_allreduce:,}
  Bytes per all-reduce: {bytes_per_allreduce / 1024:.1f} KB

Communication volume (ring algorithm):
  Per GPU: {ring_volume / 1024:.1f} KB
  Total across all GPUs: {ring_volume * world_size / 1024:.1f} KB

Comparison with non-TP:
  Non-TP: 0 bytes (no communication)
  TP: {ring_volume / 1024:.1f} KB per forward

This is the price of tensor parallelism!
But we can now handle models {world_size}x larger.
""")


def compare_scaling(rank: int, world_size: int) -> None:
    """Compare TP vs non-parallel scaling."""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print(" SCALING ANALYSIS")
    print("=" * 60)
    print("""
Memory scaling with Tensor Parallelism:

For an MLP with hidden_size H and intermediate_size 4H:

Non-parallel:
  W1: H × 4H = 4H² parameters
  W2: 4H × H = 4H² parameters
  Total: 8H² parameters per GPU

With TP degree T:
  W1: H × (4H/T) = 4H²/T parameters
  W2: (4H/T) × H = 4H²/T parameters
  Total: 8H²/T parameters per GPU

Example: H=4096, T=8 (8-way TP)
  Non-parallel: 134M parameters (537 MB in FP32)
  With 8-way TP: 16.7M parameters (67 MB per GPU)

This is how we fit 70B+ parameter models on GPUs!
""")


def worker(rank: int, world_size: int, hidden_size: int,
           batch_size: int, seq_len: int) -> None:
    """Main worker function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29509"

    # Avoid CPU oversubscription: split cores across ranks
    torch.set_num_threads(max(1, os.cpu_count() // world_size))

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Verify correctness
    verify_correctness(rank, world_size, hidden_size)
    dist.barrier()

    # Analyze communication
    analyze_communication(rank, world_size, hidden_size, batch_size, seq_len)
    dist.barrier()

    # Benchmark
    if rank == 0:
        print("\n" + "=" * 60)
        print(" BENCHMARK")
        print("=" * 60)

    avg_time, output, timings = benchmark_tp_mlp(
        rank, world_size, hidden_size, batch_size, seq_len
    )

    # Aggregate per-rank timings: max = slowest rank (sets step time), avg for imbalance
    keys = ["matmul", "gelu", "wait", "all_reduce"]
    t = torch.tensor([timings[k] for k in keys], dtype=torch.float64)
    t_max = t.clone()
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    t_avg = t.clone()
    dist.all_reduce(t_avg, op=dist.ReduceOp.SUM)
    t_avg /= world_size

    dist.barrier()

    if rank == 0:
        print(f"\nTP MLP forward pass: {avg_time * 1000:.3f} ms")
        print(f"Output shape: {output.shape}")

        print(f"\n{'Component':<12}{'max (ms)':>12}{'avg (ms)':>12}")
        for k, mx, av in zip(keys, t_max.tolist(), t_avg.tolist()):
            print(f"{k:<12}{mx * 1000:>12.3f}{av * 1000:>12.3f}")

        matmul, gelu, wait, comm = t_max.tolist()
        compute = matmul + gelu
        total = compute + wait + comm
        print(f"\nCompute (matmul + gelu): {compute * 1000:.3f} ms")
        print(f"Communication (all_reduce): {comm * 1000:.3f} ms")
        print(f"Communication %: {100 * comm / total:.1f}%  "
              f"(incl. wait: {100 * (comm + wait) / total:.1f}%)")
        print(f"Unaccounted overhead: {(avg_time - total) * 1000:.3f} ms")

        # Sanity checks: matmul throughput and effective all_reduce bandwidth
        intermediate_local = hidden_size * 4 // world_size
        flops = 2 * 2 * batch_size * seq_len * hidden_size * intermediate_local
        ring_bytes = 2 * batch_size * seq_len * hidden_size * 4 * (world_size - 1) / world_size
        print(f"Matmul throughput: {flops / matmul / 1e9:.2f} GFLOP/s per rank")
        print(f"All-reduce bandwidth: {ring_bytes / comm / 1e9:.3f} GB/s per rank")

    # Compare scaling
    compare_scaling(rank, world_size)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Tensor-Parallel MLP Block")
    parser.add_argument("--tp-size", "-t", type=int, default=4,
                        help="Tensor parallelism degree")
    parser.add_argument("--hidden-size", "-H", type=int, default=64,
                        help="Hidden dimension")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=16,
                        help="Sequence length")
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║" + " TENSOR-PARALLEL MLP BLOCK".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nTP degree: {args.tp_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Intermediate size: {args.hidden_size * 4}")

    mp.spawn(
        worker,
        args=(args.tp_size, args.hidden_size, args.batch_size, args.seq_len),
        nprocs=args.tp_size,
        join=True
    )


if __name__ == "__main__":
    main()

