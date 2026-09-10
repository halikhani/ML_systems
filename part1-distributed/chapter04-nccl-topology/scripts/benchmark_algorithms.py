"""
NCCL Algorithm Benchmark

Benchmarks all_reduce performance across:
- Message sizes
- Number of processes
- Backend (Gloo or NCCL)
- NCCL algorithm selection (Auto, Ring, or Tree)

Examples:
    python benchmark_algorithms.py

    python benchmark_algorithms.py \
        --backend nccl \
        --world-size 4 \
        --algorithm ring

    python benchmark_algorithms.py \
        --backend nccl \
        --world-size 4 \
        --algorithm tree \
        --sizes 1024,16384,262144,4194304,67108864

Notes:
- NCCL requires CUDA GPUs.
- Gloo does not expose NCCL Ring/Tree algorithm selection.
- Tree often performs better for small latency-bound messages.
- Ring often performs better for sufficiently large bandwidth-bound messages.
- The crossover point depends on GPU count, topology, NCCL version,
  message size, protocol, and hardware.
"""

import argparse
import os
import time
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def format_bytes(size: float) -> str:
    """Format a byte count into human-readable form."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} TB"


def format_bandwidth(bytes_per_sec: float) -> str:
    """Format bandwidth into human-readable form."""
    return format_bytes(bytes_per_sec) + "/s"


def benchmark_all_reduce(
    tensor: torch.Tensor,
    num_iterations: int = 50,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark synchronous completion latency of all_reduce.

    With CUDA, time.perf_counter() is paired with cuda.synchronize()
    so that the CPU timer includes actual GPU-side collective completion.
    """

    # Warmup
    for _ in range(warmup_iterations):
        dist.all_reduce(tensor)

    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)

    # Align all ranks before beginning the timed experiment.
    dist.barrier()

    times = []

    for _ in range(num_iterations):
        # Ensure no previously queued CUDA work leaks into this iteration.
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

        start_time = time.perf_counter()

        dist.all_reduce(tensor)

        # all_reduce on CUDA is asynchronous with respect to the CPU,
        # so wait for GPU-side completion before stopping the timer.
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    sorted_times = sorted(times)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "median_ms": sorted_times[len(sorted_times) // 2] * 1000,
    }


def reduce_max_across_ranks(value: float, device: torch.device) -> float:
    """
    Return the maximum scalar value observed across ranks.

    For distributed latency, the slowest rank is often the most useful
    summary because the collective is constrained by its slowest participant.
    """
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def benchmark_worker(
    rank: int,
    world_size: int,
    message_sizes: List[int],
    backend: str,
    num_iterations: int,
    algorithm: str,
) -> None:
    """Worker function run by each distributed process."""

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29508"

    # NCCL_ALGO must be configured before process-group initialization.
    if backend == "nccl":
        if algorithm == "ring":
            os.environ["NCCL_ALGO"] = "Ring"
        elif algorithm == "tree":
            os.environ["NCCL_ALGO"] = "Tree"
        else:
            # Ensure --algorithm auto is truly automatic even if the shell
            # already has NCCL_ALGO configured.
            os.environ.pop("NCCL_ALGO", None)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    device = torch.device("cpu")

    if backend == "nccl":
        num_gpus = torch.cuda.device_count()

        if num_gpus == 0:
            raise RuntimeError("NCCL backend requires at least one CUDA GPU.")

        if world_size > num_gpus:
            raise ValueError(
                f"world_size={world_size}, but only {num_gpus} CUDA GPUs "
                "are available on this node."
            )

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    results = []

    for requested_size in message_sizes:
        # float32 = 4 bytes per element.
        num_elements = (requested_size + 3) // 4
        actual_size = num_elements * 4

        # Zeros avoid numerical overflow when SUM all_reduce is repeated.
        tensor = torch.zeros(
            num_elements,
            dtype=torch.float32,
            device=device,
        )

        result = benchmark_all_reduce(
            tensor,
            num_iterations=num_iterations,
        )

        # Use the slowest rank's aggregate timing statistics as the reported
        # distributed latency metrics.
        global_mean_ms = reduce_max_across_ranks(result["mean_ms"], device)
        global_min_ms = reduce_max_across_ranks(result["min_ms"], device)
        global_max_ms = reduce_max_across_ranks(result["max_ms"], device)
        global_median_ms = reduce_max_across_ranks(result["median_ms"], device)

        seconds = global_mean_ms / 1000.0

        # Algorithm bandwidth:
        # tensor payload size divided by collective completion time.
        alg_bandwidth = actual_size / seconds

        # Ring-normalized bus bandwidth:
        # Useful when comparing against the standard Ring all-reduce traffic
        # model. It should not be interpreted as generic physical link BW for
        # arbitrary Tree executions.
        ring_bus_bandwidth = (
            2.0
            * (world_size - 1)
            / world_size
            * actual_size
            / seconds
        )

        results.append(
            {
                "requested_size": requested_size,
                "actual_size": actual_size,
                "num_elements": num_elements,
                "mean_ms": global_mean_ms,
                "min_ms": global_min_ms,
                "max_ms": global_max_ms,
                "median_ms": global_median_ms,
                "alg_bandwidth": alg_bandwidth,
                "ring_bus_bandwidth": ring_bus_bandwidth,
            }
        )

        # Keep different message-size experiments separated.
        dist.barrier()

    if rank == 0:
        print("\n" + "=" * 100)
        print(" ALL_REDUCE BENCHMARK RESULTS")
        print("=" * 100)
        print(f"Backend:              {backend}")
        print(f"Algorithm:            {algorithm}")
        print(f"World size:           {world_size}")
        print(f"Device:               {device}")
        print(f"Iterations per test:  {num_iterations}")
        print("=" * 100)

        header = (
            f"{'Size':<12}"
            f"{'Elements':<14}"
            f"{'Mean (ms)':<13}"
            f"{'Median (ms)':<15}"
            f"{'Min (ms)':<12}"
            f"{'AlgBW':<16}"
            f"{'Ring BusBW':<16}"
        )

        print("\n" + header)
        print("-" * 100)

        for r in results:
            print(
                f"{format_bytes(r['actual_size']):<12}"
                f"{r['num_elements']:<14}"
                f"{r['mean_ms']:<13.3f}"
                f"{r['median_ms']:<15.3f}"
                f"{r['min_ms']:<12.3f}"
                f"{format_bandwidth(r['alg_bandwidth']):<16}"
                f"{format_bandwidth(r['ring_bus_bandwidth']):<16}"
            )

        if len(results) >= 2:
            small = results[0]
            large = results[-1]

            size_ratio = large["actual_size"] / small["actual_size"]
            latency_ratio = large["mean_ms"] / small["mean_ms"]

            print("\n" + "=" * 100)
            print(" ANALYSIS")
            print("=" * 100)

            print("\nLatency scaling:")
            print(f"  Message size increased {size_ratio:.0f}x")
            print(f"  Mean latency increased {latency_ratio:.1f}x")

            if latency_ratio < size_ratio * 0.5:
                print(
                    "  -> Latency grows substantially slower than message size, "
                    "indicating improving bandwidth utilization."
                )
            elif latency_ratio < size_ratio:
                print(
                    "  -> Latency grows sub-linearly relative to message size."
                )
            else:
                print(
                    "  -> Latency grows at least as fast as message size; "
                    "check for communication or topology bottlenecks."
                )

            print("\nPayload bandwidth comparison:")
            print(
                f"  Small ({format_bytes(small['actual_size'])}): "
                f"{format_bandwidth(small['alg_bandwidth'])}"
            )
            print(
                f"  Large ({format_bytes(large['actual_size'])}): "
                f"{format_bandwidth(large['alg_bandwidth'])}"
            )

            if large["alg_bandwidth"] > small["alg_bandwidth"] * 1.5:
                print(
                    "  -> Large messages achieve substantially better "
                    "bandwidth utilization because fixed startup costs are amortized."
                )

        print(
            """
Interpretation:

1. SMALL MESSAGES
   - Often latency-bound.
   - Fixed communication/setup overhead is important.
   - Tree algorithms can perform well because communication depth is O(log N).

2. LARGE MESSAGES
   - Often bandwidth-bound.
   - Fixed latency matters less relative to data-transfer time.
   - Ring algorithms can perform well because they pipeline chunks and use
     available links efficiently.

3. NCCL AUTO-SELECTION
   - With --algorithm auto, NCCL chooses the communication strategy.
   - Do not assume a universal message-size crossover between Tree and Ring.
   - The choice depends on topology, GPU count, message size, protocol,
     hardware, and NCCL version.

4. BANDWIDTH COLUMNS
   - AlgBW = tensor payload size / collective time.
   - Ring BusBW applies the standard Ring all-reduce normalization factor:
         2 * (N - 1) / N
     It is useful as a Ring-oriented normalized metric, but should not be
     interpreted as the actual physical bytes moved by every possible Tree
     execution.

For an explicit comparison, run the same benchmark twice:

    python benchmark_algorithms.py --backend nccl --algorithm tree
    python benchmark_algorithms.py --backend nccl --algorithm ring
"""
        )

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NCCL Ring/Tree All-Reduce Benchmark"
    )

    parser.add_argument(
        "--sizes",
        type=str,
        default="1024,16384,262144,4194304,67108864,268435456",
        help=(
            "Comma-separated message sizes in bytes "
            "(default: 1KB,16KB,256KB,4MB,64MB,256MB)"
        ),
    )

    parser.add_argument(
        "--algorithm",
        choices=["auto", "ring", "tree"],
        default="auto",
        help="NCCL algorithm selection (default: auto).",
    )

    parser.add_argument(
        "--world-size",
        "-w",
        type=int,
        default=4,
        help="Number of distributed processes (default: 4).",
    )

    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend (default: gloo).",
    )

    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=50,
        help="Number of timed iterations per message size (default: 50).",
    )

    args = parser.parse_args()

    if args.world_size < 1:
        parser.error("--world-size must be at least 1")

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    try:
        message_sizes = [int(s.strip()) for s in args.sizes.split(",")]
    except ValueError as exc:
        parser.error(f"Invalid --sizes value: {exc}")

    if any(size <= 0 for size in message_sizes):
        parser.error("All message sizes must be positive integers.")

    if args.backend == "nccl" and not torch.cuda.is_available():
        print(
            "\n[WARN] NCCL requires CUDA, but CUDA is unavailable. "
            "Falling back to Gloo."
        )
        args.backend = "gloo"

    if args.backend != "nccl" and args.algorithm != "auto":
        print(
            f"\n[WARN] --algorithm {args.algorithm} only applies to NCCL. "
            "Gloo will ignore Ring/Tree selection."
        )

    print("╔" + "═" * 62 + "╗")
    print("║" + " NCCL ALL_REDUCE ALGORITHM BENCHMARK".center(62) + "║")
    print("╚" + "═" * 62 + "╝")

    print(f"\nMessage sizes: {[format_bytes(s) for s in message_sizes]}")
    print(f"World size:    {args.world_size}")
    print(f"Backend:       {args.backend}")
    print(f"Algorithm:     {args.algorithm}")
    print(f"Iterations:    {args.iterations}")

    mp.spawn(
        benchmark_worker,
        args=(
            args.world_size,
            message_sizes,
            args.backend,
            args.iterations,
            args.algorithm,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
