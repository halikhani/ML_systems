import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import argparse

NUM_WARMUP = 5
NUM_ITERS = 10
tensor_size = 1024 * 1024

def make_tensor(rank, device):
    return torch.randn(tensor_size, device=device)


def measure_latency_blocking(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    
    device = torch.device("cpu")
    tensor = make_tensor(rank, device)

    # warmup
    for _ in range(NUM_WARMUP):
        if rank == 0:
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        elif rank == 1:
            dist.recv(tensor, src=0)
            tensor += 10.0
            dist.send(tensor, dst=0)
    
    dist.barrier()
    
    # Benchmark
    start_time = time.perf_counter()

    for _ in range(NUM_ITERS):
        if rank == 0:
            dist.send(tensor, dst=1)
            dist.recv(tensor, src=1)
        elif rank == 1:
            dist.recv(tensor, src=0)
            tensor += 10.0
            dist.send(tensor, dst=0)
    dist.barrier()

    elapsed_time = time.perf_counter() - start_time
    if rank == 0:
        avg_latency = elapsed_time / NUM_ITERS
        print(f"Blocking latency: {avg_latency * 1000} miliseconds")
    dist.destroy_process_group()


def measure_latency_non_blocking(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29503"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    tensor = make_tensor(rank, device)

    # warmup
    for _ in range(NUM_WARMUP):
        if rank == 0:
            send_req = dist.isend(tensor, dst=1)
            recv_req = dist.irecv(tensor, src=1)

            send_req.wait()
            recv_req.wait()

        elif rank == 1:
            recv_req = dist.irecv(tensor, src=0)
            recv_req.wait()
            tensor += 10.0
            send_req = dist.isend(tensor, dst=0)
            send_req.wait()
    dist.barrier()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(NUM_ITERS):
        if rank == 0:
            send_req = dist.isend(tensor, dst=1)
            recv_req = dist.irecv(tensor, src=1)

            send_req.wait()
            recv_req.wait()
        elif rank == 1:
            recv_req = dist.irecv(tensor, src=0)
            recv_req.wait()
            tensor += 10.0
            send_req = dist.isend(tensor, dst=0)
            send_req.wait()
    dist.barrier()

    elapsed_time = time.perf_counter() - start_time
    if rank == 0:
        avg_latency = elapsed_time / NUM_ITERS
        print(f"Non-blocking latency: {avg_latency * 1000} miliseconds")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Measure latency of blocking and non-blocking point-to-point communication"
    )
    parser.add_argument(
        "--world-size", type=int, default=2, help="Number of processes"
    )
    parser.add_argument(
        "--backend", type=str, default="gloo", choices=["gloo", "nccl"], help="Backend to use"
    )
    args = parser.parse_args()

    mp.spawn(measure_latency_blocking, args=(args.world_size, args.backend), nprocs=args.world_size, join=True)
    mp.spawn(measure_latency_non_blocking, args=(args.world_size, args.backend), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()