import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def ring_worker(rank: int, world_size: int, backend: str) -> None:
    """ Worker function implementing a ring send/recv pattern. """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Get device (CPU for gloo, GPU for nccl)
    device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    if rank == 0:
        tensor = torch.tensor([42.0], device=device)
        print(f"[Rank 0] Starting the ring with value: {tensor.item()}")
        dist.send(tensor, dst=1)
        print(f"[Rank 0] Sent to rank 1")
        # wait for receiving from the last rank
        dist.recv(tensor, src=world_size - 1)
        print(f"Finished ring with value: {tensor.item()}, expected: {42.0 + 10.0 * (world_size - 1)}")

    else:
        # all other ranks add 10
        tensor = torch.zeros(1, device=device)
        dist.recv(tensor, src=rank - 1)
        tensor += 10.0
        dist.send(tensor, dst = (rank + 1) % world_size)


    # Synchronize all processes before cleanup
    dist.barrier()
    dist.destroy_process_group()




def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate ring send recv in a p2p topology"
    )
    parser.add_argument("--world-size", type=int, default=4, help="Number of processes")
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Distributed backend"
    )
    parser.add_argument(
        "--show-deadlock", action="store_true", help="Show deadlock pattern example"
    )
    args = parser.parse_args()


    print("=" * 50)
    print(" POINT-TO-POINT COMMUNICATION: CHAIN PATTERN")
    print("=" * 50)
    print(f"World size: {args.world_size}")
    print(f"Ring Pattern: Rank 0 → Rank 1 → ... → Rank {args.world_size - 1} → Rank 0")
    print(f"Operation: Each rank adds 10 before forwarding to the next rank")
    print("=" * 50 + "\n")

    mp.spawn(ring_worker, args=(args.world_size, args.backend), nprocs=args.world_size, join=True) # join=True means wait for all processes to finish before returning

if __name__ == "__main__":
    main()