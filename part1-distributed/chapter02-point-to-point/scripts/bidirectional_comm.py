import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def bidirectional_comm(rank: int, world_size: int, backend: str) -> None:
    """ Worker function implementing a bidirectional communication pattern. even ranks send to odd ranks, and odd ranks send to even ranks."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Get device (CPU for gloo, GPU for nccl)
    device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    if rank % 2 == 0:
        # even ranks send to odd ranks
        peer = rank + 1
        tensor = torch.tensor([42.0], device=device)
        
        send_req = dist.isend(tensor, dst=peer)
        recv_req = dist.irecv(tensor, src=peer)

        send_req.wait()
        recv_req.wait()
        print(f"[Rank {rank}] received {tensor.item()} from {peer}")

    else:
        # all other ranks add 10
        peer = rank - 1
        tensor = torch.zeros(1, device=device)
        
        recv_req = dist.irecv(tensor, src=peer)
        recv_req.wait()
        tensor += 10.0
        send_req = dist.isend(tensor, dst=peer)
        send_req.wait()



    # Synchronize all processes before cleanup
    dist.barrier()
    dist.destroy_process_group()




def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional communication pattern"
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

    mp.spawn(bidirectional_comm, args=(args.world_size, args.backend), nprocs=args.world_size, join=True) # join=True means wait for all processes to finish before returning

if __name__ == "__main__":
    main()