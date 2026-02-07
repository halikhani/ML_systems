import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp



def compute_ring_all_reduce_worker(rank: int, world_size: int, device: torch.device, backend: str):

    # set env vars
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29507"

    # initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # create tensor
    tensor = torch.tensor([rank], device=device, dtype=torch.float32)
    print(f"Rank {rank} local tensor: {tensor.item()}")

    # first send tensor to the next rank
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    prev_sum = torch.zeros_like(tensor)
    

    # phase 1: ring reduce to rank 0
    if rank == 0:
        dist.send(tensor, dst=next_rank)
        # receive total sum from prev rank
        dist.recv(prev_sum, src=prev_rank)
    else:
        # receive sum from prev rank
        dist.recv(prev_sum, src=prev_rank)
        # send current sum to next rank
        dist.send(prev_sum + tensor, dst=next_rank)

    dist.barrier()
    # phase 2: broadcast total sum from rank 0 to all ranks
    if rank == 0:
        dist.send(prev_sum, dst=next_rank)
    else:
        dist.recv(prev_sum, src=prev_rank)
        if rank != world_size - 1:
            dist.send(prev_sum, dst=next_rank)
    print(f"Rank {rank} received total sum of all tensors: {prev_sum.item()}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--backend", type=str, default="gloo")
    args = parser.parse_args()
    device = torch.device("cpu")

    

    mp.spawn(compute_ring_all_reduce_worker, args=(args.world_size, device, args.backend), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()