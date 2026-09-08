import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def manual_all_reduce(rank: int, world_size: int, device: torch.device) -> None:
    tensor = torch.tensor([rank + 1], device=device, dtype=torch.float32)
    if rank == 0:
        # first, rank 0 receives data from all ranks
        final_tensor = tensor.clone()
        for i in range(1, world_size):
            dist.recv(tensor, src=i)
            final_tensor += tensor
        
        # now send to all ranks the final sum
        for i in range(1, world_size):
            dist.send(final_tensor, dst=i)
        
    else:
        
        final_tensor = torch.zeros(1, device=device, dtype=torch.float32)
        dist.send(tensor, dst=0)
        dist.recv(final_tensor, src=0)

    print(f"Rank {rank}: {final_tensor.item()}", flush=True)
    

def all_reduce_worker(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu")

    if rank == 0:
        print("=" * 60)
        print("Manual ALL_REDUCE")
        print("=" * 60)

    manual_all_reduce(rank, world_size, device)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--backend", type=str, default="gloo")
    args = parser.parse_args()

    mp.spawn(all_reduce_worker, args=(args.world_size, args.backend), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()
