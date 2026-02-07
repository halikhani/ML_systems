import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp



def compute_argmax(rank: int, world_size: int, device: torch.device) -> torch.Tensor:


    # first find the global argmax
    local_tensor = torch.tensor([rank + 1], device=device)
    print(f"Rank {rank} local tensor: {local_tensor.item()}")

    argmax_element = local_tensor.clone()
    dist.all_reduce(argmax_element, op=dist.ReduceOp.MAX)
    # now argmax_element contains the global argmax

    # check if the local tensor is the global argmax
    if local_tensor.item() == argmax_element.item():
        print(f"Rank {rank} has the global argmax")


def argmax_worker(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cpu")

    if rank == 0:
        print("=" * 60)
        print(" DISTRIBUTED argmax")
        print("=" * 60)

    compute_argmax(rank, world_size, device)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--backend", type=str, default="gloo")
    args = parser.parse_args()

    mp.spawn(argmax_worker, args=(args.world_size, args.backend), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()