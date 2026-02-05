import torch.multiprocessing as mp
import os

import torch.distributed as dist

def worker(rank, world_size):
    # Each process runs this function
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    dist.init_process_group(
        backend="gloo",      # Communication backend
        init_method="env://",# Use env vars for rendezvous
        world_size=world_size,
        rank=rank
    )
    print(f"hello from rank {rank}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(worker, args=(world_size,), nprocs=world_size)

# mp.spawn():

# Creates world_size new processes
# Calls worker(rank, world_size) in each process
# Passes rank as the first argument automatically

