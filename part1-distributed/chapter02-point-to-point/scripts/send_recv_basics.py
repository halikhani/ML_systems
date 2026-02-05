
# Pitfall 1: Mismatched Send/Recv

# # Process 0: sends to 1
# dist.send(tensor, dst=1)

# # Process 1: receives from 2 (WRONG!)
# dist.recv(tensor, src=2)  # Will hang forever!
# Always ensure src/dst pairs match.


# Pitfall 2: Buffer Reuse Before Completion

# handle = dist.isend(tensor, dst=1)
# tensor.fill_(0)  # DANGER! Modifying buffer during transfer
# handle.wait()


# Pitfall 3: Forgetting to Wait

# handle = dist.isend(tensor, dst=1)
# # forgot handle.wait()
# print(tensor) # corrupted data

# Key Takeaways

# Point-to-point is surgical - You specify exactly which process sends and receives
# Blocking can deadlock - Be very careful with send/recv ordering
# Non-blocking enables overlap - isend/irecv let you compute while communicating
# Pipeline parallelism uses this heavily - Activations flow forward, gradients flow backward
# Always wait() before using data - Non-blocking doesn’t mean the data is ready



"""
Basic Point-to-Point Communication: The Chain Pattern

This script demonstrates send/recv in a chain topology:
    Rank 0 → Rank 1 → Rank 2 → Rank 3

Each process receives from the previous rank, adds 10, and sends to the next.

Usage:
    python send_recv_basic.py
    python send_recv_basic.py --world-size 8

Key concepts:
- Blocking send/recv
- Chain topology (avoiding deadlocks)
- Careful ordering of operations
"""


import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def chain_worker(rank: int, world_size: int, backend: str) -> None:
    """
    Worker function implementing a chain communication pattern.

    Data flows: Rank 0 → Rank 1 → Rank 2 → ... → Rank (world_size-1)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # get device
    device = torch.device("cpu")
    if backend == "nccl" and torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # =========================================================================
    # The Chain Pattern
    # =========================================================================
    # This pattern naturally avoids deadlocks because:
    # - Rank 0 only sends (no one sends to it first)
    # - Middle ranks receive then send (in that order)
    # - Last rank only receives (no one receives from it)

    if rank == 0:
        # first process, create the init tensor and send
        tensor = torch.tensor([42.0], device=device)
        print(f"[Rank 0] Starting chain with value: {tensor.item()}")
        dist.send(tensor, dst=1)
        print(f"[Rank 0] Sent tensor to rank 1")

    elif rank == world_size - 1:
        # last process, receive and display the result
        tensor = torch.zeros(1, device=device)
        dist.recv(tensor, src=rank - 1)
        print(f"[Rank {rank}] Received final value: {tensor.item()}")
        print(f"\n{'='*50}")
        print(f"Chain complete!")
        print(f"Original: 42.0")
        print(f"After {world_size - 1} additions of 10: {tensor.item()}")
        print(f"Expected: {42.0 + (world_size - 1) * 10}")
        print(f"{'='*50}")
    
    else:
        # middle process, receive, add 10, and send
        tensor = torch.zeros(1, device=device)
        dist.recv(tensor, src=rank - 1)
        print(f"[Rank {rank}] Received: {tensor.item()}")

        tensor += 10.0
        print(f"[Rank {rank}] After adding 10: {tensor.item()}")
        dist.send(tensor, dst=rank + 1)
        print(f"[Rank {rank}] Sent to rank {rank + 1}")

    # synchronize all processes
    dist.barrier()
    dist.destroy_process_group()


def demonstrate_deadlock_pattern():
    """
    Educational function showing a deadlock pattern (DO NOT RUN).
    """
    print("""
    ⚠️  DEADLOCK PATTERN (DO NOT USE):

    # Process 0                # Process 1
    send(tensor, dst=1)        send(tensor, dst=0)
    recv(tensor, src=1)        recv(tensor, src=0)

    Both processes block on send(), waiting for the other to receive.
    Neither can proceed → DEADLOCK!

    ✓ CORRECT PATTERN (interleaved):

    # Process 0                # Process 1
    send(tensor, dst=1)        recv(tensor, src=0)
    recv(tensor, src=1)        send(tensor, dst=0)

    Process 0 sends while Process 1 receives → both can proceed.
    """)



def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate send/recv basics in a chain p2p topology"
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

    if args.show_deadlock:
        demonstrate_deadlock_pattern()
        return

    print("=" * 50)
    print(" POINT-TO-POINT COMMUNICATION: CHAIN PATTERN")
    print("=" * 50)
    print(f"World size: {args.world_size}")
    print(f"Pattern: Rank 0 → Rank 1 → ... → Rank {args.world_size - 1}")
    print(f"Operation: Each rank adds 10 before forwarding")
    print("=" * 50 + "\n")

    mp.spawn(chain_worker, args=(args.world_size, args.backend), nprocs=args.world_size, join=True) # join=True means wait for all processes to finish before returning

if __name__ == "__main__":
    main()