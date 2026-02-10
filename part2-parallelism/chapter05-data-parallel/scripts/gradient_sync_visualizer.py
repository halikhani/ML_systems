"""
Gradient Synchronization Visualizer

This script visualizes how gradient synchronization works in distributed
training. It shows gradients before and after synchronization, demonstrating
the averaging that makes data parallelism work.

Key insights:
- Each rank computes different gradients (different data)
- After all_reduce + averaging, all ranks have identical gradients
- This is mathematically equivalent to training on the full batch

Usage:
    python gradient_sync_visualizer.py
    python gradient_sync_visualizer.py --verbose
"""


import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp


class TinyModel(nn.Module):
    """A tiny model for visualization purposes."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def print_gradients(model: nn.Module, rank: int, prefix: str = "") -> Dict[str, torch.Tensor]:
    """Print gradients for all parameters."""
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
            if rank == 0:
                print(f"  {prefix}{name}:")
                print(f"    shape: {list(param.grad.shape)}")
                print(f"    grad[0,0]: {param.grad[0,0].item():.6f}")
    return gradients


def visualize_sync(rank: int, world_size: int, verbose: bool) -> None:
    """Main visualization function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29507"

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Setup: Create identical models on all ranks
    # =========================================================================

    torch.manual_seed(42)
    model = TinyModel().to(device)

    # Broadcast weights to ensure identical starting point
    # NOTE: dist.broadcast writes into each param in place, and these param require grad (model params)
    # Pytorch forbids in-place ops on such tensors, so we need to use torch.no_grad() to avoid this.
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(" GRADIENT SYNCHRONIZATION VISUALIZATION")
        print("=" * 60)
        print(f"\nWorld size: {world_size}")
        print(f"Model: {model}")

    
    # =========================================================================
    # Step 1: Each rank processes DIFFERENT data
    # =========================================================================
    dist.barrier()

    if rank == 0:
        print("\n" + "-" * 60)
        print(" STEP 1: Each rank has different input data")
        print("-" * 60)

    
    # Create rank-specific data (simulating distributed batch)
    torch.manual_seed(42 + rank) # different seed for each rank

    local_input = torch.randn(8, 4, device=device) # batch of size 8, 4 features
    local_target = torch.randint(2, (8,), device=device) # batch of size 8, 2 classes

    dist.barrier()

    # =========================================================================
    # Step 2: Forward and backward (compute LOCAL gradients)
    # =========================================================================
    if rank == 0:
        print("\n" + "-" * 60)
        print(" STEP 2: Compute gradients LOCALLY (before sync)")
        print("-" * 60)

    outputs = model(local_input)
    loss = ((outputs - local_target).pow(2).mean()) # MSE loss
    model.zero_grad()
    loss.backward()

    dist.barrier()

    # Show gradients before sync
    print(f"\n[Rank {rank}] Loss: {loss.item():.6f}")

    # Collect pre-sync gradients
    pre_sync_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            pre_sync_grads[name] = param.grad.clone()
            if verbose:
                print(f"[Rank {rank}] {name} grad[0,0]: {param.grad[0,0].item():.6f}")

    dist.barrier()

    if rank == 0:
        print("\n[Note] Gradients are DIFFERENT on each rank because")
        print("       each rank processed different input data!")

    # =========================================================================
    # Step 3: Synchronize gradients (all_reduce + average)
    # =========================================================================
    dist.barrier()

    if rank == 0:
        print("\n" + "-" * 60)
        print(" STEP 3: Synchronize gradients (all_reduce + average)")
        print("-" * 60)


    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size
    
    dist.barrier()

    # Show gradients after sync
    post_sync_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            post_sync_grads[name] = param.grad.clone()
            if verbose:
                print(f"[Rank {rank}] {name} grad[0,0]: {param.grad[0,0].item():.6f}")

    dist.barrier()

    if rank == 0:
        print("\n[Note] After sync, ALL ranks have IDENTICAL gradients!")
        print("       These are the averaged gradients from all local batches.")

    # =========================================================================
    # Step 4: Verify all ranks have identical gradients
    # =========================================================================
    dist.barrier()

    if rank == 0:
        print("\n" + "-" * 60)
        print(" STEP 4: Verify gradient synchronization")
        print("-" * 60)

    
    # Gather gradients from all ranks to rank 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_flattened = param.grad.flatten()
            gathered = [torch.zeros_like(grad_flattened) for _ in range(world_size)]
            dist.all_gather(gathered, grad_flattened)

            if rank == 0:
                # Check all ranks have identical gradients
                all_same = all(torch.allclose(gathered[0], g) for g in gathered[1:])
                status = "✓" if all_same else "✗"
                print(f"  {status} {name}: all ranks identical = {all_same}")


    # =========================================================================
    # Step 5: Mathematical verification
    # =========================================================================
    dist.barrier()

    if rank == 0:
        print("\n" + "-" * 60)
        print(" MATHEMATICAL INSIGHT")
        print("-" * 60)
        print("""
The synchronized gradient is mathematically equivalent to computing
the gradient on the ENTIRE distributed batch:

  Let B = B₀ ∪ B₁ ∪ B₂ ∪ B₃ (union of all local batches)

  ∇L(B) = (1/|B|) Σᵢ ∇L(xᵢ)

        = (1/4) [∇L(B₀) + ∇L(B₁) + ∇L(B₂) + ∇L(B₃)]

        = all_reduce(local_gradients, SUM) / world_size

This is why data parallelism gives the SAME result as training
on a single GPU with a larger batch size!
""")
    # =========================================================================
    # Bonus: Show gradient change magnitude
    # =========================================================================
    dist.barrier()

    if verbose and rank == 0:
        print("-" * 60)
        print(" GRADIENT CHANGE ANALYSIS")
        print("-" * 60)

        print("\nHow much did gradients change after sync?")
        print("(This shows how different each rank's gradients were)\n")

    dist.barrier()

    for name in pre_sync_grads:
        pre_grad = pre_sync_grads[name]
        post_grad = post_sync_grads[name]
        change = (post_grad - pre_grad).abs().mean().item()
        change_pct = change / (pre_grad.abs().mean().item() + 1e-8) * 100

        if verbose:
            print(f"[Rank {rank}] {name}: changed by {change_pct:.1f}%")

    
    dist.barrier()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Gradient Sync Visualizer")
    parser.add_argument("--world-size", "-w", type=int, default=4)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed gradient values")
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║" + " GRADIENT SYNCHRONIZATION VISUALIZER".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    mp.spawn(
        visualize_sync,
        args=(args.world_size, args.verbose),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
