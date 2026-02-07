"""
Simple DDP Implementation

This script shows two approaches to data-parallel training:
1. Manual gradient synchronization (educational)
2. PyTorch's DDP wrapper (production)

Understanding the manual approach helps you appreciate what DDP does
automatically and why it's optimized the way it is.

Usage:
    python simple_ddp.py
    python simple_ddp.py --epochs 10 --batch-size 64
"""

import argparse
import os
import time
from typing import Tuple


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class SimpleModel(nn.Module):
    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_dataset(num_samples: int, input_size: int, num_classes: int) -> TensorDataset:
    """Create a dummy dataset for testing."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def manual_gradient_sync(model: nn.Module, world_size: int) -> None:
    """
    Manually synchronize gradients across all processes.

    This is what DDP does automatically (but more efficiently).
    """
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size


def train_manual(
    rank: int,
    world_size: int,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ) -> Tuple[float, float]:
    """
    Train for one epoch with MANUAL gradient synchronization.

    This is educational - showing exactly what DDP automates.
    """
    model.train()
    total_loss = 0.0
    sync_time = 0.0

    for batch_idx, (data, targets) in enumerate(dataloader):
        # move data to device
        data, targets = data.to(device), targets.to(device)

        # Forward pass (local)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        # manula gradient sync
        sync_start = time.perf_counter()
        manual_gradient_sync(model, world_size)
        sync_time += time.perf_counter() - sync_start

        # optimizer step (local, but with averaged gradients)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), sync_time


def train_ddp(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    ) -> float:
    """
    Train for one epoch with PyTorch DDP.

    DDP automatically handles gradient synchronization during backward().
    """

    model.train()
    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        # DDP hooks into backward() to automatically sync gradients
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def compare_gradients(model1: nn.Module, model2: nn.Module) -> float:
    """Compare gradients between two models (should be identical after sync)."""
    max_diff = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        diff = (param1.grad - param2.grad).abs().max()
        max_diff = max(max_diff, diff)
    return max_diff


def worker(
    rank: int,
    world_size: int, 
    args: argparse.Namespace,
) -> None:
    """Worker function for each process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # create dataset and distributed sampler

    dataset = create_dummy_dataset(
        num_samples=1000,
        input_size=784,
        num_classes=10
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # =========================================================================
    # Method 1: Manual Gradient Sync (Educational)
    # =========================================================================

    if rank == 0:
        print("\n" + "=" * 60)
        print(" METHOD 1: MANUAL GRADIENT SYNCHRONIZATION")
        print("=" * 60)

    # create model (same initialization on all ranks via seeding)
    torch.manual_seed(42)
    model_manual = SimpleModel().to(device)
    # Broadcast initial weights from rank 0 to ensure all replicas start identical
    for param in model_manual.parameters():
        dist.broadcast(param, src=0)

    optimizer_manual = optim.SGD(model_manual.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dist.barrier()

    # train on one epoch manually
    manual_loss, manual_sync_time = train_manual(
        rank=rank,
        world_size=world_size,
        model=model_manual,
        dataloader=dataloader,
        optimizer=optimizer_manual,
        criterion=criterion,
        device=device,
    )

    dist.barrier()

    if rank == 0:
        print(f"\n[Manual] Loss: {manual_loss:.4f}")
        print(f"[Manual] Time spent in gradient sync: {manual_sync_time*1000:.2f} ms")

    # =========================================================================
    # Method 2: PyTorch DDP (Production)
    # =========================================================================

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(" METHOD 2: PYTORCH DDP (AUTOMATIC)")
        print("=" * 60)

    # Create fresh model with same seed
    torch.manual_seed(42)
    model_ddp = SimpleModel().to(device)
    # Wrap with DDP - this enables automatic gradient sync
    model_ddp = DDP(model_ddp)

    optimizer_ddp = optim.SGD(model_ddp.parameters(), lr=0.01)

    # reset the sampler for a new epoch
    sampler.set_epoch(0)
    # DistributedSampler uses the epoch number to seed its shuffle, 
    # so every rank gets a different, non-overlapping slice, 
    # but all ranks agree on the same global order. (here we only have 1 epoch so it doesn't matter)

    dist.barrier()

    # train one epoch with DDP
    start_time = time.perf_counter()
    ddp_loss = train_ddp(
        model=model_ddp,
        dataloader=dataloader,
        optimizer=optimizer_ddp,
        criterion=criterion,
        device=device,
    )
    ddp_time = time.perf_counter() - start_time

    dist.barrier()

    if rank == 0:
        print(f"\n[DDP] Loss: {ddp_loss:.4f}")
        print(f"[DDP] Total training time: {ddp_time*1000:.2f} ms")

    # =========================================================================
    # Comparison
    # =========================================================================
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print(" COMPARISON")
        print("=" * 60)

        print(f"""
What DDP does that our manual approach doesn't:

1. GRADIENT BUCKETING
   - Groups small gradients into larger buffers
   - Reduces number of all_reduce calls
   - Our manual: one all_reduce per parameter

2. OVERLAP WITH BACKWARD
   - Starts all_reduce before backward completes
   - Hides communication latency
   - Our manual: all_reduce only after full backward

3. SMART BUFFER MANAGEMENT
   - Reuses communication buffers
   - Avoids memory allocation overhead
   - Our manual: allocates on each call

4. BROADCAST ON FIRST FORWARD
   - Ensures consistent initialization
   - We did this manually with broadcast

Why DDP is faster:
   - Fewer, larger all_reduce calls (bucketing)
   - Communication overlapped with computation
   - Highly optimized NCCL integration
""")

    dist.barrier()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Simple DDP Implementation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--world-size", "-w", type=int, default=4)
    args = parser.parse_args()

    print("╔" + "═" * 58 + "╗")
    print("║" + " SIMPLE DDP: MANUAL vs AUTOMATIC".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\nWorld size: {args.world_size}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Effective batch size: {args.batch_size * args.world_size}")


    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()


