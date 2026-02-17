"""
Parallel Strategy Calculator

Given model specifications and hardware constraints, this script helps
you determine the optimal parallelism strategy.

It calculates:
- Memory requirements for different strategies
- Communication volumes
- Recommended configuration

Usage:
    python parallel_strategy_calculator.py
    python parallel_strategy_calculator.py --params 70 --gpus 64 --memory 80


example output:
=== Parallelism Strategy Calculator ===

Model: 70B parameters, 80 layers, hidden=8192
Hardware: 64 GPUs (8 per node), 80GB each, NVLink intra-node

Configuration options:

| Strategy      | TP | PP | DP | Memory/GPU | Comm Volume | Feasible? |
|---------------|----|----|----| -----------|-------------|-----------|
| Pure DP       | 1  | 1  | 64 | 840 GB     | 2D/step     | No        |
| TP=8, DP=8    | 8  | 1  | 8  | 105 GB     | 8D/layer    | No        |
| TP=8, PP=2    | 8  | 2  | 4  | 52 GB      | 8D/layer    | Yes       |
| TP=8, PP=4    | 8  | 4  | 2  | 26 GB      | 8D/layer    | Yes (rec) |

Recommended: TP=8 (within node), PP=4 (across nodes), DP=2

Reasoning:
- TP=8 uses NVLink bandwidth efficiently
- PP=4 distributes 80 layers across 4 stages (20 layers each)
- DP=2 provides batch parallelism for throughput
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class ModelConfig:
    """Model configuration."""
    params_billions: float
    hidden_size: int = 8192
    num_layers: int = 80
    num_heads: int = 64
    vocab_size: int = 128000
    intermediate_ratio: float = 4.0
    is_moe: bool = False
    num_experts: int = 1
    top_k: int = 1 # experts activated per token


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    num_gpus: int
    memory_per_gpu_gb: int
    intra_node_bandwidth_gbps: int = 900 # NVLink
    inter_node_bandwidth_gbps: int = 50 # InfiniBand
    gpus_per_node: int = 8


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 1024
    sequence_length: int = 4096
    dtype_bytes: int = 2 # fp16 or bf16



def estimate_model_memory(config: ModelConfig, dtype_bytes: int = 2) -> dict:
    """Estimate memory requirements for model parameters."""
    # Parameter count estimation
    # Embedding: vocab_size * hidden_size (matrix to convert embeddings to hidden states)
    # Per layer: 4 * hidden_size^2 (QKV and O) + 2 * hidden_size * intermediate_size (FFN)

    embedding_params = config.vocab_size * config.hidden_size
    attention_params = 4 * config.hidden_size**2  # Q, K, V, O projections
    ffn_params = 2 * config.hidden_size * int(config.hidden_size * config.intermediate_ratio)

    if config.is_moe:
        # MoE: multiply FFN by number of experts
        ffn_params *= config.num_experts

    
    layer_params = attention_params + ffn_params
    total_params = embedding_params + layer_params * config.num_layers

    # convert to bytes
    param_bytes = total_params * dtype_bytes
    gradient_bytes = param_bytes #  same size for gradients
    optimizer_state_bytes = 2 * param_bytes # Adam optimizer FP32

    return {
        'params': param_bytes / 1024**3, # GB
        'gradient': gradient_bytes / 1024**3, # GB
        'optimizer': optimizer_state_bytes / 1024**3, # GB
        'total': (param_bytes + gradient_bytes + optimizer_state_bytes) / 1024**3, # GB
        'param_count': total_params,
    }


def estimate_activation_memory(
    config: ModelConfig,
    training: TrainingConfig,
    tp_degree: int = 1,
    pp_degree: int = 1,
) -> float:

    """Estimate activation memory per GPU in GB."""
    batch_per_gpu = training.batch_size // tp_degree
    seq_len = training.sequence_length
    hidden = config.hidden_size


    # per layer activations (simplified)
    # Input to attention, attention output, FFN intermediate, etc.

    activation_per_layer = 10 * batch_per_gpu * seq_len * hidden // tp_degree

    layers_per_stage = config.num_layers // pp_degree

    total_activations_bytes = activation_per_layer * layers_per_stage * training.dtype_bytes

    return total_activations_bytes / 1024**3 # GB


def calculate_communication_volume(
    config: ModelConfig,
    training: TrainingConfig,
    tp_degree: int,
    dp_degree: int,
    pp_degree: int
) -> dict:
    """Calculate communication volume for different parallelism types."""
    batch = training.batch_size
    seq = training.sequence_length
    hidden = config.hidden_size
    dtype = training.dtype_bytes


    # TP communication: all_reduce per layer
    # 2 all_reduce per transformer layer(attention + ffn)
    tp_volume_per_layer = 4 * batch * seq * hidden * dtype * (tp_degree - 1) / tp_degree
    # NOTE: 4 for QKV and O, batch * seq * hidden for size of activation tensor per layer
    # NOTE: (tp_degree - 1) / tp_degree is the fraction of data communication in an all_reduce across all GPUs

    tp_volume_total = tp_volume_per_layer * config.num_layers / pp_degree

    # PP communication: activations between stages
    pp_volume = 2 * batch * seq * hidden * dtype  # Forward and backward

    # DP communication: gradients all_reduce
    params_per_stage = config.params_billions * 1e9 / pp_degree
    dp_volume = 2 * params_per_stage * dtype * (dp_degree - 1) / dp_degree

    return {
        'tp_per_step_gb': tp_volume_total / 1e9,
        'pp_per_step_gb': pp_volume / 1e9,
        'dp_per_step_gb': dp_volume / 1e9,
        'total_gb': (tp_volume_total + pp_volume + dp_volume) / 1e9,
    }


def find_optimal_strategy(
    model: ModelConfig,
    hardware: HardwareConfig,
    training: TrainingConfig
) -> dict:
    """Find optimal parallelism strategy given constraints."""
    pass

    













