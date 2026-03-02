"""=== KV Cache Calculator ===
KV Cache Calculator

Calculate KV cache memory requirements for different LLM configurations.
This helps understand memory constraints and capacity planning.

Usage:
    python kv_cache_calculator.py
    python kv_cache_calculator.py --model llama-70b
    python kv_cache_calculator.py --custom --layers 80 --heads 64 --dim 128


Model: LLaMA-70B
  Layers: 80
  KV Heads: 8 (GQA)
  Head Dim: 128
  Dtype: FP16

KV Cache Size:
  Per token: 320 KB
  Per request (4K context): 1.28 GB
  Per request (32K context): 10.24 GB

With 80 GB GPU Memory:
  Model weights (FP16): 140 GB (requires 2+ GPUs)
  After weights on 8x H100: ~480 GB available

  Max concurrent requests:
    At 4K context: 375 requests
    At 8K context: 187 requests
    At 32K context: 46 requests

Warning: Long context dramatically reduces concurrency!

kv_bytes_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes
                     ↑   ↑                             ↑
                     K+V layers                        2 for FP16



"""



import argparse
from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    """Model configuration for KV cache calculation."""
    name: str
    num_layers: int
    num_kv_heads: int  # KV heads (may differ from attention heads with GQA)
    head_dim: int
    vocab_size: int = 128000


    @property
    def kv_bytes_per_token(self) -> int:
        """Calculate KV cache bytes per token (for FP16)."""
        # K and V for each layer
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2 # 2 for FP16


# Common model configurations
MODELS = {
    "llama-7b": ModelConfig("LLaMA-7B", num_layers=32, num_kv_heads=32, head_dim=128),
    "llama-13b": ModelConfig("LLaMA-13B", num_layers=40, num_kv_heads=40, head_dim=128),
    "llama-70b": ModelConfig("LLaMA-70B", num_layers=80, num_kv_heads=8, head_dim=128),  # GQA
    "mistral-7b": ModelConfig("Mistral-7B", num_layers=32, num_kv_heads=8, head_dim=128),  # GQA
    "qwen-72b": ModelConfig("Qwen-72B", num_layers=80, num_kv_heads=8, head_dim=128),
    "deepseek-67b": ModelConfig("DeepSeek-67B", num_layers=95, num_kv_heads=8, head_dim=128),
}


def format_bytes(bytes_val: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def calculate_kv_cache(model: ModelConfig, context_lengths: list,
                       dtype_bytes: int = 2) -> Dict:
    """Calculate KV cache requirements."""
    kv_per_token = (2 * model.num_layers * model.num_kv_heads *
                    model.head_dim * dtype_bytes)

    results = {
        'model': model.name,
        'layers': model.num_layers,
        'kv_heads': model.num_kv_heads,
        'head_dim': model.head_dim,
        'kv_bytes_per_token': kv_per_token,
        'contexts': {}
    }

    for ctx_len in context_lengths:
        kv_per_request = kv_per_token * ctx_len
        results['contexts'][ctx_len] = {
            'per_request': kv_per_request,
            'per_request_formatted': format_bytes(kv_per_request),
        }

    return results


def analyze_capacity(model: ModelConfig, gpu_memory_gb: float,
                     model_size_gb: float, context_length: int,
                     dtype_bytes: int = 2) -> Dict:

    """Analyze how many concurrent requests can be served."""
    # Available memory for KV cache

    overhead_gb = 2  # CUDA kernels, activations, etc.
    available_gb = gpu_memory_gb - model_size_gb - overhead_gb

    if available_gb <= 0:
        return {
            'error': 'Model does not fit in GPU memory',
            'available_gb': available_gb,
        }

    # KV cache per request
    kv_per_token = (2 * model.num_layers * model.num_kv_heads *
                    model.head_dim * dtype_bytes)
    kv_per_request = kv_per_token * context_length
    kv_per_request_gb = kv_per_request / (1024 ** 3)

    # Max concurrent requests
    max_requests = int(available_gb / kv_per_request_gb)

    # With PagedAttention (assuming 95% utilization vs 50% without)
    requests_without_paging = int(max_requests * 0.5)  # 50% utilization
    requests_with_paging = int(max_requests * 0.95)    # 95% utilization

    return {
        'gpu_memory_gb': gpu_memory_gb,
        'model_size_gb': model_size_gb,
        'available_for_kv_gb': available_gb,
        'context_length': context_length,
        'kv_per_request_gb': kv_per_request_gb,
        'max_theoretical_requests': max_requests,
        'requests_without_paging': requests_without_paging,
        'requests_with_paging': requests_with_paging,
        'paging_improvement': f"{(requests_with_paging / requests_without_paging - 1) * 100:.0f}%"
    }


def compare_fragmentation(model: ModelConfig, requests: int,
                          avg_context: int, max_context: int,
                          dtype_bytes: int = 2) -> Dict:
    """Compare memory usage with and without paging."""
    kv_per_token = (2 * model.num_layers * model.num_kv_heads *
                    model.head_dim * dtype_bytes)

    # Without paging: allocate max_context for each request
    memory_without_paging = requests * max_context * kv_per_token

    # With paging: only allocate what's actually used
    memory_with_paging = requests * avg_context * kv_per_token

    waste = memory_without_paging - memory_with_paging
    waste_pct = (waste / memory_without_paging) * 100

    return {
        'requests': requests,
        'avg_context': avg_context,
        'max_context': max_context,
        'memory_without_paging': format_bytes(memory_without_paging),
        'memory_with_paging': format_bytes(memory_with_paging),
        'memory_wasted': format_bytes(waste),
        'waste_percentage': f"{waste_pct:.1f}%",
    }


