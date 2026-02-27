"""┌─────────────────────────────────────────────────┐
│           Minimal Inference Server               │
│                                                  │
│  Request Queue ──► Batcher ──► Model ──► Output │
│                                                  │
│  Components:                                     │
│  - RequestQueue: FIFO queue for incoming prompts│
│  - SimpleBatcher: Groups requests for GPU        │
│  - MockModel: Simulates forward pass            │
│  - Generator: Token-by-token output loop        │
└─────────────────────────────────────────────────┘


This script demonstrates the core components of an inference server:
- Request management
- Simple batching
- Token generation loop

This is educational, not production-ready. Real servers like vLLM and
SGLang have much more sophisticated implementations.

Usage:
    python minimal_inference_server.py
    python minimal_inference_server.py --num-requests 10

"""


import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, AsyncIterator
from collections import deque
import random


@dataclass
class GenerateRequest:
    """A request to generate text."""
    id: int
    prompt: str
    prompt_tokens: List[int]
    max_tokens: int = 50
    temperature: float = 1.0
    created_at: float = field(default_factory=time.time)

    # tracking
    generated_tokens: List[int] = field(default_factory=list)
    is_finished: bool = False
    prefill_done: bool = False


@dataclass
class Batch:
    """A batch of requests to process together."""
    requests: List[GenerateRequest]
    is_prefill: bool # True for prefill, False for decode


