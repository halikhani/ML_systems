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
from calendar import c
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



class SimpleTokenizer:
    """
    A simplified tokenizer for demonstration.

    Real tokenizers (like SentencePiece or tiktoken) are more complex.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # Simple word-based tokenization
        self.token_to_id = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        self.id_to_token = {0: "<pad>", 1: "<eos>", 2: "<unk>"}


    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        # simple: assign random ids to each word
        words = text.lower().split()
        tokens = []
        for word in words:
            # Hash word to get consistent token ID
            token_id = hash(word) % (self.vocab_size - 3) + 3 # reserve 3 for special tokens
            tokens.append(token_id)
        return tokens
    
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        # Simplified: just return placeholder
        return f"[Generated {len(token_ids)} tokens]"


class SimpleModelRunner:
    """
    A simplified model runner for demonstration.

    Real model runners load actual neural networks and run GPU inference.
    """


    def __init__(self, vocab_size: int = 1000, latency_ms: float = 10):
        self.vocab_size = vocab_size
        self.latency_ms = latency_ms

    
    async def prefill(self, request: GenerateRequest) -> int:
        """
        Process prompt and return first generated token.

        Real prefill:
        1. Run all prompt tokens through model in parallel
        2. Build KV cache for all tokens
        3. Sample first output token
        """
        # Simulate compute time (proportional to prompt length)
        prompt_len = len(request.prompt_tokens)
        await asyncio.sleep(self.latency_ms * prompt_len / 100)
        # "Generate" first token
        first_token = random.randint(3, self.vocab_size - 1)
        return first_token

    
    async def decode(self, batch: List[GenerateRequest]) -> List[int]:
        """
        Generate next token for each request in batch.

        Real decode:
        1. Run single token through model for each request
        2. Update KV cache with new KV pairs
        3. Sample next token for each request
        """

        # Simulate compute time (roughly constant per batch)
        await asyncio.sleep(self.latency_ms)

        # Generate next tokens
        next_tokens = []
        for request in batch:
            # 10% chance of generating EOS
            if random.random() < 0.1:
                next_tokens.append(1)  # EOS
            else:
                next_tokens.append(random.randint(3, self.vocab_size - 1))
        return next_tokens


class Scheduler:
    """
    Manages request queue and batching decisions.

    Key responsibilities:
    1. Accept new requests
    2. Decide which requests to process together
    3. Manage prefill vs decode scheduling
    """


    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size
        self.waiting_queue: deque = deque() # requests waiting for prefill
        self.running_batch: List[GenerateRequest] = [] # requests being processed in the decode phase
        self.completed: List[GenerateRequest] = [] # requests that have been completed


    def add_request(self, request: GenerateRequest):
        """Add a new request to the waiting queue."""
        self.waiting_queue.append(request)
        print(f"[Scheduler] Added request {request.id} to queue "
              f"(queue size: {len(self.waiting_queue)})")

    
    def get_next_batch(self) -> Optional[Batch]:
        """
        Decide what to process next.

        Strategy (simplified):
        1. If we have requests waiting AND room in running batch, do prefill
        2. If running batch has requests, do decode
        """
        # Check for finished requests first
        self.running_batch = [r for r in self.running_batch if not r.is_finished]

        # Prefill new requests if we have capacity
        while self.waiting_queue and len(self.running_batch) < self.max_batch_size:
            request = self.waiting_queue.popleft()
            return Batch(requests=[request], is_prefill=True)

        # Decode existing requests
        if self.running_batch:
            return Batch(requests=self.running_batch, is_prefill=False)

        return None

    
    def process_decode_result(self, request: GenerateRequest, token: int):
        """Handle result from decode."""
        request.generated_tokens.append(token)
        
        # check if finished
        if token == 1 or len(request.generated_tokens) >= request.max_tokens:
            request.is_finished = True
            self.completed.append(request)
            print(f"[Scheduler] Request {request.id} finished "
                  f"({len(request.generated_tokens)} tokens)")


    def has_work(self) -> bool:
        """Check if there's more work to do."""
        return bool(self.waiting_queue or self.running_batch)



class InferenceServer:
    """
    Main inference server orchestrating all components.
    """

    def __init__(self, max_batch_size: int = 4):
        self.tokenizer = SimpleTokenizer()
        self.model_runner = SimpleModelRunner()
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.request_counter = 0

    
    async def generate(self, prompt: str, max_tokens: int = 50) -> GenerateRequest:
        """Submit a generation request."""
        # NOTE: generate() can be called from an async server 
        # (e.g., handling multiple concurrent requests) without blocking; it returns quickly after enqueueing work.
        # tokenize
        tokens = self.tokenizer.encode(prompt)

        # create request
        request = GenerateRequest(
            id=self.request_counter,
            prompt=prompt,
            prompt_tokens=tokens,
            max_tokens=max_tokens,
        )
        self.request_counter += 1

        # submit to scheduler
        self.scheduler.add_request(request)
        return request


    async def run_step(self):
        """Run one step of inference."""
        # NOTE: run_step() awaits model_runner.prefill/decode, 
        # which likely represent I/O-bound or long-running compute. 
        # Making these awaitable lets the event loop interleave other tasks 
        # (like accepting new requests or handling responses) while a step runs.
        batch = self.scheduler.get_next_batch()
        if batch is None:
            return False

        if batch.is_prefill:
            # Prefill phase
            request = batch.requests[0]
            print(f"[Server] Prefill request {request.id} "
                  f"({len(request.prompt_tokens)} prompt tokens)")
            
            tokens = await self.model_runner.prefill(request)
            self.scheduler.process_decode_result(request, tokens)

        else:
            # Decode phase
            print(f"[Server] Decode batch of {len(batch.requests)} requests")

            tokens = await self.model_runner.decode(batch.requests)
            for request, token in zip(batch.requests, tokens):
                self.scheduler.process_decode_result(request, token)

        return True


    async def run_until_complete(self):
        """Run until all requests are complete."""
        # NOTE: run_until_complete() awaits each step so the loop cooperates 
        # with the event loop rather than busy-waiting.
        while self.scheduler.has_work():
            await self.run_step()




