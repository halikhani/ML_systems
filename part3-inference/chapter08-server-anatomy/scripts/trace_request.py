
import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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
    created_at: float = field(default_factory=time.perf_counter)

    generated_tokens: List[int] = field(default_factory=list)
    prefill_done: bool = False
    is_finished: bool = False

    # tracing
    marks: Dict[str, float] = field(default_factory=dict)  # stage boundary -> timestamp
    decode_time: float = 0.0  # time spent inside decode steps
    decode_steps: int = 0
    verbose: bool = False  # print the timeline for this request

    def log(self, component: str, event: str, mark: Optional[str] = None):
        now = time.perf_counter()
        if mark:
            self.marks[mark] = now
        if self.verbose:
            print(f"[{(now - self.created_at) * 1000:8.2f} ms] "
                  f"[{component:<11}] req {self.id}: {event}")


@dataclass
class Batch:
    """A batch of requests to process together."""
    requests: List[GenerateRequest]
    is_prefill: bool # True for prefill, False for decode


class SimpleTokenizer:
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
    def __init__(self, vocab_size: int = 1000, latency_ms: float = 10):
        self.vocab_size = vocab_size
        self.latency_ms = latency_ms


    async def prefill(self, request: GenerateRequest) -> int:
        request.log("ModelRunner", f"prefill start ({len(request.prompt_tokens)} prompt tokens)",
                    mark="prefill_start")
        # Simulate compute time (proportional to prompt length)
        prompt_len = len(request.prompt_tokens)
        await asyncio.sleep(self.latency_ms * prompt_len / 1000)
        # "Generate" first token
        first_token = random.randint(3, self.vocab_size - 1)
        request.log("ModelRunner", f"prefill end -> first token {first_token}", mark="prefill_end")
        return first_token

    async def decode(self, batch: List[GenerateRequest]) -> List[int]:
        start = time.perf_counter()
        # Simulate compute time (roughly constant per batch)
        await asyncio.sleep(self.latency_ms / 1000)
        step_time = time.perf_counter() - start

        # Generate next tokens
        next_tokens = []
        for request in batch:
            # every request in the batch pays the full step time
            request.decode_time += step_time
            request.decode_steps += 1
            request.log("ModelRunner", f"decode step {request.decode_steps} "
                        f"(batch={len(batch)}, {step_time * 1000:.2f} ms)")
            if random.random() < 0.1:
                next_tokens.append(1)  # EOS
            else:
                next_tokens.append(random.randint(3, self.vocab_size - 1))
        return next_tokens


class Scheduler:
    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size
        self.waiting_queue: deque = deque() # requests waiting for prefill
        self.running_batch: List[GenerateRequest] = [] # requests being processed in the decode phase
        self.completed: List[GenerateRequest] = [] # requests that have been completed


    def add_request(self, request: GenerateRequest):
        self.waiting_queue.append(request)
        request.log("Scheduler", f"enqueued (queue size: {len(self.waiting_queue)})", mark="enqueued")

    def get_next_batch(self) -> Optional[Batch]:
        """
        Decide what to process next.

        Strategy (simplified):
        1. If we have requests waiting AND room in running batch, do prefill
        2. If running batch has requests, do decode
        """

        # Check for finished requests first
        self.running_batch = [r for r in self.running_batch if not r.is_finished]


        if self.waiting_queue and len(self.running_batch) < self.max_batch_size:
            request = self.waiting_queue.popleft()
            request.log("Scheduler", f"picked for prefill (running batch: {len(self.running_batch)})")
            return Batch(requests=[request], is_prefill=True)

        # If not, check if we can start a new decode batch
        if self.running_batch:
            return Batch(requests=self.running_batch, is_prefill=False)

        return None

    def process_decode_result(self, request: GenerateRequest, token: int):
        request.generated_tokens.append(token)

        # check if finished
        if token == 1 or len(request.generated_tokens) >= request.max_tokens:
            request.is_finished = True
            self.completed.append(request)
            request.log("Scheduler", f"finished ({len(request.generated_tokens)} tokens)", mark="finished")

    def has_work(self) -> bool:
        return bool(self.waiting_queue or self.running_batch)


class InferenceServer:
    def __init__(self, max_batch_size: int = 4):
        self.tokenizer = SimpleTokenizer()
        self.model_runner = SimpleModelRunner()
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.request_counter = 0


    async def generate(self, prompt: str, max_tokens: int = 50, verbose: bool = False) -> GenerateRequest:
        # start the clock before tokenizing so tokenizer time is part of the request's latency
        start = time.perf_counter()
        tokens = self.tokenizer.encode(prompt)
        request = GenerateRequest(
            id=self.request_counter,
            prompt=prompt,
            prompt_tokens=tokens,
            max_tokens=max_tokens,
            created_at=start,
            verbose=verbose,
        )
        request.log("Tokenizer", f"encoded {len(tokens)} tokens", mark="tokenized")
        self.request_counter += 1
        self.scheduler.add_request(request)
        return request


    async def run_step(self):

        batch = self.scheduler.get_next_batch()

        if batch is None:
            return False

        if batch.is_prefill:
            request = batch.requests[0]

            tokens = await self.model_runner.prefill(request)
            self.scheduler.process_decode_result(request, tokens)
            request.prefill_done = True
            if not request.is_finished:
                self.scheduler.running_batch.append(request)

        else:
            # Decode
            tokens = await self.model_runner.decode(batch.requests)
            for request, token in zip(batch.requests, tokens):
                self.scheduler.process_decode_result(request, token)

        return True

    async def run_until_complete(self):
        while self.scheduler.has_work():
            await self.run_step()

        return self.scheduler.completed


def stage_breakdown(r: GenerateRequest) -> Dict[str, float]:
    """Seconds spent by one request in each stage."""
    m = r.marks
    total = m["finished"] - r.created_at
    stages = {
        "tokenize": m["tokenized"] - r.created_at,
        "queue wait": m["prefill_start"] - m["enqueued"],
        "prefill": m["prefill_end"] - m["prefill_start"],
        "decode": r.decode_time,
        # in the running batch but not decoding: the server was prefilling someone else
        "decode stall": (m["finished"] - m["prefill_end"]) - r.decode_time,
    }
    stages["other"] = total - sum(stages.values())
    stages["total"] = total
    return stages


def print_breakdown(requests: List[GenerateRequest]):
    cols = list(stage_breakdown(requests[0]).keys())
    print(f"\n{'req':>4} {'tokens':>6} " + " ".join(f"{c:>12}" for c in cols) + f" {'TTFT':>10}")
    for r in sorted(requests, key=lambda r: r.id):
        b = stage_breakdown(r)
        ttft = r.marks["prefill_end"] - r.created_at
        print(f"{r.id:>4} {len(r.generated_tokens):>6} "
              + " ".join(f"{b[c] * 1000:>9.2f} ms" for c in cols)
              + f" {ttft * 1000:>7.2f} ms")


async def run_trace(num_requests: int, max_batch_size: int, max_tokens: int):
    server = InferenceServer(max_batch_size=max_batch_size)
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What is machine learning?",
        "Tell me a joke.",
        "How does the internet work?",
    ]

    print(f"Tracing request 0 (of {num_requests}, max batch size {max_batch_size})\n")

    requests = []
    for i in range(num_requests):
        # only request 0 prints its timeline; the rest just record marks
        requests.append(await server.generate(prompts[i % len(prompts)], max_tokens=max_tokens,
                                              verbose=(i == 0)))

    await server.run_until_complete()

    print_breakdown(requests)


def main():
    parser = argparse.ArgumentParser(description="Trace a request through the minimal inference server")
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    asyncio.run(run_trace(args.num_requests, args.max_batch_size, args.max_tokens))


if __name__ == "__main__":
    main()
