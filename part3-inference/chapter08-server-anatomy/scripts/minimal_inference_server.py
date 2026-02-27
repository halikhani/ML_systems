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
