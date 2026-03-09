"""
Speculative Decoding Demonstration

This script demonstrates how speculative decoding works:
- Draft model generates multiple candidate tokens
- Target model verifies them in parallel
- Acceptance/rejection based on probability ratio

Usage:
    python speculative_demo.py
    python speculative_demo.py --draft-length 5 --acceptance-rate 0.8
"""


import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Token:
    """Represents a generated token."""
    id: int
    text: str
    draft_prob: float
    target_prob: float


def simulate_draft_model(prompt: str, num_tokens: int,
                         vocab: List[str]) -> List[Token]:
    """
    Simulate a draft model generating tokens.

    In reality, this would be a small LLM like LLaMA-7B.
    """

    tokens = []
    for _ in range(num_tokens):
        # Random token selection (simulated)
        token_id = random.randint(0, len(vocab) - 1)
        token_text = vocab[token_id]

        # Random probabilities (simulated)
        # Draft model is less confident
        draft_prob = random.uniform(0.3, 0.9)

        tokens.append(
            Token(
                id=token_id,
                text=token_text,
                draft_prob=draft_prob,
                target_prob=0  # Set by target model
            )
        )
    
    return tokens


def simulate_target_verification(draft_tokens: List[Token],
                                  base_acceptance_rate: float) -> List[Token]:
    
    """
    Simulate target model verification of draft tokens.

    In reality, this would run the large model (e.g., LLaMA-70B)
    on all draft tokens in parallel.
    """

    for token in draft_tokens:
        # Target model's probability (simulated)
        # Higher acceptance rate = closer to draft distribution
        if random.random() < base_acceptance_rate:
            # Target agrees or is more confident
            token.target_prob = token.draft_prob * random.uniform(0.9, 1.5)
        else:
            # Target disagrees
            token.target_prob = token.draft_prob * random.uniform(0.1, 0.8)

        # Clamp to valid probability
        token.target_prob = min(1.0, token.target_prob)

    return draft_tokens


def speculative_acceptance(tokens: List[Token]) -> Tuple[List[Token], Optional[Token]]:
    """
    Apply speculative decoding acceptance criterion.

    For each token:
    - If p_target >= p_draft: ACCEPT
    - Else: ACCEPT with probability p_target / p_draft
    - On first rejection: sample from adjusted distribution, stop
    """
    accepted = []
    correction_tokens = []

    for i, token in enumerate(tokens):
        if token.target_prob >= token.draft_prob:
            # Definitely accept
            accepted.append(token)
        else:
            # Probablistic acceptance
            acceptance_prob = token.target_prob / token.draft_prob
            if random.random() < acceptance_prob:
                accepted.append(token)
            else:
                # Reject: sample from (target - draft) distribution
                # Simulated as a new random token
                correction_token = Token(
                    id=random.randint(0, 99),
                    text=f"[corrected_{i}]",
                    draft_prob=0,
                    target_prob=token.target_prob
                )
                break  # Stop accepting after first rejection
    return accepted, correction_tokens