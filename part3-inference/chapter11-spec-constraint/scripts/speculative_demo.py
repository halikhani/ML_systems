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
