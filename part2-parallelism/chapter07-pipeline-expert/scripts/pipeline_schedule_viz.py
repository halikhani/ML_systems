"""
Pipeline Schedule Visualizer

This script visualizes different pipeline parallelism scheduling strategies:
- Naive (fill-drain): Simple but high peak memory
- 1F1B: Interleaved forward/backward for lower memory
- Shows bubble ratios and GPU utilization

Both charts come from one dependency-driven simulator (see `_simulate`),
so the pictures cannot drift from the rules they claim to illustrate.
Cost model: every F and every B takes exactly 1 time unit, communication
is free. (Real backward is ~2x forward; that changes the numbers below,
not the shape of the schedule.)

Usage:
    python pipeline_schedule_viz.py
    python pipeline_schedule_viz.py --stages 4 --microbatches 8


Example output (P=4, M=8):

=== Naive Fill-Drain Schedule ===

Time →    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
GPU 0:   F0 F1 F2 F3 F4 F5 F6 F7  .  .  .  .  .  . B0 B1 B2 B3 B4 B5 B6 B7
GPU 1:    . F0 F1 F2 F3 F4 F5 F6 F7  .  .  .  . B0 B1 B2 B3 B4 B5 B6 B7  .
GPU 2:    .  . F0 F1 F2 F3 F4 F5 F6 F7  .  . B0 B1 B2 B3 B4 B5 B6 B7  .  .
GPU 3:    .  .  . F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7  .  .  .

Bubble fraction: 27% (6 idle slots per GPU out of 22 = 2(P-1) / 2(M+P-1))
                     ((P-1)/(M+P-1) = 3/11 ~ 0.27)
Peak memory: 8 microbatches of activations (= M)

=== 1F1B Schedule ===

Time →    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
GPU 0:   F0 F1 F2 F3  .  .  . B0 F4 B1 F5 B2 F6 B3 F7 B4  . B5  . B6  . B7
GPU 1:    . F0 F1 F2  .  . B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5  . B6  . B7  .
GPU 2:    .  . F0 F1  . B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6  . B7  .  .
GPU 3:    .  .  . F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7  .  .  .

Bubble fraction: 27% -- IDENTICAL to naive. Same makespan, same idle slots.
Peak memory: 4 microbatches of activations (= P, not M!)

Read the two charts side by side: the win is memory, not time. 1F1B pulls
each B as early as its dependencies allow, so GPU 0 retires B0 at t=7
instead of t=14 and never holds more than P activations. The bubbles are
just moved around (spread through the middle) rather than removed.

Three invariants hold for BOTH schedules, and are asserted in `_render`:
  - makespan        == 2(M + P - 1)
  - idle per stage  == 2(P - 1)
  - the last stage is dense (zero internal bubbles) under 1F1B
"""

import argparse
from typing import Dict, List, Tuple

# An op is ("F", microbatch) or ("B", microbatch).
Op = Tuple[str, int]


def _stage_op_order(stages: int, microbatches: int, schedule: str) -> List[List[Op]]:
    """
    The fixed sequence of ops each stage executes, in order.

    This is the ONLY thing that differs between the two schedules -- the
    timing falls out of the dependencies afterwards.

    naive: all M forwards, then all M backwards.
    1F1B:  stage s warms up with w = min(P-1-s, M) forwards, then alternates
           F/B, then drains the w backwards it still owes. Deeper stages warm
           up less: the last stage starts its 1F1B rhythm immediately.
    """
    orders: List[List[Op]] = []
    for stage in range(stages):
        if schedule == "naive":
            ops: List[Op] = [("F", m) for m in range(microbatches)]
            ops += [("B", m) for m in range(microbatches)]
        else:
            warmup = max(0, min(stages - 1 - stage, microbatches))
            ops = [("F", m) for m in range(warmup)]
            for i in range(microbatches - warmup):        # steady state: 1F1B
                ops += [("F", warmup + i), ("B", i)]
            ops += [("B", m) for m in range(microbatches - warmup, microbatches)]
        orders.append(ops)
    return orders


def _simulate(stages: int, microbatches: int, schedule: str) -> Dict[Tuple[int, Op], int]:
    """
    Earliest-start times under the pipeline data dependencies.

    Dependencies (each op takes 1 unit):
      F(m, s) needs F(m, s-1)   -- activations flow forward,  stage 0 is free
      B(m, s) needs B(m, s+1)   -- gradients flow backward
      B(m, P-1) needs F(m, P-1) -- the last stage turns the microbatch around

    Plus the per-stage ordering from `_stage_op_order` (a GPU does one thing
    at a time). Start times are a fixed point, so we just repeatedly schedule
    whatever op has both its predecessors already placed.
    """
    orders = _stage_op_order(stages, microbatches, schedule)
    start: Dict[Tuple[int, Op], int] = {}
    next_op = [0] * stages
    total_ops = sum(len(o) for o in orders)

    while len(start) < total_ops:
        progress = False
        for stage in range(stages):
            i = next_op[stage]
            if i >= len(orders[stage]):
                continue
            kind, mb = orders[stage][i]

            # Can't start before this GPU finished its previous op.
            t = 0 if i == 0 else start[(stage, orders[stage][i - 1])] + 1

            if kind == "F":
                dep = (stage - 1, ("F", mb)) if stage > 0 else None
            elif stage < stages - 1:
                dep = (stage + 1, ("B", mb))
            else:
                dep = (stage, ("F", mb))

            if dep is not None:
                if dep not in start:
                    continue                     # upstream not placed yet
                t = max(t, start[dep] + 1)

            start[(stage, (kind, mb))] = t
            next_op[stage] = i + 1
            progress = True

        if not progress:                         # would mean a cyclic order
            raise RuntimeError(f"deadlock in {schedule} schedule")

    return start


def _peak_memory(start: Dict[Tuple[int, Op], int], stages: int) -> int:
    """
    Max activations in flight on any stage: forwards done minus backwards done,
    maximised over time. This is measured from the schedule, not asserted.
    """
    peak = 0
    for stage in range(stages):
        ops = sorted(((t, k) for (s, (k, _)), t in start.items() if s == stage))
        live = 0
        for _, kind in ops:
            live += 1 if kind == "F" else -1
            peak = max(peak, live)
    return peak


def _render(stages: int, microbatches: int, schedule: str,
            max_cols: int = 40) -> Tuple[str, float, int]:
    """Draw the Gantt chart and return (chart, bubble_ratio, peak_memory)."""
    start = _simulate(stages, microbatches, schedule)
    makespan = max(start.values()) + 1

    grid = [["."] * makespan for _ in range(stages)]
    for (stage, (kind, mb)), t in start.items():
        assert grid[stage][t] == ".", f"two ops on GPU {stage} at t={t}"
        grid[stage][t] = f"{kind}{mb}"

    # Invariants -- if these ever trip, the chart above is lying.
    assert makespan == 2 * (microbatches + stages - 1)
    for row in grid:
        assert row.count(".") == 2 * (stages - 1)
    if schedule == "1f1b":
        last = grid[-1]
        busy = [t for t, x in enumerate(last) if x != "."]
        assert busy[-1] - busy[0] + 1 == len(busy), "last stage must be dense"

    shown = min(makespan, max_cols)
    truncated = makespan > shown
    out = ["Time →  " + "".join(f"{t:>3}" for t in range(shown)),
           "-" * (8 + shown * 3)]
    for stage, row in enumerate(grid):
        out.append(f"GPU {stage}:  " + "".join(f"{x:>3}" for x in row[:shown])
                   + ("..." if truncated else ""))
    if truncated:
        out.append(f"(showing first {shown} of {makespan} time steps)")

    bubble_ratio = (makespan - 2 * microbatches) / makespan
    return "\n".join(out), bubble_ratio, _peak_memory(start, stages)


def visualize_naive_schedule(stages: int, microbatches: int) -> Tuple[str, float, int]:
    """
    Visualize naive fill-drain pipeline schedule.

    In naive scheduling:
    1. Forward all microbatches through the pipeline
    2. Then backward all microbatches

    Every stage holds all M activations until the drain reaches it, so peak
    memory is M. Bubble ratio is (P-1)/(M+P-1) -- one big fill, one big drain.
    """
    return _render(stages, microbatches, "naive")


def visualize_1f1b_schedule(stages: int, microbatches: int) -> Tuple[str, float, int]:
    """
    Visualize 1F1B (One Forward, One Backward) pipeline schedule.

    Key insight: after a short warmup, each stage does 1F then 1B, so a
    backward retires an activation as fast as a forward creates one.

    Peak memory = warmup depth of stage 0 = min(P, M), i.e. P once M >= P,
    and independent of M after that.
    Bubble ratio is unchanged from naive: (P-1)/(M+P-1).
    """
    return _render(stages, microbatches, "1f1b")


def analyze_schedules(stages: int, microbatches: int) -> None:
    """Compare different scheduling strategies."""
    print("=" * 70)
    print(" PIPELINE SCHEDULE COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration: {stages} stages, {microbatches} microbatches\n")

    # Naive schedule
    print("-" * 70)
    print(" NAIVE (Fill-Drain) SCHEDULE")
    print("-" * 70)
    print("""
Strategy: Complete all forwards, then all backwards.
Memory: Must store activations for ALL microbatches.
""")
    naive_viz, naive_bubble, naive_memory = visualize_naive_schedule(stages, microbatches)
    print(naive_viz)
    print(f"\nBubble ratio: {naive_bubble:.1%}")
    print(f"Peak activation memory: {naive_memory} microbatches worth")

    print("\n")

    # 1F1B schedule
    print("-" * 70)
    print(" 1F1B (One Forward, One Backward) SCHEDULE")
    print("-" * 70)
    print("""
Strategy: After warmup, alternate 1 forward then 1 backward.
Memory: Only store activations for 'stages' microbatches.
""")
    fb_viz, fb_bubble, fb_memory = visualize_1f1b_schedule(stages, microbatches)
    print(fb_viz)
    print(f"\nBubble ratio: {fb_bubble:.1%}")
    print(f"Peak activation memory: {fb_memory} microbatches worth")

    # Comparison
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"""
{'Metric':<25} {'Naive':<20} {'1F1B':<20}
{'-'*65}
{'Bubble ratio':<25} {naive_bubble:<20.1%} {fb_bubble:<20.1%}
{'Peak memory':<25} {naive_memory:<20} {fb_memory:<20}

Key insights:
1. 1F1B has the SAME bubble ratio but LOWER memory
2. More microbatches → lower bubble ratio (approaches 0 as M→∞)
3. Peak memory in 1F1B is bounded by pipeline depth
""")



def demonstrate_bubble_reduction() -> None:
    """Show how bubble ratio decreases with more microbatches."""
    print("\n" + "=" * 70)
    print(" BUBBLE RATIO vs MICROBATCHES")
    print("=" * 70)
    print("""
Bubble ratio = (P-1) / (M + P - 1)

Where P = pipeline stages, M = microbatches
""")

    stages = 4
    print(f"For P = {stages} stages:\n")
    print(f"{'Microbatches':<15} {'Bubble Ratio':<15} {'Efficiency':<15}")
    print("-" * 45)

    for mb in [1, 2, 4, 8, 16, 32, 64]:
        bubble = (stages - 1) / (mb + stages - 1)
        efficiency = 1 - bubble
        print(f"{mb:<15} {bubble:<15.1%} {efficiency:<15.1%}")

    print("""
Takeaway: Use at least 4x pipeline stages as microbatches
          for > 80% efficiency.
""")


def explain_memory_tradeoff() -> None:
    """Explain the memory-throughput tradeoff."""
    print("\n" + "=" * 70)
    print(" MEMORY vs THROUGHPUT TRADEOFF")
    print("=" * 70)
    print("""
The fundamental tradeoff in pipeline parallelism:

MORE MICROBATCHES:
  ✓ Lower bubble ratio (better throughput)
  ✗ More activation memory (naive) or same (1F1B)
  ✗ Smaller per-microbatch batch size (worse GPU utilization)

FEWER MICROBATCHES:
  ✗ Higher bubble ratio (worse throughput)
  ✓ Less activation memory
  ✓ Larger per-microbatch batch size (better GPU utilization)

1F1B ADVANTAGE:
  With 1F1B, memory is bounded by pipeline depth, NOT microbatches.
  This allows many microbatches for low bubbles without memory explosion.

Example calculation:
  Model: 24 layers, 4096 hidden dim, batch 512
  Pipeline: 4 stages (6 layers each)
  Microbatches: 16 (32 samples each)

  Naive memory: 16 × activations ≈ 16 × 32 × 4096 × 6 = 12.6 GB per stage (must keep all activations in memory)
  1F1B memory:   4 × activations ≈  4 × 32 × 4096 × 6 =  3.1 GB per stage (only need to keep activations for 4 stages in memory)

  4x memory reduction!
""")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Schedule Visualizer")
    parser.add_argument("--stages", "-s", type=int, default=4,
                        help="Number of pipeline stages")
    parser.add_argument("--microbatches", "-m", type=int, default=8,
                        help="Number of microbatches")
    args = parser.parse_args()

    print("╔" + "═" * 68 + "╗")
    print("║" + " PIPELINE PARALLELISM SCHEDULE VISUALIZER".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    analyze_schedules(args.stages, args.microbatches)
    demonstrate_bubble_reduction()
    explain_memory_tradeoff()


if __name__ == "__main__":
    main()
