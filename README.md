# Canny-Edge-RISCV
Embedded Project
## Phase 4: Optimization Sweep Results

The following table summarizes the impact of compiler optimization flags on binary size, runtime, and vectorization behavior for the Canny Edge pipeline:

| Optimization | Binary Size | Runtime (100 iters) | Vectorization Notes |
|--------------|-------------|--------------------------|---------------------|
| -O0          | 3.0M        | 4.26145s                 | None                |
| -O2          | 3.0M        | 1.75745s                 | Some loops          |
| -O3          | 3.0M        | 0.588669s                | More aggressive     |
| -Os          | 3.0M        | 1.70915s                 | Size-focused        |
| -Ofast       | 3.0M        | 0.586594s                | Max speed           |

**Observations:**
- Binary sizes stayed roughly constant (~3 MB).
- Runtime dropped significantly with higher optimization levels.
- Vectorization was minimal at `-O0`, moderate at `-O2`, aggressive at `-O3`, and maximized at `-Ofast`.
- `-Os` prioritized smaller code size but sacrificed some speed.

