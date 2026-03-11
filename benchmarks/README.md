# python-loop to numba-loop
## Purpose
- To compare the performance of the original Python backtest loop against the Numba-compiled loop.

## What was compared
- `old_backtest_loop`
- `new_backtest_loop`

## Benchmarks rules
- Identical precomputed inputs were used for both: `close`, `entries`, `exits`
- Numba warmed up before timing
- Reports result are the average of 7 runs
- This benchmark measures engine-only, not full pipeline

## Hardware / environment
- 13th Gen Intel(R) Core(TM) i5-13600KF (3.50 GHz)
- ram ddr4 32GB
- python 3.13
- numpy 2.4.2
- numba 0.64.0

## Results table
| Bars | old_backtest_loop (s) | new_backtest_loop (s) | Speedup |
|-----------|-------------------|-------------------|---------|
| 365   |       0.000153             |     0.000015       |   10x    |
| 10,000 |          0.003208          |       0.000030            |   106x      |
|100,000 |        0.022099           |        0.000338           |     65x    |
| 1,000,000 |        0.258648           |       0.005824            |    44x     |
| 3,000,000 |        0.778702           |        0.017139           |    45x     |
| 10,000,000 |      2.794590             |         0.051294          |    54x     |
| 30,000,000 |       8.256668            |       0.151360            |    54x     |

## Large-scale single-run results
Due to runtime and memory cost, the following large-scale benchmarks were run once only.
These numbers should be treated as approximate observations rather than stable averages.
| Bars | old_backtest_loop (s) | new_backtest_loop (s) | Speedup |
|-----------|-------------------|-------------------|---------|
| 100,000,000   |       26.968113             |     0.501939       |   53x    |
| 300,000,000 |          127.671255          |    8.969570 |   14x |

## Notes / caveats:
- first call of numba takes longer due to JIT complie cost
- trade array is still over-allocated in numba version

## trade-off
- performance gain but memory overhead is still present

## Conclusion
The Numba-based loop delivers a significant speedup over the original Python loop.

## Correctness validation
All test passed before perfomance benchmarks were recorded.