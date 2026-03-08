# simple-backtester

A minimal, long-only backtesting engine built with NumPy, Polars and Bottleneck. 
No paid dependencies. No black boxes. You pass in signals, you get results.

## Why

Most backtesting libraries do too much. This one does the minimum:
take entry/exit signals, simulate trades with fees and slippage, 
return equity curve and basic metrics.

## Install

```bash
pip install numpy polars bottleneck
```

Clone or copy backtester.py into your project.

```python
import numpy as np
from backtester import SimpleBacktester

close   = np.array([100, 102, 105, 103, 108, 110])
entries = np.array([True, False, False, False, True, False])
exits   = np.array([False, False, True, False, False, True])

bt = SimpleBacktester(
    close=close,
    entries=entries,
    exits=exits,
    fees=0.001,
    slippage=0.0005,
    size_usd=1000,
    init_cash=10000,
)

print(bt.summary())
```

Output:
```text
col_names	column_0
str	f64
"Start"	10000.0
"End"	10443.68
"win_rate[%]"	33.33
"total_return[%]"	4.44
"max_drawdown[%]"	-5.44
"sharpe_per_period"	0.037
"profit_factor"	2.3814
"closed_trades"	6.0
```

## What You Get
- bt.equity_array — full equity curve as numpy array
- bt.trades — all entries/exits as Polars DataFrame
- bt.summary() — dict with core metrics

## Metrics
|Metric|	Description|
|---|---|
|Total Return %	|End equity vs starting cash|
|Max Drawdown %	|Largest peak-to-trough decline|
|Sharpe Ratio	|Mean return / std of returns (per period)|
|Win Rate %	|Winning trades / total closed trades|
|Profit Factor	|Gross profit / gross loss|

## Limitations
- Long-only — no short selling
- Close prices only — no high/low/open
- Fixed position size in USD — no percentage-based sizing
- Integer index — no datetime handling, map to your own dates
- No benchmark comparison
- Sharpe is per-period, not annualized — multiply by √(periods_per_year) yourself

## Tests
```bash
pip install pytest
pytest test_backtester.py -v
```