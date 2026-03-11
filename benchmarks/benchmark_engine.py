import numpy as np
from numba import njit
import time
import bottleneck as bn

np.random.seed(36)

def old_backtest_loop(close, entries, exits, fees, slippage, size_usd, init_cash):
    cash = init_cash
    pos_size = 0.0
    hold_pos = False
    equity = init_cash
    trades = []
    equity_list = []
    cash_list = []
    asset_list = []
    
    for i in range(len(close)):
        # mark-to-market equity first
        equity = cash + (pos_size * close[i])

        if hold_pos and exits[i]:
            exit_value = pos_size * (close[i] * (1 - slippage))
            fee_paid = exit_value * fees
            cash += exit_value - fee_paid
            
            realized_pnl = (exit_value - fee_paid) - (size_usd * (1 + fees))
            trades.append((
                i, 'exit', close[i], pos_size, size_usd,
                cash, equity, realized_pnl
            ))

            pos_size = 0.0
            hold_pos = False
            equity = cash

        elif (not hold_pos) and entries[i] and cash >= size_usd * (1 + fees):
            entry_cost = size_usd * (1 + fees)
            cash -= entry_cost
            pos_size = size_usd / (close[i] * (1 + slippage))
            hold_pos = True
            equity = cash + (pos_size * close[i])

            trades.append((
                i, 'entry', close[i], pos_size, size_usd,
                cash, equity, 0.0
            ))

        equity_list.append(equity)
        cash_list.append(cash)
        asset_list.append(pos_size * close[i])
        
    return equity_list, cash_list, asset_list, trades


@njit
def new_backtest_loop(close, entries, exits, fees, slippage, size_usd, init_cash):
    cash = init_cash
    pos_size = 0.0
    hold_pos = False
    equity = init_cash
    equity_list = np.empty(len(close), dtype=float)
    cash_list = np.empty(len(close), dtype=float)
    asset_list = np.empty(len(close), dtype=float)
    trades = np.empty((len(close), 8), dtype=float)
    trade_count = 0
    for i in range(len(close)):
        # mark-to-market equity first
        equity = cash + (pos_size * close[i])

        if hold_pos and exits[i]:
            exit_value = pos_size * (close[i] * (1 - slippage))
            fee_paid = exit_value * fees
            cash += exit_value - fee_paid

            realized_pnl = (exit_value - fee_paid) - (size_usd * (1 + fees))
            trades[trade_count,0] = float(i)
            trades[trade_count,1] = -1.0
            trades[trade_count,2] = close[i]
            trades[trade_count,3] = pos_size
            trades[trade_count,4] = size_usd
            trades[trade_count,5] = cash
            trades[trade_count,6] = equity
            trades[trade_count,7] = realized_pnl

            trade_count += 1

            pos_size = 0.0
            hold_pos = False
            equity = cash

        elif (not hold_pos) and entries[i] and cash >= size_usd * (1 + fees):
            entry_cost = size_usd * (1 + fees)
            cash -= entry_cost
            pos_size = size_usd / (close[i] * (1 + slippage))
            hold_pos = True
            equity = cash + (pos_size * close[i])

            trades[trade_count,0] = float(i)
            trades[trade_count,1] = 1.0
            trades[trade_count,2] = close[i]
            trades[trade_count,3] = pos_size
            trades[trade_count,4] = size_usd
            trades[trade_count,5] = cash
            trades[trade_count,6] = equity
            trades[trade_count,7] = 0.0
            trade_count += 1

        asset_list[i] = pos_size * close[i]
        equity_list[i] = equity
        cash_list[i] = cash

    return equity_list, cash_list, asset_list, trades[:trade_count]


@njit
def generate_mockup_market(bars: int = 365) -> np.ndarray:
    s0 = 100.0
    mu = 0.1
    sigma = 0.1
    days = bars
    dt = 1/days
    
    daily_returns = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        size =days
    )
    
    return s0 * np.exp(np.cumsum(daily_returns))
    
def generate_close_entries_exits(bars: int = 1_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = generate_mockup_market(bars)
    sma20 = bn.move_mean(close, 20)
    sma50 = bn.move_mean(close, 50)
    
    condition_ = sma20 > sma50
    
    entries = condition_ & np.roll(~condition_, 1)
    entries[0] = False
    exits = ~condition_ & np.roll(condition_, 1)
    exits[0] = False
    return close, entries, exits


#Warm-up for njit
close, entries, exits = generate_close_entries_exits(365)
no_warm_up = time.perf_counter()
new_backtest_loop(close, entries, exits, 0.01, 0.01, 1_000.0, 10_000.0) # warm-up
end_no_warm_up = time.perf_counter()

print(f'Data points = {len(close):,d}')
start = time.perf_counter()
old_backtest_loop(close, entries, exits, 0.01, 0.01, 1_000.0, 10_000.0)
end = time.perf_counter()
print(f"old_backtest_loop took {(end - start):5f} seconds")

start = time.perf_counter()
new_backtest_loop(close, entries, exits, 0.01, 0.01, 1_000.0, 10_000.0)
end = time.perf_counter()
print(f"new_backtest_loop + compile time took {(end_no_warm_up - no_warm_up):5f} seconds")
print(f"new_backtest_loop only took {(end - start):5f} seconds")

data_point_list = [10_000, 100_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000]
total_times = 6
for data in data_point_list:
    close, entries, exits = generate_close_entries_exits(data)
    print(f'Data points = {len(close):,d}')
    start = time.perf_counter()
    for i in range(total_times):
        _, _, _, _ = old_backtest_loop(close, entries, exits, 0.01, 0.01, 1_000.0, 10_000.0)
    end = time.perf_counter()
    print(f"old_backtest_loop took {(end - start)/total_times:5f} seconds")
    
    start = time.perf_counter()
    for i in range(total_times):
        _, _, _, _ = new_backtest_loop(close, entries, exits, 0.01, 0.01, 1_000.0, 10_000.0)
    end = time.perf_counter()
    print(f"new_backtest_loop took {(end - start)/total_times:5f} seconds")
    
data_point_list = [100_000_000,300_000_000]
total_times = 1
for data in data_point_list:
    close, entries, exits = generate_close_entries_exits(data)
    print(f'Data points = {len(close):,d}')
    start = time.perf_counter()
    for i in range(total_times):
        _, _, _, _ = old_backtest_loop(close, entries, exits, 0.01, 0.01, 1_000.0, 10_000.0)
    end = time.perf_counter()
    print(f"old_backtest_loop took {(end - start)/total_times:5f} seconds")
    
    start = time.perf_counter()
    for i in range(total_times):
        _, _, _, _ = new_backtest_loop(close, entries, exits, 0.01, 0.01, 1_000.0, 10_000.0)
    end = time.perf_counter()
    print(f"new_backtest_loop took {(end - start)/total_times:5f} seconds")