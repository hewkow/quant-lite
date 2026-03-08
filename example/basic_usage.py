import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import yfinance as yf

# Add the parent directory (quant-lite) to Python's module search path
sys.path.append(str(Path(__file__).parent.parent))

from backtester import SimpleBacktester


aapl = yf.Ticker("AAPL")
df = aapl.history(start='2024-03-07', end='2026-03-06')

df['sma20'] = df['Close'].rolling(20).mean()
df['sma50'] = df['Close'].rolling(50).mean()

condition = df['sma20'] > df['sma50']
df['entries'] = condition & (~condition.shift(1, fill_value=False))
df['exits'] = (~condition) & (condition.shift(1, fill_value=False))

bt = SimpleBacktester(
    close=df['Close'].to_numpy(),
    entries=df['entries'].to_numpy(),
    exits=df['exits'].to_numpy(),
    fees=0.001,
    slippage=0.0005,
    size_usd=2_000,
    init_cash=10_000,
)

print(bt.summary())
print(bt.trades)
print(bt.equity_list[-3:])

entry_points = df[df['entries']]
exit_points = df[df['exits']]

fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df.index, y=df['sma20'], mode='lines', name='SMA20'))
fig.add_trace(go.Scatter(x=df.index, y=df['sma50'], mode='lines', name='SMA50'))

fig.add_trace(go.Scatter(
    x=entry_points.index, y=entry_points['Close'], mode='markers', name='Entries',
    marker=dict(symbol='triangle-up', size=10, color='limegreen')
))

fig.add_trace(go.Scatter(
    x=exit_points.index, y=exit_points['Close'], mode='markers', name='Exits',
    marker=dict(symbol='triangle-down', size=10, color='red')
))

fig.update_layout(
    title='SMA20 / SMA50 Crossover',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_white',
    height=550,
)

fig.show()

