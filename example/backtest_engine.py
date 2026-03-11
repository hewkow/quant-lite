import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    import numpy as np
    import bottleneck as bn
    from numba import njit



    @njit
    def run_backtest_loop(close, entries, exits, fees, slippage, size_usd, init_cash):
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


    class SimpleBacktester:
        def __init__(
            self,
            close: np.ndarray,
            entries: np.ndarray,
            exits: np.ndarray,
            fees: float = 0.001,
            slippage: float = 0.0005,
            size_usd: float = 2_000,
            init_cash: float = 10_000,
        ):
            if len(close) != len(entries) or len(close) != len(exits):
                raise ValueError("close, entries, exits must have same length")

            self.close = np.asarray(close, dtype=float)
            self.entries = np.asarray(entries, dtype=bool)
            self.exits = np.asarray(exits, dtype=bool)
            self.fees = float(fees)
            self.slippage = float(slippage)
            self.size_usd = float(size_usd)
            self.init_cash = float(init_cash)

            self.pos_size = 0.0
            self.hold_pos = False
            self.equity = self.init_cash
            self.equity_list = []
            self.trades = []
            self.trades_index = [
                "index",
                "type",
                "price",
                "pos_size",
                "size_usd",
                "cash",
                "equity",
                "realized_pnl",
            ]

            self.equity_list, self.cash_list, self.asset_list, self.trades = run_backtest_loop(
                self.close,
                self.entries,
                self.exits,
                self.fees,
                self.slippage,
                self.size_usd,
                self.init_cash,
            )
            self._post_process()

        def _post_process(self):
            self.trades = pl.DataFrame(self.trades, schema=self.trades_index, orient="row")
            self.trades = self.trades.with_columns(
                pl.when(pl.col('type') == 1.0)
                .then(pl.lit('entry'))
                .otherwise(pl.lit('exit'))
                .alias('type')
            )
            exits_df = self.trades.filter(pl.col("type") == "exit")
            win = exits_df.filter(pl.col("realized_pnl") > 0).height
            lose = exits_df.filter(pl.col("realized_pnl") < 0).height
            total_closed = win + lose
            self.win_rate = (win / total_closed * 100) if total_closed > 0 else np.nan

            self.total_return = (self.equity_list[-1] / self.init_cash - 1) * 100

            rolling_high = bn.move_max(
                self.equity_list, window=len(self.equity_list), min_count=1
            )
            drawdowns = (self.equity_list / rolling_high - 1) * 100
            self.max_drawdown = float(drawdowns.min())

            if len(self.equity_list) >= 2:
                self.returns = self.equity_list[1:] / self.equity_list[:-1] - 1
            else:
                self.returns = np.array([], dtype=float)

            if len(self.returns) >= 2 and np.std(self.returns, ddof=1) > 0:
                self.sharpe_ratio_per_period = float(
                    np.mean(self.returns) / np.std(self.returns, ddof=1)
                )
            else:
                self.sharpe_ratio_per_period = np.nan

            gross_profit = (
                exits_df.filter(pl.col("realized_pnl") > 0)
                .select(pl.col("realized_pnl").sum())
                .item()
                or 0.0
            )
            gross_loss = abs(
                exits_df.filter(pl.col("realized_pnl") < 0)
                .select(pl.col("realized_pnl").sum())
                .item()
                or 0.0
            )

            if gross_loss > 0:
                self.profit_factor = float(gross_profit / gross_loss)
            elif gross_profit > 0 and gross_loss == 0:
                self.profit_factor = np.inf
            else:
                self.profit_factor = np.nan

        def summary(self):
            return (
                pl.DataFrame(
                    {
                        "Start": self.init_cash,
                        "End": round(self.equity_list[-1], 2),
                        "win_rate[%]": None
                        if np.isnan(self.win_rate)
                        else round(self.win_rate, 2),
                        "total_return[%]": round(self.total_return, 2),
                        "max_drawdown[%]": round(self.max_drawdown, 2),
                        "sharpe_per_period": None
                        if np.isnan(self.sharpe_ratio_per_period)
                        else round(self.sharpe_ratio_per_period, 4),
                        "profit_factor": None
                        if np.isnan(self.profit_factor)
                        else round(self.profit_factor, 4),
                        "closed_trades": int(
                            self.trades.filter(pl.col("type") == "exit").height
                        ),
                    }
                )
                .transpose(include_header=True, header_name="col_names")
                .rename({"col_names": "Metrics", "column_0": "Value"})
            )

    return SimpleBacktester, np


@app.cell
def _(SimpleBacktester):
    import yfinance as yf
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
    return (bt,)


@app.cell
def _(bt):
    import plotly.graph_objects as go
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(y=bt.equity_list,mode='lines',name='Equity'))
    fig_eq.add_trace(go.Scatter(y=bt.cash_list,mode='lines',name='Hold cash'))
    fig_eq.add_trace(go.Scatter(y=bt.asset_list,mode='lines',name='Hold asset'))
    fig_eq.show()
    return (go,)


@app.cell
def _(go, np):
    s0 = 100.0
    mu = 0.1
    sigma = 0.1
    days = 252
    dt = 1 / days
    daily_returns = np.random.normal((mu - 0.5 * sigma ** 2) * dt, sigma * np.sqrt(dt), size=days)
    price_path = s0 * np.exp(np.cumsum(daily_returns))
    fig_eq_1 = go.Figure()
    fig_eq_1.add_trace(go.Scatter(y=price_path, mode='lines', name='Close'))
    fig_eq_1.show()
    return


if __name__ == "__main__":
    app.run()
