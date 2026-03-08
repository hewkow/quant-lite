import polars as pl
import numpy as np
import bottleneck as bn


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
            raise ValueError('close, entries, exits must have same length')

        self.close = np.asarray(close, dtype=float)
        self.entries = np.asarray(entries, dtype=bool)
        self.exits = np.asarray(exits, dtype=bool)
        self.fees = float(fees)
        self.slippage = float(slippage)
        self.size_usd = float(size_usd)
        self.init_cash = float(init_cash)

        self.pos_size = 0.0
        self.cash = self.init_cash
        self.hold_pos = False
        self.equity = self.init_cash
        self.equity_list = []
        self.trades = []
        self.trades_index = [
            'index', 'type', 'price', 'pos_size', 'size_usd', 'cash', 'equity', 'realized_pnl'
        ]

        self.run()
        self._post_process()

    def run(self):
        for i in range(len(self.close)):
            # mark-to-market equity first
            self.equity = self.cash + (self.pos_size * self.close[i])

            if self.hold_pos and self.exits[i]:
                exit_value = self.pos_size * (self.close[i] * (1 - self.slippage))
                fee_paid = exit_value * self.fees
                self.cash += exit_value - fee_paid

                realized_pnl = (exit_value - fee_paid) - (self.size_usd * (1 + self.fees))
                self.trades.append((
                    i, 'exit', self.close[i], self.pos_size, self.size_usd,
                    self.cash, self.equity, realized_pnl
                ))

                self.pos_size = 0.0
                self.hold_pos = False
                self.equity = self.cash

            elif (not self.hold_pos) and self.entries[i] and self.cash >= self.size_usd * (1 + self.fees):
                entry_cost = self.size_usd * (1 + self.fees)
                self.cash -= entry_cost
                self.pos_size = self.size_usd / (self.close[i] * (1 + self.slippage))
                self.hold_pos = True
                self.equity = self.cash + (self.pos_size * self.close[i])

                self.trades.append((
                    i, 'entry', self.close[i], self.pos_size, self.size_usd,
                    self.cash, self.equity, 0.0
                ))

            self.equity_list.append(self.equity)

    def _post_process(self):
        self.equity_array = np.asarray(self.equity_list, dtype=float)
        self.trades = pl.DataFrame(self.trades, schema=self.trades_index, orient='row')

        exits_df = self.trades.filter(pl.col('type') == 'exit')
        win = exits_df.filter(pl.col('realized_pnl') > 0).height
        lose = exits_df.filter(pl.col('realized_pnl') < 0).height
        total_closed = win + lose
        self.win_rate = (win / total_closed * 100) if total_closed > 0 else np.nan

        self.total_return = (self.equity_array[-1] / self.init_cash - 1) * 100

        rolling_high = bn.move_max(self.equity_array, window=len(self.equity_array), min_count=1)
        drawdowns = (self.equity_array / rolling_high - 1) * 100
        self.max_drawdown = float(drawdowns.min())

        if len(self.equity_array) >= 2:
            self.returns = self.equity_array[1:] / self.equity_array[:-1] - 1
        else:
            self.returns = np.array([], dtype=float)

        if len(self.returns) >= 2 and np.std(self.returns, ddof=1) > 0:
            self.sharpe_ratio_per_period = float(np.mean(self.returns) / np.std(self.returns, ddof=1))
        else:
            self.sharpe_ratio_per_period = np.nan

        gross_profit = exits_df.filter(pl.col('realized_pnl') > 0).select(pl.col('realized_pnl').sum()).item() or 0.0
        gross_loss = abs(exits_df.filter(pl.col('realized_pnl') < 0).select(pl.col('realized_pnl').sum()).item() or 0.0)

        if gross_loss > 0:
            self.profit_factor = float(gross_profit / gross_loss)
        elif gross_profit > 0 and gross_loss == 0:
            self.profit_factor = np.inf
        else:
            self.profit_factor = np.nan

    def summary(self):
        return pl.DataFrame({
            'Start' : self.init_cash,
            'End' : round(self.equity_array[-1], 2),
            'win_rate[%]': None if np.isnan(self.win_rate) else round(self.win_rate, 2),
            'total_return[%]': round(self.total_return, 2) ,
            'max_drawdown[%]': round(self.max_drawdown, 2) ,
            'sharpe_per_period': None if np.isnan(self.sharpe_ratio_per_period) else round(self.sharpe_ratio_per_period, 4),
            'profit_factor': None if np.isnan(self.profit_factor) else round(self.profit_factor, 4),
            'closed_trades': int(self.trades.filter(pl.col('type') == 'exit').height),
        }).transpose(include_header=True, header_name="col_names").rename({"col_names": "Metrics", "column_0": "Value"})

