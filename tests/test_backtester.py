from backtester import SimpleBacktester
import numpy as np
import pytest


def test_round_trip_same_price():
    """
    Buy and sell at same prices.
    We should lose exactly fees + slippage, nothing else.
    """
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    entries = np.array([False, True, False, False, False])
    exits = np.array([False, False, True, False, False])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.001,
        slippage=0.001,
        size_usd=1000,
        init_cash=10000,
    )

    # Do the math by hand FIRST, then put your answer here
    # Entry (bar 1):
    #   cash -= 1000 * 1.001 = 1001.0
    #   cash = 10000 - 1001 = 8999.0
    #   pos_size = 1000 / (100 * 1.001) = 9.99000999...
    #
    # Exit (bar 3):
    #   realize = 9.99000999 * (100 * 0.999) = 998.002998...
    #   cash += 998.002998 - (998.002998 * 0.001) = 998.002998 - 0.998002998
    #   cash += 997.004995
    #   cash = 8999.0 + 997.004995 = 9996.004995
    #
    # Round trip cost = 10000 - 9996.005 ≈ 3.995

    expected_final_cash = 9996.004995

    assert bt.cash == pytest.approx(expected_final_cash, abs=0.01)
    assert not bt.hold_pos , "Should not be holding after exit"
    assert bt.pos_size == 0, "Position size should be 0 after exit"
    assert len(bt.trades) == 2, f"Should have 2 trades, got {len(bt.trades)}"

    print("PASSED: round trip same price")

def test_equity_curve_length():
    """Equity curve should have same length as close"""
    close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    entries = np.array([False, True, False, False, False])
    exits = np.array([False, False, True, False, False])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.001,
        slippage=0.001,
        size_usd=1000,
        init_cash=10000,
    )

    assert len(bt.equity_list) == len(close), "Equity curve should have same length as close"

    print("PASSED: equity curve length")

def test_not_enough_cash():
    """Entry signal but not enough cash. Should skip"""
    close = np.array([100.0, 100.0, 100.0])
    entries = np.array([False, True, False])
    exits = np.array([False, False, False])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.001,
        slippage=0.001,
        size_usd=1000.0,
        init_cash=500.0,  # not enough
    )

    assert bt.cash == 500.0
    assert not bt.hold_pos
    assert len(bt.trades) == 0

    print("PASSED: not enough cash")

def test_profitable_trade():
    """Should have correct profit"""
    close = np.array([100,105,110,115,120])
    entries = np.array([True, False, False, False, False])
    exits = np.array([False, False, False, False, True])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.0,
        slippage=0.0,
        size_usd=1000,
        init_cash=10000,
    )

    # buy 1000 usd of 100 usd stock price = 10 shares
    # sell 10 shares we should expect 10*120 = 1200 usd
    # profit = 200 usd
    # final cash = 10000 + 200 = 10200
    assert bt.cash == 10200, "Should have correct profit"
    assert not bt.hold_pos, "Should not be holding after exit"
    assert len(bt.trades) == 2, f"Should have 2 trades, got {len(bt.trades)}"

    print("PASSED: profitable trade")

def test_losing_trade():
    """ Should have correct loss"""
    close = np.array([100,95,90,85,80])
    entries = np.array([True, False, False, False, False])
    exits = np.array([False, False, False, False, True])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.0,
        slippage=0.0,
        size_usd=1000,
        init_cash=10000,
    )

    # buy 1000usd of 100 usd stock price = 10 shares
    # sell 10 shares of 80 usd = 800 usd
    # equity should be 9000 + 800 = 9800
    assert bt.cash == 9800, "Should have correct loss"
    assert not bt.hold_pos, "Should not be holding after exit"
    assert len(bt.trades) == 2, f"Should have 2 trades, got {len(bt.trades)}"

    print("PASSED: losing trade")

def test_entry_exit_same_bar():
    """Entry and exit on same bar. If doesn't hold position it should entry, if hold position it should exit"""
    close = np.array([100,105,110,115,120,125,130])
    entries = np.array([True, False, True, True, False, True,False])
    exits = np.array([True, False, True, True, False, True, False])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.0,
        slippage=0.0,
        size_usd=1000,
        init_cash=10000,
    )


    assert not bt.hold_pos, "Should not be holding after exit"
    assert len(bt.trades) == 4, f"Should have 4 trades, got {len(bt.trades)}"

    print("PASSED: entry and exit on same bar")
    
def test_multiple_entries_one_exit():
    """Multiple entries and one exit"""
    close = np.array([100,105,110,115,120,125,130])
    entries = np.array([True, False, True, True, False, True,False])
    exits = np.array([False, False, False, False, False, False, True])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.0,
        slippage=0.0,
        size_usd=1000,
        init_cash=10000,
    )

    assert not bt.hold_pos, "Should not be holding after exit"
    assert len(bt.trades) == 2, f"Should have 2 trades, got {len(bt.trades)}"

    print("PASSED: multiple entries one exit")

def test_multiple_trades():
    """Multiple trades and profit should correct"""
    close = np.array([100,105,110,115,120,125,130])
    entries = np.array([True, False, True, False, False, True,False])
    exits = np.array([False, True, False, True, False, False, True])

    bt = SimpleBacktester(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.0,
        slippage=0.0,
        size_usd=1000,
        init_cash=10000,
    )

    # First round equity 105 usd* 10 share = 9000 + 1050 = 10050
    # Second round equity 115 usd* 9.0909090909 share = (10050 - 1000) + 1,045.4545454535 = 10,095.4545454535
    # Thrid round equity 130 usd* 8 share = (10095.4545454535 - 1000) + 1,040 = 10,135.4545454535
    assert bt.cash == pytest.approx(10_135.4545454535, abs=0.01)
    assert not bt.hold_pos, "Should not be holding after exit"
    assert len(bt.trades) == 6, f"Should have 6 trades, got {len(bt.trades)}"

    print("PASSED: multiple trades")
