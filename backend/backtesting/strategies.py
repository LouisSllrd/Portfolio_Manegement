import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .backtest import BaseStrategy  

# ================= STRATEGY 1: BarUpDn ===================
class BarUpDnStrategy(BaseStrategy):
    """
    - Green bar + Open > prev Close => long
    - Red bar + Open < prev Close => short
    """
    PARAMS_SCHEMA = []
    def generate_signal(self, i, df):
        if i == 0:
            return None

        row = df.iloc[i]
        prev = df.iloc[i-1]

        is_green = row["Close"] > row["Open"]
        is_red = row["Close"] < row["Open"]

        if is_green and row["Open"] > prev["Close"]:
            return "long"
        if is_red and row["Open"] < prev["Close"]:
            return "short"
        return None

    def plot(self, df, trades_df):
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Close"], label="Close Price", alpha=0.6)

        # Trades
        longs = trades_df[trades_df["side"] == "long"]
        shorts = trades_df[trades_df["side"] == "short"]
        plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", label="Long Entry")
        plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", label="Short Entry")

        plt.title("BarUpDn Strategy - Trade points")
        plt.legend()
        plt.grid(True)
        plt.show()

# ================= STRATEGY 2: Bollinger's Bands Strategy ===================
class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion strategy:
      - Buy when price closes below lower band
      - Sell (short) when price closes above upper band
    """
    PARAMS_SCHEMA = [
        {"name": "window", "label": "Window (days)", "type": "number", "default": 20},
        {"name": "n_std", "label": "Std Factor", "type": "number", "default": 2},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_signal(self, i, df):
        if i < self.window:
            return None  # Not enough data for bands

        # Compute rolling mean and std
        rolling_mean = df["Close"].iloc[i-self.window+1:i+1].mean()
        rolling_std = df["Close"].iloc[i-self.window+1:i+1].std()

        upper_band = rolling_mean + self.n_std * rolling_std
        lower_band = rolling_mean - self.n_std * rolling_std

        close_price = df["Close"].iloc[i]

        if close_price < lower_band:
            return "long"
        elif close_price > upper_band:
            return "short"
        else:
            return None

    def plot(self, df, trades_df):
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Close"], label="Close Price", alpha=0.6)

        # Compute bands for plotting
        rolling_mean = df["Close"].rolling(window=self.window).mean()
        rolling_std = df["Close"].rolling(window=self.window).std()
        upper_band = rolling_mean + self.n_std * rolling_std
        lower_band = rolling_mean - self.n_std * rolling_std

        plt.plot(df["Date"], upper_band, label="Upper Band", color="red", linestyle="--")
        plt.plot(df["Date"], lower_band, label="Lower Band", color="green", linestyle="--")
        plt.plot(df["Date"], rolling_mean, label="SMA", color="blue", linestyle="-")

        # Plot trades
        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", label="Long Entry")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", label="Short Entry")

        plt.title("Bollinger Bands Strategy")
        plt.legend()
        plt.grid(True)
        plt.show()

        # ================= STRATEGY: Bollinger Bands Directed ===================
class BollingerDirectedStrategy(BaseStrategy):
    """
    Bollinger Bands Directed Strategy:
      - Buy when price closes below lower band (long)
      - Sell when price closes above upper band (short)
      - Can restrict to long-only, short-only, or both
    """

    PARAMS_SCHEMA = [
        {"name": "window", "label": "Window (days)", "type": "number", "default": 20},
        {"name": "std_factor", "label": "Std Factor", "type": "number", "default": 2},
        {"name": "direction", "label": "Direction (-1=short, 0=both, 1=long)", "type": "number", "default": 0},
    ]

    def __init__(self, window=20, std_factor=2, direction=0, **kwargs):
        self.window = window
        self.std_factor = std_factor
        self.direction = direction

    def generate_signal(self, i, df):
        if i < self.window:
            return None  # Not enough data

        rolling_mean = df["Close"].iloc[i-self.window+1:i+1].mean()
        rolling_std = df["Close"].iloc[i-self.window+1:i+1].std()
        upper_band = rolling_mean + self.std_factor * rolling_std
        lower_band = rolling_mean - self.std_factor * rolling_std
        close_price = df["Close"].iloc[i]

        if close_price < lower_band and self.direction in [0, 1]:
            return "long"
        elif close_price > upper_band and self.direction in [0, -1]:
            return "short"
        return None

    def plot(self, df, trades_df):
        df = df.copy()
        df["MB"] = df["Close"].rolling(self.window).mean()
        df["STD"] = df["Close"].rolling(self.window).std()
        df["UB"] = df["MB"] + self.std_factor * df["STD"]
        df["LB"] = df["MB"] - self.std_factor * df["STD"]

        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")
        plt.plot(df["Date"], df["UB"], label="Upper Band", linestyle="--", color="red")
        plt.plot(df["Date"], df["MB"], label="Middle Band", linestyle="-.", color="blue")
        plt.plot(df["Date"], df["LB"], label="Lower Band", linestyle="--", color="green")

        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", label="Long Entry", s=50)
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", label="Short Entry", s=50)

        plt.title("Bollinger Directed Strategy - Price & Trades")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()

# ================= STRATEGY 4: Channel BreakOut ===================
class ChannelBreakOutStrategy(BaseStrategy):
    """
    Channel BreakOut Strategy:
      - Buy when price breaks above highest high of last N bars
      - Sell when price breaks below lowest low of last N bars
    """
    PARAMS_SCHEMA = [
        {"name": "length", "label": "Channel Length (bars)", "type": "number", "default": 5},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.length = kwargs.get("length", 5)

    def generate_signal(self, i, df):
        if i < self.length:
            return None

        prev_window = df.iloc[i-self.length:i]
        upper = prev_window["High"].max()
        lower = prev_window["Low"].min()

        row = df.iloc[i]

        if row["High"] > upper:
            return "long"
        elif row["Low"] < lower:
            return "short"
        else:
            return None

    def plot(self, df, trades_df):
        df = df.copy()
        df["Upper"] = df["High"].shift(1).rolling(self.length).max()
        df["Lower"] = df["Low"].shift(1).rolling(self.length).min()

        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")
        plt.plot(df["Date"], df["Upper"], label="Upper Channel", linestyle="--")
        plt.plot(df["Date"], df["Lower"], label="Lower Channel", linestyle="--")

        # Plot trades
        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Buy Signal")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Sell Signal")

        plt.title(f"Channel BreakOut Strategy: Price & Signals (Length={self.length})")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ================= STRATEGY 5: Consecutive Up/Down ===================
class ConsecutiveUpDownStrategy(BaseStrategy):
    """
    Consecutive Up/Down Strategy:
      - Buy when there are >= N consecutive up bars
      - Sell when there are >= M consecutive down bars
    """
    PARAMS_SCHEMA = [
        {"name": "consecutive_up", "label": "Consecutive Up Bars", "type": "number", "default": 3},
        {"name": "consecutive_down", "label": "Consecutive Down Bars", "type": "number", "default": 3},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.consecutive_up = kwargs.get("consecutive_up", 3)
        self.consecutive_down = kwargs.get("consecutive_down", 3)

    def generate_signal(self, i, df):
        if i == 0:
            self.up_count = 0
            self.down_count = 0
            return None

        # Count consecutive bars
        if df["Close"].iloc[i] > df["Close"].iloc[i-1]:
            self.up_count = getattr(self, "up_count", 0) + 1
            self.down_count = 0
        elif df["Close"].iloc[i] < df["Close"].iloc[i-1]:
            self.down_count = getattr(self, "down_count", 0) + 1
            self.up_count = 0
        else:
            self.up_count = self.down_count = 0

        if self.up_count >= self.consecutive_up:
            return "long"
        elif self.down_count >= self.consecutive_down:
            return "short"
        else:
            return None

    def plot(self, df, trades_df):
        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")

        # Plot trades
        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Signal")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Signal")

        plt.title(f"Consecutive Up/Down Strategy: Price & Signals (Up={self.consecutive_up}, Down={self.consecutive_down})")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

# ================= STRATEGY 5: Greedy Strategy ===================
class GreedyStrategy(BaseStrategy):
    """
    Greedy Strategy:
      - Enter on gaps up/down
      - Pyramiding allowed up to max_intraday_orders
      - Take Profit / Stop Loss in ticks
    """
    PARAMS_SCHEMA = [
        {"name": "tp_ticks", "label": "Take Profit (ticks)", "type": "number", "default": 10},
        {"name": "sl_ticks", "label": "Stop Loss (ticks)", "type": "number", "default": 10},
        {"name": "max_intraday_orders", "label": "Max Intraday Orders", "type": "number", "default": 5},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tp_ticks = kwargs.get("tp_ticks", 10)
        self.sl_ticks = kwargs.get("sl_ticks", 10)
        self.max_intraday_orders = kwargs.get("max_intraday_orders", 5)
        self.intraday_orders = 0  # compteur de pyramiding

    def generate_signal(self, i, df):
        """Retourne 'long', 'short' ou None. Ne modifie pas la position directement."""
        if i == 0:
            self.intraday_orders = 0
            return None

        prev = df.iloc[i-1]
        row = df.iloc[i]
        next_row = df.iloc[i+1] if i+1 < len(df) else row

        up_gap = row["Open"] > prev["High"]
        down_gap = row["Open"] < prev["Low"]
        green_candle = next_row["Close"] > next_row["Open"]
        red_candle = next_row["Close"] < next_row["Open"]

        # Signal d'entrée
        enter_long = up_gap
        enter_short = down_gap

        # Signal pyramiding
        if self.intraday_orders >= self.max_intraday_orders:
            return None

        if enter_long:
            self.intraday_orders += 1
            return "long"
        elif enter_short:
            self.intraday_orders += 1
            return "short"

        return None

    def post_trade_update(self, trade_closed: bool):
        """
        Reset intraday counter when un trade est fermé (comme dans backtest_greedy)
        """
        if trade_closed:
            self.intraday_orders = 0

    def plot(self, df, trades_df):
        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")
        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Signal")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Signal")
        plt.title("Greedy Strategy: Price & Signals")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class InsideBarStrategy(BaseStrategy):
    """
    Inside Bar Strategy:
      - Detect inside bars
      - Enter long if next candle breaks above high
      - Enter short if next candle breaks below low
    """
    PARAMS_SCHEMA = []  # aucun paramètre configurable pour le moment

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position = None

    def generate_signal(self, i, df):
        if i == 0:
            self.position = None
            return None

        prev = df.iloc[i-1]
        row = df.iloc[i]
        next_row = df.iloc[i+1] if i+1 < len(df) else row

        # Detect inside bar
        inside_bar = row["High"] <= prev["High"] and row["Low"] >= prev["Low"]

        # Entry signals
        enter_long = inside_bar and next_row["Close"] > row["High"]
        enter_short = inside_bar and next_row["Close"] < row["Low"]

        return "long" if enter_long else "short" if enter_short else None

    def plot(self, df, trades_df):
        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")

        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Entry")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Entry")

        plt.title("Inside Bar Strategy: Price & Signals")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class KeltnerChannelsStrategy(BaseStrategy):
    """
    Keltner Channels Strategy:
      - Compute Keltner Channels
      - Long if price crosses above upper band
      - Short if price crosses below lower band
    """
    PARAMS_SCHEMA = [
        {"name": "length", "label": "Moving Average Length", "type": "number", "default": 20},
        {"name": "multiplier", "label": "ATR Multiplier", "type": "number", "default": 2.0},
        {"name": "use_ema", "label": "Use EMA", "type": "boolean", "default": False},
        {"name": "atr_length", "label": "ATR Length", "type": "number", "default": 14},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.length = kwargs.get("length", 20)
        self.multiplier = kwargs.get("multiplier", 2.0)
        self.use_ema = kwargs.get("use_ema", False)
        self.atr_length = kwargs.get("atr_length", 14)

        self._channels_computed = False

    def _compute_channels(self, df):
        if self.use_ema:
            df["Basis"] = df["Close"].ewm(span=self.length, adjust=False).mean()
        else:
            df["Basis"] = df["Close"].rolling(self.length).mean()
        
        df["TR"] = df[["High","Low","Close"]].apply(
            lambda row: max(row["High"]-row["Low"], abs(row["High"]-row["Close"]), abs(row["Low"]-row["Close"])), axis=1
        )
        df["ATR"] = df["TR"].rolling(self.atr_length).mean()
        df["Upper"] = df["Basis"] + self.multiplier*df["ATR"]
        df["Lower"] = df["Basis"] - self.multiplier*df["ATR"]
        self._channels_computed = True
        return df

    def generate_signal(self, i, df):
        if not self._channels_computed:
            df[:] = self._compute_channels(df)

        if i == 0:
            return None

        prev = df.iloc[i-1]
        row = df.iloc[i]

        # Entry signals
        enter_long = (prev["Close"] <= prev["Upper"]) and (row["Close"] > row["Upper"])
        enter_short = (prev["Close"] >= prev["Lower"]) and (row["Close"] < row["Lower"])

        return "long" if enter_long else "short" if enter_short else None

    def plot(self, df, trades_df):
        if not self._channels_computed:
            df[:] = self._compute_channels(df)

        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")
        plt.plot(df["Date"], df["Upper"], linestyle="--", color="red", label="Upper Band")
        plt.plot(df["Date"], df["Basis"], linestyle="-.", color="blue", label="Middle Band")
        plt.plot(df["Date"], df["Lower"], linestyle="--", color="green", label="Lower Band")

        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Entry")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Entry")

        plt.title("Keltner Channels Strategy: Price & Signals")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class MACDStrategy(BaseStrategy):
    """
    MACD Strategy:
      - Buy when MACD histogram crosses from negative to positive
      - Sell when MACD histogram crosses from positive to negative
    """
    PARAMS_SCHEMA = [
        {"name": "fast_len", "label": "Fast EMA Length", "type": "number", "default": 12},
        {"name": "slow_len", "label": "Slow EMA Length", "type": "number", "default": 26},
        {"name": "signal_len", "label": "Signal Line Length", "type": "number", "default": 9},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fast_len = kwargs.get("fast_len", 12)
        self.slow_len = kwargs.get("slow_len", 26)
        self.signal_len = kwargs.get("signal_len", 9)
        self._macd_computed = False

    def _compute_macd(self, df):
        df["EMA_fast"] = df["Close"].ewm(span=self.fast_len, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.slow_len, adjust=False).mean()
        df["MACD_Line"] = df["EMA_fast"] - df["EMA_slow"]
        df["Signal_Line"] = df["MACD_Line"].ewm(span=self.signal_len, adjust=False).mean()
        df["Histogram"] = df["MACD_Line"] - df["Signal_Line"]
        self._macd_computed = True
        return df

    def generate_signal(self, i, df):
        if not self._macd_computed:
            df[:] = self._compute_macd(df)

        if i == 0:
            return None

        prev = df.iloc[i-1]
        row = df.iloc[i]

        # Histogram cross
        enter_long = (prev["Histogram"] <= 0) and (row["Histogram"] > 0)
        enter_short = (prev["Histogram"] >= 0) and (row["Histogram"] < 0)

        return "long" if enter_long else "short" if enter_short else None

    def plot(self, df, trades_df):
        if not self._macd_computed:
            df[:] = self._compute_macd(df)

        fig, ax1 = plt.subplots(figsize=(16,7))

        # Price & signals
        ax1.plot(df["Date"], df["Close"], label="Close Price", color="black")
        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            ax1.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Entry")
            ax1.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Entry")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper left", title="Price & Signals")

        # MACD components on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(df["Date"], df["MACD_Line"], label="MACD Line", color="blue")
        ax2.plot(df["Date"], df["Signal_Line"], label="Signal Line", color="red", linestyle="--")
        ax2.bar(df["Date"], df["Histogram"], label="Histogram", color="gray", alpha=0.3)
        ax2.set_ylabel("MACD")
        ax2.legend(loc="upper right", title="MACD Components")

        plt.title("MACD Strategy: Price, Signals & MACD Components")
        plt.tight_layout()
        plt.show()
class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy:
      - Long if change and momentum > 0
      - Short if change and momentum < 0
    """
    PARAMS_SCHEMA = [
        {"name": "period", "label": "Momentum Period", "type": "number", "default": 5},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.period = kwargs.get("period", 5)
        self._momentum_computed = False

    def _compute_momentum(self, df):
        df["Change"] = df["Close"].diff(self.period)
        df["Momentum"] = df["Change"].diff()
        self._momentum_computed = True
        return df

    def generate_signal(self, i, df):
        if not self._momentum_computed:
            df[:] = self._compute_momentum(df)

        if i < self.period:
            return None

        row = df.iloc[i]
        enter_long = (row["Change"] > 0) and (row["Momentum"] > 0)
        enter_short = (row["Change"] < 0) and (row["Momentum"] < 0)

        return "long" if enter_long else "short" if enter_short else None

    def plot(self, df, trades_df):
        if not self._momentum_computed:
            df[:] = self._compute_momentum(df)

        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")
        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Entry")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Entry")
        plt.title(f"Momentum Strategy: Price & Signals (Period={self.period})")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class MovingAvgCrossStrategy(BaseStrategy):
    """
    Moving Average 2-Line Cross Strategy:
      - Long if Fast MA crosses above Slow MA
      - Short if Fast MA crosses below Slow MA
    """
    PARAMS_SCHEMA = [
        {"name": "fast_len", "label": "Fast MA Length", "type": "number", "default": 9},
        {"name": "slow_len", "label": "Slow MA Length", "type": "number", "default": 18},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fast_len = kwargs.get("fast_len", 9)
        self.slow_len = kwargs.get("slow_len", 18)
        self._ma_computed = False

    def _compute_ma(self, df):
        df["Fast_MA"] = df["Close"].rolling(self.fast_len).mean()
        df["Slow_MA"] = df["Close"].rolling(self.slow_len).mean()
        self._ma_computed = True
        return df

    def generate_signal(self, i, df):
        if not self._ma_computed:
            df[:] = self._compute_ma(df)

        if i == 0:
            return None

        prev = df.iloc[i-1]
        row = df.iloc[i]

        enter_long = (prev["Fast_MA"] <= prev["Slow_MA"]) and (row["Fast_MA"] > row["Slow_MA"])
        enter_short = (prev["Fast_MA"] >= prev["Slow_MA"]) and (row["Fast_MA"] < row["Slow_MA"])

        return "long" if enter_long else "short" if enter_short else None

    def plot(self, df, trades_df):
        if not self._ma_computed:
            df[:] = self._compute_ma(df)

        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")
        plt.plot(df["Date"], df["Fast_MA"], label=f"Fast MA ({self.fast_len})", color="blue", linestyle="-")
        plt.plot(df["Date"], df["Slow_MA"], label=f"Slow MA ({self.slow_len})", color="red", linestyle="--")

        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Entry")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Entry")

        plt.title("MovingAvg2Line Cross Strategy: Price, MAs & Signals")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class MovingAvgCrossConfirmStrategy(BaseStrategy):
    """
    Moving Average Cross Strategy with ConfirmBars:
      - Long if price stays above MA for `confirm_bars` consecutive bars
      - Short if price stays below MA for `confirm_bars` consecutive bars
    """
    PARAMS_SCHEMA = [
        {"name": "ma_len", "label": "MA Length", "type": "number", "default": 9},
        {"name": "confirm_bars", "label": "Confirm Bars", "type": "number", "default": 3},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ma_len = kwargs.get("ma_len", 9)
        self.confirm_bars = kwargs.get("confirm_bars", 3)
        self._ma_computed = False

    def _compute_ma(self, df):
        df["MA"] = df["Close"].rolling(self.ma_len).mean()
        df["above_ma"] = (df["Close"] > df["MA"]).astype(int)
        df["below_ma"] = (df["Close"] < df["MA"]).astype(int)
        df["consec_above"] = df["above_ma"].rolling(self.confirm_bars).sum()
        df["consec_below"] = df["below_ma"].rolling(self.confirm_bars).sum()
        self._ma_computed = True
        return df

    def generate_signal(self, i, df):
        if not self._ma_computed:
            df[:] = self._compute_ma(df)

        row = df.iloc[i]

        enter_long = row["consec_above"] >= self.confirm_bars
        enter_short = row["consec_below"] >= self.confirm_bars

        return "long" if enter_long else "short" if enter_short else None

    def plot(self, df, trades_df):
        if not self._ma_computed:
            df[:] = self._compute_ma(df)

        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black")
        plt.plot(df["Date"], df["MA"], label=f"MA ({self.ma_len})", color="blue", linestyle="-")

        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Entry")
            plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Entry")

        plt.title("MovingAvg Cross Confirm Strategy: Price, MA & Signals")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class OutSideBarStrategy(BaseStrategy):
    """
    OutSide Bar Strategy:
      - Identify outside bars (current high > prev high and current low < prev low)
      - Go long if close > open, short if close < open
    """
    PARAMS_SCHEMA = []  # pas de paramètres pour cette stratégie

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bars_computed = False

    def _compute_outside_bars(self, df):
        df["Outside"] = ((df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1)))
        df["Side"] = df.apply(lambda x: "long" if x["Close"] > x["Open"] else "short", axis=1)
        self._bars_computed = True
        return df

    def generate_signal(self, i, df):
        if not self._bars_computed:
            df[:] = self._compute_outside_bars(df)

        row = df.iloc[i]
        enter_long = row["Outside"] and row["Side"] == "long"
        enter_short = row["Outside"] and row["Side"] == "short"

        return "long" if enter_long else "short" if enter_short else None

    def plot(self, df, trades_df):
        if not self._bars_computed:
            df[:] = self._compute_outside_bars(df)

        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(16,7))

        # Candlesticks
        for idx, row in df.iterrows():
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            ax.plot([row['Date'], row['Date']], [row['Low'], row['High']], color='black')  # High-low line
            ax.add_patch(plt.Rectangle(
                (mdates.date2num(row['Date'])-0.2, min(row['Open'], row['Close'])),
                0.4, abs(row['Close']-row['Open']), color=color
            ))

        # Trade signals
        if not trades_df.empty:
            longs = trades_df[trades_df["side"]=="long"]
            shorts = trades_df[trades_df["side"]=="short"]
            ax.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="blue", s=100, label="Long Entry")
            ax.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="orange", s=100, label="Short Entry")

        ax.xaxis_date()
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.set_title("OutSide Bar Strategy: Candles & Trade Entries")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

class ParabolicSARStrategy(BaseStrategy):
    """
    Parabolic SAR Strategy:
      - Compute PSAR manually
      - Go long if trend switches from short to long
      - Go short if trend switches from long to short
    """
    PARAMS_SCHEMA = [
        {"name": "start", "label": "Start AF", "type": "number", "default": 0.02},
        {"name": "increment", "label": "AF Increment", "type": "number", "default": 0.02},
        {"name": "max_af", "label": "Max AF", "type": "number", "default": 0.2},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start = kwargs.get("start", 0.02)
        self.increment = kwargs.get("increment", 0.02)
        self.max_af = kwargs.get("max_af", 0.2)
        self._psar_computed = False

    def _compute_psar(self, df):
        # keep original df shape/order; return a copy with PSAR/Trend columns
        d = df.copy().reset_index(drop=True)
        n = len(d)
        psar = [0]*n
        trend = [1]*n  # 1 = long, -1 = short
        af = self.start
        ep = d["High"].iloc[0]  # Initial extreme point
        psar[0] = d["Low"].iloc[0]

        for i in range(1, n):
            psar[i] = psar[i-1] + af*(ep - psar[i-1])
            if trend[i-1] == 1:
                if d["Low"].iloc[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = ep
                    af = self.start
                    ep = d["Low"].iloc[i]
                else:
                    trend[i] = 1
                    if d["High"].iloc[i] > ep:
                        ep = d["High"].iloc[i]
                        af = min(af + self.increment, self.max_af)
            else:
                if d["High"].iloc[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = ep
                    af = self.start
                    ep = d["High"].iloc[i]
                else:
                    trend[i] = -1
                    if d["Low"].iloc[i] < ep:
                        ep = d["Low"].iloc[i]
                        af = min(af + self.increment, self.max_af)

        # attach columns to the copy and return
        d["PSAR"] = psar
        d["Trend"] = trend
        return d

    def generate_signal(self, i, df):
        # compute and inject PSAR/Trend into the df used by backtester, only once
        if not self._psar_computed:
            detected = self._compute_psar(df)
            # assign explicitly ensuring alignment by position
            # (use values sliced to len(df) in case of mismatch)
            df.loc[:, "PSAR"] = detected["PSAR"].values[: len(df)]
            df.loc[:, "Trend"] = detected["Trend"].values[: len(df)]
            self._psar_computed = True

        # safe guards
        if i == 0:
            return None
        if i - 1 < 0 or i >= len(df):
            return None

        prev = df.iloc[i - 1]
        row = df.iloc[i]

        # use .get in case Trend missing for some reason
        prev_trend = prev.get("Trend", None)
        row_trend = row.get("Trend", None)

        enter_long = (prev_trend == -1) and (row_trend == 1)
        enter_short = (prev_trend == 1) and (row_trend == -1)

        if enter_long:
            return "long"
        if enter_short:
            return "short"
        return None

    def plot(self, df, trades_df):
        # ensure psar computed and columns present
        if not self._psar_computed:
            detected = self._compute_psar(df)
            df.loc[:, "PSAR"] = detected["PSAR"].values[: len(df)]
            df.loc[:, "Trend"] = detected["Trend"].values[: len(df)]
            self._psar_computed = True

        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], color="black", label="Close Price")
        if "PSAR" in df.columns:
            plt.scatter(df["Date"], df["PSAR"], color="blue", label="Parabolic SAR", s=20)

        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]
            if not longs.empty:
                plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="green", s=100, label="Long Entry")
            if not shorts.empty:
                plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="red", s=100, label="Short Entry")

        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title("Parabolic SAR Strategy: Price & Signals")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class PivotExtensionStrategy(BaseStrategy):
    """
    Pivot Extension Strategy:
      - Detect pivot highs / pivot lows (left, right)
      - When PivotHigh is flagged -> enter SHORT at next bar open
      - When PivotLow  is flagged -> enter LONG  at next bar open
      - Strategy parameters: left, right (both integers)
    """
    PARAMS_SCHEMA = [
        {"name": "left", "label": "Left Pivot Bars", "type": "number", "default": 3},
        {"name": "right", "label": "Right Pivot Bars", "type": "number", "default": 3},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # initialize params (BaseStrategy.__init__ already sets attributes from PARAMS_SCHEMA)
        self.left = int(kwargs.get("left", 3))
        self.right = int(kwargs.get("right", 3))
        self._pivots_computed = False

    def _detect_pivots(self, df):
        """
        Returns a copy of df with boolean columns 'PivotHigh' and 'PivotLow'.
        The detection follows the pattern in your function: a pivot at index i
        sets True at index i+right (so pivot is 'confirmed' after right bars).
        """
        d = df.copy().reset_index(drop=False)  # keep original positional index in column 'index'
        n = len(d)
        d["PivotHigh"] = np.zeros(n, dtype=bool)
        d["PivotLow"] = np.zeros(n, dtype=bool)

        left = int(self.left)
        right = int(self.right)

        # guard: need at least left+right+1 bars
        if n < left + right + 1:
            # return columns filled False but keep Date etc.
            d["PivotHigh"] = d["PivotHigh"].astype(bool)
            d["PivotLow"] = d["PivotLow"].astype(bool)
            return d

        for i in range(left, n - right):
            highs = d["High"].iloc[i-left : i+right+1]
            lows  = d["Low"].iloc[i-left : i+right+1]

            # pivot high at i?
            if d["High"].iloc[i] == highs.max():
                tgt = i + right
                if tgt < n:
                    d.at[tgt, "PivotHigh"] = True

            # pivot low at i?
            if d["Low"].iloc[i] == lows.min():
                tgt = i + right
                if tgt < n:
                    d.at[tgt, "PivotLow"] = True

        d["PivotHigh"] = d["PivotHigh"].fillna(False).astype(bool)
        d["PivotLow"]  = d["PivotLow"].fillna(False).astype(bool)

        return d

    def generate_signal(self, i, df):
        """
        Called by generic backtester at index i (signal will be executed at i+1 open).
        We compute pivots once and keep flags in df. The detection writes booleans
        at i (already shifted by right inside _detect_pivots) so checking row i is correct.
        """
        if not self._pivots_computed:
            # compute and copy flags back into df (preserve original columns)
            detected = self._detect_pivots(df)
            # keep Date alignment: detected has same order as df.reset_index()
            # replace df in-place so plotting/other methods see Pivot flags
            # ensure we preserve df shape/columns but add flags
            df_length = len(df)
            # detected is a reset-index copy; align by position
            df.loc[:, "PivotHigh"] = detected["PivotHigh"].values[:df_length]
            df.loc[:, "PivotLow"]  = detected["PivotLow"].values[:df_length]
            self._pivots_computed = True

        # safe guard for edges
        if i < 0 or i >= len(df):
            return None

        row = df.iloc[i]
        pivot_high = bool(row.get("PivotHigh", False))
        pivot_low  = bool(row.get("PivotLow", False))

        enter_short = pivot_high
        enter_long  = pivot_low

        if enter_long:
            return "long"
        if enter_short:
            return "short"
        return None

    def plot(self, df, trades_df):
        """
        Price plot with pivot markers and trade entries.
        Matches the visual style used for other strategies.
        """
        # Ensure pivots computed so we can plot them
        if not self._pivots_computed:
            detected = self._detect_pivots(df)
            df.loc[:, "PivotHigh"] = detected["PivotHigh"].values[:len(df)]
            df.loc[:, "PivotLow"]  = detected["PivotLow"].values[:len(df)]
            self._pivots_computed = True

        plt.figure(figsize=(16,7))
        plt.plot(df["Date"], df["Close"], label="Close Price", color="black", linewidth=1.2)

        # pivot markers (plot only where True)
        ph_mask = df["PivotHigh"].fillna(False).astype(bool)
        pl_mask = df["PivotLow"].fillna(False).astype(bool)

        if ph_mask.any():
            plt.scatter(df.loc[ph_mask, "Date"], df.loc[ph_mask, "High"], marker="v", color="red", s=100, label="Pivot High")
        if pl_mask.any():
            plt.scatter(df.loc[pl_mask, "Date"], df.loc[pl_mask, "Low"], marker="^", color="green", s=100, label="Pivot Low")

        # trade entries (if any)
        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]
            if not longs.empty:
                plt.scatter(longs["entry_date"], longs["entry_price"], marker="^", color="blue", s=120, label="Long Entry")
            if not shorts.empty:
                plt.scatter(shorts["entry_date"], shorts["entry_price"], marker="v", color="orange", s=120, label="Short Entry")

        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.title(f"Pivot Extension Strategy (L={self.left}, R={self.right}) — Price, Pivots & Entries")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class PriceChannelStrategy(BaseStrategy):
    """
    Price Channel Breakout Strategy:
      - Compute rolling Upper/Lower channel
      - Long if price breaks above Upper + trend filter
      - Short if price breaks below Lower + trend filter
      - SL/TP based on ATR multiples
    """
    PARAMS_SCHEMA = [
        {"name": "length", "label": "Channel Length", "type": "number", "default": 20},
        {"name": "atr_length", "label": "ATR Length", "type": "number", "default": 14},
        {"name": "risk_per_trade", "label": "Risk % per Trade", "type": "number", "default": 0.01},
        {"name": "k_sl", "label": "ATR Multiplier SL", "type": "number", "default": 1.5},
        {"name": "k_tp", "label": "ATR Multiplier TP", "type": "number", "default": 3.0},
        {"name": "use_trend_filter", "label": "Use EMA50 Filter", "type": "boolean", "default": True}
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.length = int(kwargs.get("length", 20))
        self.atr_length = int(kwargs.get("atr_length", 14))
        self.risk_per_trade = float(kwargs.get("risk_per_trade", 0.01))
        self.k_sl = float(kwargs.get("k_sl", 1.5))
        self.k_tp = float(kwargs.get("k_tp", 3.0))
        self.use_trend_filter = bool(kwargs.get("use_trend_filter", True))

        self._computed_indicators = False

    # -------------------------------------------------------
    def _compute_indicators(self, df):
        d = df.copy()

        # Price Channels
        d["Upper"] = d["High"].rolling(self.length).max()
        d["Lower"] = d["Low"].rolling(self.length).min()

        # ATR
        d["H-L"] = d["High"] - d["Low"]
        d["H-PC"] = (d["High"] - d["Close"].shift(1)).abs()
        d["L-PC"] = (d["Low"] - d["Close"].shift(1)).abs()
        d["TR"] = d[["H-L", "H-PC", "L-PC"]].max(axis=1)
        d["ATR"] = d["TR"].rolling(self.atr_length).mean()

        # Trend filter
        if self.use_trend_filter:
            d["EMA50"] = d["Close"].ewm(span=50).mean()
        else:
            d["EMA50"] = d["Close"]  # neutral

        return d

    # -------------------------------------------------------
    def generate_signal(self, i, df):
        """
        Return:
          - "long"
          - "short"
          - None
        """
        if not self._computed_indicators:
            ind = self._compute_indicators(df)
            df.loc[:, ind.columns] = ind
            self._computed_indicators = True

        if i <= self.length + 50:
            return None

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        atr = prev["ATR"]
        if atr is None or np.isnan(atr) or atr == 0:
            return None

        # ---------------- ENTRY LOGIC --------------------
        long_ok = (
            row["High"] >= prev["Upper"]
            and row["Close"] > prev["EMA50"]
        )

        short_ok = (
            row["Low"] <= prev["Lower"]
            and row["Close"] < prev["EMA50"]
        )

        if long_ok:
            return "long"
        if short_ok:
            return "short"

        return None

    # -------------------------------------------------------
    def plot(self, df, trades_df):
        """
        Price chart with channel + trade markers.
        """
        if not self._computed_indicators:
            ind = self._compute_indicators(df)
            df.loc[:, ind.columns] = ind
            self._computed_indicators = True

        plt.figure(figsize=(16, 7))
        plt.plot(df["Date"], df["Close"], color="black", label="Close")

        if "Upper" in df:
            plt.plot(df["Date"], df["Upper"], color="blue", linestyle="--", label="Upper")
        if "Lower" in df:
            plt.plot(df["Date"], df["Lower"], color="red", linestyle="--", label="Lower")

        # Trade markers
        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]

            if not longs.empty:
                plt.scatter(longs["entry_date"], longs["entry_price"],
                            marker="^", color="green", s=100, label="Long Entry")

            if not shorts.empty:
                plt.scatter(shorts["entry_date"], shorts["entry_price"],
                            marker="v", color="orange", s=100, label="Short Entry")

            if "exit_date" in trades_df:
                plt.scatter(trades_df["exit_date"], trades_df["exit_price"],
                            marker="x", color="red", s=100, label="Exit")

        plt.title("Price Channel Strategy — Channel + Entries")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

class ADXBreakoutStrategy(BaseStrategy):
    """
    ADX Breakout Strategy:
      - ADX identifies strong trend
      - Buy if Close > previous Close and ADX >= threshold
      - Sell if Close < previous Close and ADX >= threshold
    """

    PARAMS_SCHEMA = [
        {"name": "adx_period", "label": "ADX Period", "type": "number", "default": 14},
        {"name": "adx_threshold", "label": "ADX Threshold", "type": "number", "default": 25},
        {"name": "risk_per_trade", "label": "Risk % per Trade", "type": "number", "default": 0.01}
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adx_period = int(kwargs.get("adx_period", 14))
        self.adx_threshold = float(kwargs.get("adx_threshold", 25))
        self.risk_per_trade = float(kwargs.get("risk_per_trade", 0.01))

        self._computed_indicators = False

    # -------------------------------------------------------
    def _compute_indicators(self, df):
        d = df.copy()

        # TR, DM+, DM-
        d["H-L"] = d["High"] - d["Low"]
        d["H-PC"] = (d["High"] - d["Close"].shift(1)).abs()
        d["L-PC"] = (d["Low"] - d["Close"].shift(1)).abs()
        d["TR"] = d[["H-L", "H-PC", "L-PC"]].max(axis=1)

        d["+DM"] = np.where(
            (d["High"] - d["High"].shift(1)) > (d["Low"].shift(1) - d["Low"]),
            np.maximum(d["High"] - d["High"].shift(1), 0),
            0
        )
        d["-DM"] = np.where(
            (d["Low"].shift(1) - d["Low"]) > (d["High"] - d["High"].shift(1)),
            np.maximum(d["Low"].shift(1) - d["Low"], 0),
            0
        )

        # Smoothed values
        per = self.adx_period
        d["TR_smooth"] = d["TR"].rolling(per).sum()
        d["+DM_smooth"] = d["+DM"].rolling(per).sum()
        d["-DM_smooth"] = d["-DM"].rolling(per).sum()

        d["+DI"] = 100 * d["+DM_smooth"] / d["TR_smooth"]
        d["-DI"] = 100 * d["-DM_smooth"] / d["TR_smooth"]

        d["DX"] = 100 * (d["+DI"] - d["-DI"]).abs() / (d["+DI"] + d["-DI"])
        d["ADX"] = d["DX"].rolling(per).mean()

        return d

    # -------------------------------------------------------
    def generate_signal(self, i, df):
        """
        Return:
            "long", "short", or None
        """
        if not self._computed_indicators:
            ind = self._compute_indicators(df)
            df.loc[:, ind.columns] = ind
            self._computed_indicators = True

        if i <= self.adx_period + 1:
            return None

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Entry conditions
        enter_long = row["Close"] > prev["Close"] and row["ADX"] >= self.adx_threshold
        enter_short = row["Close"] < prev["Close"] and row["ADX"] >= self.adx_threshold

        if enter_long:
            return "long"
        if enter_short:
            return "short"

        return None

    # -------------------------------------------------------
    def plot(self, df, trades_df):
        """
        Price chart + ADX indicator.
        """
        if not self._computed_indicators:
            ind = self._compute_indicators(df)
            df.loc[:, ind.columns] = ind
            self._computed_indicators = True

        fig, ax1 = plt.subplots(figsize=(16, 7))

        # Price
        ax1.plot(df["Date"], df["Close"], color="black", label="Close Price")

        # Trades
        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]

            if not longs.empty:
                ax1.scatter(longs["entry_date"], longs["entry_price"],
                            marker="^", color="green", s=100, label="Long Entry")

            if not shorts.empty:
                ax1.scatter(shorts["entry_date"], shorts["entry_price"],
                            marker="v", color="red", s=100, label="Short Entry")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper left")

        # ADX
        ax2 = ax1.twinx()
        ax2.plot(df["Date"], df["ADX"], color="blue", label="ADX", linewidth=1.5)
        ax2.axhline(self.adx_threshold, color="gray", linestyle="--", label="ADX Threshold")

        ax2.set_ylabel("ADX")
        ax2.legend(loc="upper right")

        plt.title("ADX Breakout Strategy — Price + ADX")
        plt.tight_layout()
        plt.show()

class StochasticSlowStrategy(BaseStrategy):
    """
    Stochastic Slow Strategy:
      - Compute %K and %D (slow)
      - Buy when %K crosses above %D in oversold zone
      - Sell when %K crosses below %D in overbought zone
    """

    PARAMS_SCHEMA = [
        {"name": "k_period", "label": "%K Lookback", "type": "number", "default": 14},
        {"name": "k_smooth", "label": "%K Smoothing", "type": "number", "default": 3},
        {"name": "d_smooth", "label": "%D Smoothing", "type": "number", "default": 3},
        {"name": "oversold", "label": "Oversold Level", "type": "number", "default": 20},
        {"name": "overbought", "label": "Overbought Level", "type": "number", "default": 80},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k_period = int(kwargs.get("k_period", 14))
        self.k_smooth = int(kwargs.get("k_smooth", 3))
        self.d_smooth = int(kwargs.get("d_smooth", 3))
        self.oversold = float(kwargs.get("oversold", 20))
        self.overbought = float(kwargs.get("overbought", 80))

        self._computed = False

    # -------------------------------------------------------
    def _compute_stochastic(self, df):
        d = df.copy()

        d["L_k"] = d["Low"].rolling(self.k_period).min()
        d["H_k"] = d["High"].rolling(self.k_period).max()

        d["K_raw"] = 100 * (d["Close"] - d["L_k"]) / (d["H_k"] - d["L_k"])
        d["K"] = d["K_raw"].rolling(self.k_smooth).mean()
        d["D"] = d["K"].rolling(self.d_smooth).mean()

        return d

    # -------------------------------------------------------
    def generate_signal(self, i, df):
        if not self._computed:
            st = self._compute_stochastic(df)
            df.loc[:, st.columns] = st
            self._computed = True

        # Not enough bars
        if i < self.k_period + self.d_smooth:
            return None

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Cross up in oversold → LONG
        cross_up = (prev["K"] < prev["D"]) and (row["K"] > row["D"])
        long_signal = cross_up and (row["K"] < self.oversold)

        # Cross down in overbought → SHORT
        cross_down = (prev["K"] > prev["D"]) and (row["K"] < row["D"])
        short_signal = cross_down and (row["K"] > self.overbought)

        if long_signal:
            return "long"
        if short_signal:
            return "short"
        return None

    # -------------------------------------------------------
    def plot(self, df, trades_df):
        if not self._computed:
            st = self._compute_stochastic(df)
            df.loc[:, st.columns] = st
            self._computed = True

        fig, ax1 = plt.subplots(figsize=(16, 7))

        # Price
        ax1.plot(df["Date"], df["Close"], color="black", label="Close Price")

        # Trades
        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]

            if not longs.empty:
                ax1.scatter(longs["entry_date"], longs["entry_price"],
                            marker="^", color="green", s=100, label="Long Entry")
            if not shorts.empty:
                ax1.scatter(shorts["entry_date"], shorts["entry_price"],
                            marker="v", color="red", s=100, label="Short Entry")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper left")

        # Stochastic indicator
        ax2 = ax1.twinx()
        ax2.plot(df["Date"], df["K"], color="blue", label="%K", linewidth=1.5)
        ax2.plot(df["Date"], df["D"], color="orange", label="%D", linewidth=1.5)

        ax2.axhline(self.oversold, color="green", linestyle="--", label="Oversold")
        ax2.axhline(self.overbought, color="red", linestyle="--", label="Overbought")

        ax2.set_ylabel("Stochastic %")
        ax2.legend(loc="upper right")

        plt.title("Stochastic Slow Strategy — Price & Stochastic")
        plt.tight_layout()
        plt.show()

class SupertrendStrategy(BaseStrategy):
    """
    Supertrend Strategy:
      - Computes Supertrend (ATR-based)
      - Buy when trend flips from short → long
      - Sell when trend flips from long → short
    """

    PARAMS_SCHEMA = [
        {"name": "atr_period", "label": "ATR Period", "type": "number", "default": 10},
        {"name": "factor", "label": "ATR Multiplier", "type": "number", "default": 3},
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atr_period = int(kwargs.get("atr_period", 10))
        self.factor = float(kwargs.get("factor", 3))

        self._computed = False

    # -------------------------------------------------------
    def _compute_supertrend(self, df):
        d = df.copy()

        # True Range
        d["TR"] = np.maximum(
            d["High"] - d["Low"],
            np.maximum(
                abs(d["High"] - d["Close"].shift(1)),
                abs(d["Low"] - d["Close"].shift(1)),
            ),
        )
        d["ATR"] = d["TR"].rolling(self.atr_period).mean()

        # Bands
        hl2 = (d["High"] + d["Low"]) / 2
        d["UpperBand"] = hl2 + self.factor * d["ATR"]
        d["LowerBand"] = hl2 - self.factor * d["ATR"]

        st = [np.nan] * len(d)
        st_dir = [1] * len(d)  # 1 = long trend, -1 = short trend

        for i in range(self.atr_period, len(d)):
            if i == self.atr_period:
                st[i] = d["LowerBand"].iloc[i]
                st_dir[i] = 1
                continue

            prev_st = st[i - 1]
            curr_close = d["Close"].iloc[i]
            upper = d["UpperBand"].iloc[i]
            lower = d["LowerBand"].iloc[i]

            # Fix NaN floating comparison
            if np.isnan(prev_st):
                prev_st = lower

            if st_dir[i - 1] == 1:  # was long
                if curr_close <= upper:
                    st[i] = upper
                    st_dir[i] = -1
                else:
                    st[i] = max(lower, prev_st)
                    st_dir[i] = 1
            else:  # was short
                if curr_close >= lower:
                    st[i] = lower
                    st_dir[i] = 1
                else:
                    st[i] = min(upper, prev_st)
                    st_dir[i] = -1

        d["Supertrend"] = st
        d["ST_dir"] = st_dir

        return d

    # -------------------------------------------------------
    def generate_signal(self, i, df):
        # Compute once
        if not self._computed:
            st = self._compute_supertrend(df)
            df.loc[:, st.columns] = st
            self._computed = True

        if i <= self.atr_period:
            return None

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Trend change
        enter_long = prev["ST_dir"] == -1 and row["ST_dir"] == 1
        enter_short = prev["ST_dir"] == 1 and row["ST_dir"] == -1

        if enter_long:
            return "long"
        if enter_short:
            return "short"
        return None

    # -------------------------------------------------------
    def plot(self, df, trades_df):
        if not self._computed:
            st = self._compute_supertrend(df)
            df.loc[:, st.columns] = st
            self._computed = True

        fig, ax1 = plt.subplots(figsize=(16, 7))

        # Price
        ax1.plot(df["Date"], df["Close"], label="Close Price", color="black")

        # Trades
        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]

            if not longs.empty:
                ax1.scatter(
                    longs["entry_date"],
                    longs["entry_price"],
                    marker="^",
                    color="green",
                    s=100,
                    label="Long Entry",
                )
            if not shorts.empty:
                ax1.scatter(
                    shorts["entry_date"],
                    shorts["entry_price"],
                    marker="v",
                    color="red",
                    s=100,
                    label="Short Entry",
                )

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.legend(loc="upper left")

        # Supertrend line
        ax2 = ax1.twinx()
        ax2.plot(df["Date"], df["Supertrend"], color="blue", label="Supertrend")
        ax2.set_ylabel("Supertrend")
        ax2.legend(loc="upper right")

        plt.title("Supertrend Strategy — Price & Supertrend")
        plt.tight_layout()
        plt.show()
