import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# ================================================================
# ======================== BASE CLASSES ===========================
# ================================================================

class BaseStrategy:
    """
    Base class: defines the minimal interface required for a strategy.

    Each subclass must implement:
        - generate_signal(i, df) -> "long" | "short" | None
          (signal evaluated at index i, executed on i+1)
    """
    PARAMS_SCHEMA = []  # liste des paramètres par défaut (vide)

    def __init__(self, **kwargs):
        # Initialisation des paramètres à partir du schéma
        for param in getattr(self, "PARAMS_SCHEMA", []):
            name = param["name"]
            default = param.get("default")
            setattr(self, name, kwargs.get(name, default))
    def generate_signal(self, i, df):
        raise NotImplementedError("Each strategy must implement generate_signal().")

    def plot(self, df, trades_df):
        """Optional custom plot for the strategy."""
        pass


class Backtester:
    """
    Generic backtester usable with any strategy:
    - signal on day i
    - entry on next day's open
    - intraday stop
    - position size = equity / price
    """

    def __init__(
        self, 
        data, 
        strategy: BaseStrategy,
        initial_capital=10.0,
        max_intraday_loss_pct=2.0,
        lookback_days=None
    ):
        # Defensive copy
        self.data = data.copy()

        # Flatten yfinance MultiIndex
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = [col[0] for col in self.data.columns]

        # Keep only required columns
        self.data = self.data[["Open", "High", "Low", "Close"]].copy()

        # Ensure numeric
        for col in ["Open", "High", "Low", "Close"]:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        # Si 'Date' n'existe pas, le créer depuis l'index
        if "Date" not in data.columns:
            data = data.reset_index()  # index devient la colonne 'Date'

        self.data = data.copy()

        # Flatten yfinance MultiIndex
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = [col[0] for col in self.data.columns]

        # Garder uniquement les colonnes essentielles
        self.data = self.data[["Date", "Open", "High", "Low", "Close"]].copy()

        # Convertir Date en datetime (important)
        self.data["Date"] = pd.to_datetime(self.data["Date"])

        # Apply lookback
        if lookback_days is not None and lookback_days < len(self.data):
            self.data = self.data.iloc[-lookback_days:].reset_index(drop=True)



        # Parameters
        self.strategy = strategy
        self.initial_capital = float(initial_capital)
        self.max_intraday_loss_pct = float(max_intraday_loss_pct)
        self.lookback_days = lookback_days

    # ================================================================
    #                         RUN BACKTEST
    # ================================================================
    def run(self):
        df = self.data
        n = len(df)

        equity = self.initial_capital
        equity_curve = [{"Date": df.loc[0, "Date"], "Equity": equity}]
        trades = []
        position = None

        for i in range(0, n - 1):
            today = df.iloc[i]
            tomorrow = df.iloc[i + 1]

            # 1) Close existing position (stop + exit)
            if position:
                if position["side"] == "long":
                    stop_price = position["entry_price"] * (1 - self.max_intraday_loss_pct / 100)
                    exit_price = stop_price if tomorrow["Low"] <= stop_price else tomorrow["Close"]
                    pnl = (exit_price - position["entry_price"]) * position["size"]
                else:
                    stop_price = position["entry_price"] * (1 + self.max_intraday_loss_pct / 100)
                    exit_price = stop_price if tomorrow["High"] >= stop_price else tomorrow["Close"]
                    pnl = (position["entry_price"] - exit_price) * position["size"]

                position["exit_date"] = tomorrow["Date"]
                position["exit_price"] = float(exit_price)
                position["pnl"] = float(pnl)
                trades.append(position)

                equity += pnl
                position = None
                # Reset intraday pyramiding in strategy if needed
                if hasattr(self.strategy, "post_trade_update"):
                    self.strategy.post_trade_update(trade_closed=True)

            # 2) Get signal from strategy
            signal = self.strategy.generate_signal(i, df)

            # 3) Enter position
            if signal in ("long", "short"):
                entry_price = tomorrow["Open"]
                if not pd.isna(entry_price) and entry_price > 0:
                    size = equity / entry_price
                    position = {
                        "side": signal,
                        "entry_date": tomorrow["Date"],
                        "entry_price": float(entry_price),
                        "size": float(size),
                        "exit_date": None,
                        "exit_price": None,
                        "pnl": None,
                    }

            equity_curve.append({"Date": tomorrow["Date"], "Equity": equity})

        # 4) Close last position at final close
        if position:
            last = df.iloc[-1]
            exit_price = last["Close"]

            if position["side"] == "long":
                pnl = (exit_price - position["entry_price"]) * position["size"]
            else:
                pnl = (position["entry_price"] - exit_price) * position["size"]

            position["exit_date"] = last["Date"]
            position["exit_price"] = float(exit_price)
            position["pnl"] = float(pnl)

            equity += pnl
            
            trades.append(position)
            equity_curve.append({"Date": last["Date"], "Equity": equity})

        # Final DataFrames
        self.equity_df = pd.DataFrame(equity_curve).drop_duplicates("Date").set_index("Date")
        self.trades_df = pd.DataFrame(trades)

        return self.equity_df, self.trades_df

    # ===================== COMPUTE METRICS ======================
    def compute_metrics(self):
        if (not hasattr(self, "equity_df")) or (not hasattr(self, "trades_df")):
            equity_df, trades_df = self.run()
        else:
            equity_df, trades_df = self.equity_df, self.trades_df

        initial_capital = self.initial_capital

        if equity_df.empty:
            return {}

        final_equity = equity_df["Equity"].iloc[-1]
        total_pnl = final_equity - initial_capital
        total_pnl_pct = 100 * total_pnl / initial_capital

        # Max drawdown
        cum = equity_df["Equity"]
        running_max = cum.cummax()
        dd = running_max - cum
        max_dd_usd = dd.max()
        max_dd_pct = 100 * max_dd_usd / running_max.max()

        # Trades
        total_trades = len(trades_df)
        profitable = len(trades_df[trades_df["pnl"] > 0])
        gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        gross_loss = trades_df[trades_df["pnl"] < 0]["pnl"].sum()

        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")

        return {
            "final_equity": final_equity,
            "total_pnl_usd": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "max_drawdown_usd": max_dd_usd,
            "max_drawdown_pct": max_dd_pct,
            "total_trades": total_trades,
            "profitable_trades": profitable,
            "profit_factor": profit_factor,
        }

    # ===================== PLOT EQUITY ======================
    def plot_equity(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.equity_df.index, self.equity_df["Equity"], label="Strategy")
        plt.title("Equity Curve")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ===================== BUY & HOLD ======================
    def compute_buy_and_hold(self):
        """
        Uses exactly the same lookback window as the backtest.
        """
        df = self.data  # already cropped by lookback

        start_price = df["Open"].iloc[0]
        end_prices = df["Close"]

        # Correct index
        bh_curve = (end_prices / start_price) * self.initial_capital
        bh_curve.index = df["Date"]

        return bh_curve


    # ===================== PLOT STRATEGY VS BUY & HOLD ======================
    def plot_vs_buy_and_hold(self):
        bh = self.compute_buy_and_hold()

        plt.figure(figsize=(10, 6))
        plt.plot(self.equity_df.index, self.equity_df["Equity"], label="Strategy")
        plt.plot(bh.index, bh.values, label="Buy & Hold")
        plt.title("Strategy vs Buy & Hold")
        plt.grid(True)
        plt.legend()
        plt.show()

    
