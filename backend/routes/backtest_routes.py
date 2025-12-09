import inspect
from fastapi import APIRouter
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <-- backend non-interactif
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
import base64

from backtesting.backtest import Backtester, BaseStrategy
import backtesting.strategies as strategies

router = APIRouter()

# ----------------- Load Strategies -----------------
STRATEGIES = {
    cls.__name__: cls
    for name, cls in inspect.getmembers(strategies, inspect.isclass)
    if issubclass(cls, BaseStrategy) and cls is not BaseStrategy
}

# ----------------- Pydantic -----------------
from pydantic import BaseModel
from typing import Optional, Dict

class BacktestRequest(BaseModel):
    ticker: str
    strategy_name: str
    initial_capital: Optional[float] = 100000
    max_intraday_loss_pct: Optional[float] = 2
    lookback_days: Optional[int] = 252
    strategy_params: Optional[Dict] = None


# ----------------- Helper -----------------
def init_backtester(req: BacktestRequest):
    data = yf.download(
        req.ticker, period=f"{req.lookback_days}d", auto_adjust=False
    )[["Open","High","Low","Close"]]

    # Si l'index est datetime, on le convertit en colonne 'Date'
    if not "Date" in data.columns:
        data = data.reset_index()

    strategy_cls = STRATEGIES[req.strategy_name]
    strategy = strategy_cls(**(req.strategy_params or {}))
    bt = Backtester(
        data,
        strategy,
        initial_capital=req.initial_capital,
        max_intraday_loss_pct=req.max_intraday_loss_pct,
        lookback_days=req.lookback_days
    )
    return bt


# ----------------- Endpoints -----------------

@router.post("/metrics")
def backtest_metrics(req: BacktestRequest):
    bt = init_backtester(req)
    bt.run()
    try:
        metrics = bt.compute_metrics()
        return {"metrics": metrics}
    except KeyError as e:
        if str(e) == "'pnl'":
            return {"error": "No trades found. Adjust the strategy parameters."}
        else:
            raise e



@router.post("/trades")
def backtest_trades(req: BacktestRequest):
    bt = init_backtester(req)
    bt.run()

    # ----- PRICE DATA -----
    price_data = bt.data[["Date","Open","High","Low","Close"]].copy()
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    price_data["Date"] = price_data["Date"].dt.strftime("%Y-%m-%d")
    price_data = price_data.to_dict(orient="records")

    # ----- TRADES DATA -----
    trades_df = bt.trades_df.copy()
    trade_dates = []

    if not trades_df.empty:

        def align_date_to_bt(date):
            """
            Aligne une date avec le DataFrame du backtester pour s'assurer
            qu'elle correspond à une date valide (jours ouvrés uniquement).
            """
            date = pd.to_datetime(date)
            future_dates = bt.data[bt.data["Date"] >= date]["Date"]
            return future_dates.iloc[0] if not future_dates.empty else bt.data["Date"].iloc[-1]

        # Décalage pour stratégies avec window (ex: BollingerBands)
        strategy_window = getattr(bt.strategy, "window", 0)

        trades_df["entry_date"] = trades_df["entry_date"].apply(align_date_to_bt)
        trades_df["exit_date"] = trades_df["exit_date"].apply(
            lambda x: align_date_to_bt(x) if pd.notnull(x) else None
        )

        # Si stratégie a une fenêtre, s'assurer que les trades commencent après cette fenêtre
        if strategy_window > 0:
            first_valid_date = bt.data["Date"].iloc[strategy_window]
            trades_df["entry_date"] = trades_df["entry_date"].apply(lambda d: max(d, first_valid_date))
            trades_df["exit_date"] = trades_df["exit_date"].apply(lambda d: max(d, first_valid_date) if d is not None else None)

        trades_df["entry_date"] = trades_df["entry_date"].dt.strftime("%Y-%m-%d")
        trades_df["exit_date"] = trades_df["exit_date"].apply(
            lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notnull(x) else None
        )

        trades_df["entry_price"] = trades_df["entry_price"].astype(float)
        trades_df["exit_price"] = trades_df["exit_price"].astype(float)
        trades_df["price"] = trades_df["entry_price"]

        # Collect unique trade dates pour le frontend
        trade_dates = sorted(set(trades_df["entry_date"].tolist() + trades_df["exit_date"].dropna().tolist()))

    trades = trades_df.to_dict(orient="records")

    return {
        "price_data": price_data,
        "trades": trades,
        "trade_dates": trade_dates,
    }



@router.post("/equity_curve")
def backtest_equity_curve(req: BacktestRequest):
    bt = init_backtester(req)
    bt.run()
    equity_df = bt.equity_df
    return {
        "equity_curve": equity_df["Equity"].tolist(),
        "dates": equity_df.index.strftime("%Y-%m-%d").tolist()
    }


@router.post("/buy_and_hold")
def backtest_buy_and_hold(req: BacktestRequest):
    bt = init_backtester(req)
    bh_curve = bt.compute_buy_and_hold()
    return {
        "buy_and_hold": bh_curve.tolist(),
        "dates": bh_curve.index.strftime("%Y-%m-%d").tolist()
    }


@router.post("/plot")
def backtest_plot(req: BacktestRequest):
    bt = init_backtester(req)
    bt.run()

    # ----- Forcer backend non-interactif (déjà fait au top du fichier) -----
    # matplotlib.use("Agg")

    # Créer un buffer pour capturer l'image
    buf = BytesIO()
    
    # Appeler la méthode plot de la stratégie
    # Si la méthode plot crée des figures, il faut s'assurer qu'elle ne tente pas d'ouvrir une fenêtre GUI
    bt.strategy.plot(bt.data, bt.trades_df)

    # Récupérer la figure courante et la sauvegarder dans le buffer
    fig = plt.gcf()  # Get current figure (générée par bt.strategy.plot)
    fig.savefig(buf, format='png', bbox_inches='tight')  # PNG
    plt.close(fig)  # Fermer la figure pour éviter les crashs GUI
    buf.seek(0)

    # Encoder en base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return {"plot_base64": img_base64}

@router.post("/performance_plot")
def backtest_performance_plot(req: BacktestRequest):
    bt = init_backtester(req)
    bt.run()

    # Courbes
    equity_df = bt.equity_df
    bh_curve = bt.compute_buy_and_hold()

    # ---- Création du graphique combiné ----
    fig, ax = plt.subplots(figsize=(12, 6))

    # Equity curve (Stratégie)
    ax.plot(
        equity_df.index,
        equity_df["Equity"],
        label="Strategy Equity",
        linewidth=2
    )

    # Buy & Hold curve
    ax.plot(
        bh_curve.index,
        bh_curve.values,
        label="Buy & Hold",
        linewidth=2
    )

    # Style
    ax.set_title(f"{req.ticker} — Strategy vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True)
    ax.legend()

    # ---- Conversion en Base64 ----
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {"plot_base64": img_base64}

@router.get("/strategy_params/{strategy_name}")
def get_strategy_params(strategy_name: str):
    strategy_cls = STRATEGIES.get(strategy_name)
    if not strategy_cls:
        return {"params": []}
    
    # Retourne le PARAMS_SCHEMA directement
    return {"params": getattr(strategy_cls, "PARAMS_SCHEMA", [])}
