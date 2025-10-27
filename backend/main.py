from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize

# ------------------ PORTFOLIO CLASS ------------------
class Portfolio:

    def __init__(self, start0, end0, tickers0, pos0, window=None):
        self.start = start0
        self.end = end0
        self.tickers = tickers0
        self.pos = pos0
        self.window = window

    def prices(self):
        # --- Télécharger les prix ---
        df = yf.download(self.tickers, start=self.start, end=self.end, progress=False, auto_adjust=True)['Close']
        df = df.dropna(how='all').ffill()

        '''# --- Récupérer la devise de chaque ticker ---
        currencies = {}
        for ticker in self.tickers:
            info = yf.Ticker(ticker).info
            currencies[ticker] = info.get("currency", "USD")  # default USD

        # --- Identifier les tickers non-USD ---
        non_usd_tickers = [t for t, cur in currencies.items() if cur != "USD"]

        # --- Convertir chaque ticker non-USD en USD ---
        for ticker in non_usd_tickers:
            currency = currencies[ticker]
            fx_ticker = f"{currency}USD=X"  # Exemple: EURUSD=X
            fx = yf.download(fx_ticker, start=self.start, end=self.end, progress=False, auto_adjust=True)['Close']
            fx = fx.ffill().reindex(df.index).fillna(method='ffill')  # aligner les dates
            df[ticker] = df[ticker] * fx'''

        return df

    def positions(self):
        return dict(zip(self.tickers, self.pos))

    def portfolio_values(self):
        positions = self.positions()
        return self.prices().mul(pd.Series(positions), axis=1).sum(axis=1)

    def daily_returns(self):
        prices = self.prices()
        # inclure une journée avant le début de la fenêtre pour pct_change
        if self.window and len(prices) > self.window:
            prices = prices.iloc[-(self.window + 1):]  

        portf_values = prices.mul(pd.Series(self.positions()), axis=1).sum(axis=1)
        portf_returns = portf_values.pct_change().dropna()
        returns = prices.pct_change().dropna()

        # ne garder que la longueur window exacte
        if self.window:
            portf_returns = portf_returns.iloc[-self.window:]
            returns = returns.iloc[-self.window:]

        return returns, portf_returns

    def cum_return(self):
        _, portf_returns = self.daily_returns()
        return (1 + portf_returns).cumprod() - 1


    def CAGR(self):
        portf_values = self.portfolio_values()
        if self.window:
            portf_values = portf_values[-self.window:]
        days = (portf_values.index[-1] - portf_values.index[0]).days
        years = days / 365.25
        return (portf_values.iloc[-1] / portf_values.iloc[0]) ** (1 / years) - 1

    def annual_volatility(self):
        returns = self.daily_returns()[1]
        return returns.std() * np.sqrt(252)

    def sharpe_ratio(self):
        returns = self.daily_returns()[1]
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

    def max_drawdown(self):
        portf_values = self.portfolio_values()
        if self.window:
            portf_values = portf_values[-self.window:]
        rolling_max = portf_values.cummax()
        drawdown = (portf_values - rolling_max) / rolling_max
        return drawdown.min()

    def pnl(self):
        pnl = self.portfolio_values().diff().dropna()
        if self.window:
            pnl = pnl[-self.window:]
        return pnl

    def var_historical(self, alpha=0.95):
        portf_values = self.portfolio_values()
        pnl_pct = self.portfolio_values().pct_change().dropna()  # daily returns in %
        
        if self.window:
            pnl_pct = pnl_pct[-self.window:]
        
        # VaR en pourcentage
        return -np.percentile(pnl_pct, 100 * (1 - alpha)) * 100  # multiplier par 100 pour %
        

    def cvar_historical(self, alpha=0.95):
        portf_values = self.portfolio_values()
        pnl_pct = self.portfolio_values().pct_change().dropna()  # daily returns in %
        
        if self.window:
            pnl_pct = pnl_pct[-self.window:]
        
        thresh = np.percentile(pnl_pct, 100 * (1 - alpha))
        tail = pnl_pct[pnl_pct <= thresh]
        
        # CVaR en pourcentage
        return -tail.mean() * 100  # multiplier par 100 pour %


    # ----------------- OPTIMIZATION -----------------
    def optimize_weights(self, returns=None, rf=0.0):
        if returns is None or returns.empty:
            returns = self.prices().pct_change().dropna()
            if self.window:
                returns = returns[-self.window:]
        if returns.empty:
            raise ValueError("Pas assez de données pour optimiser les poids.")

        mu = returns.mean()
        cov = returns.cov()
        n = len(mu)

        def neg_sharpe(w):
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol == 0:
                return np.inf
            return (-(port_ret - rf) / port_vol) * np.sqrt(252)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        w0 = np.ones(n) / n

        opt = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints)
        return dict(zip(self.tickers, opt.x))

    def optimize_pos(self, total_capital=None, returns=None):
        opt_w = self.optimize_weights(returns)
        if total_capital is None:
            total_capital = self.portfolio_values().iloc[-1]
        last_prices = self.prices().iloc[-1]
        weights = np.array(list(opt_w.values()))
        capital_alloc = weights * total_capital
        new_pos = (capital_alloc / last_prices).astype(int)
        return pd.Series(new_pos, index=self.tickers)

    # ----------------- STRATEGIES -----------------
    def backtest_momentum(self, short_window=5, long_window=30):
        prices = self.prices()
        index = prices.index
        current_pos = pd.Series(self.pos, index=self.tickers, dtype=float)
        portfolio_values = pd.Series(0.0, index=index)
        cash = 0.0

        for i in range(long_window):
            portfolio_values.iloc[i] = (current_pos * prices.iloc[i]).sum() + cash

        for i in range(long_window, len(prices)):
            short_ma = prices.iloc[:i+1].rolling(short_window).mean().iloc[-1]
            long_ma = prices.iloc[:i+1].rolling(long_window).mean().iloc[-1]
            signals = (short_ma > long_ma).astype(int)

            today_prices = prices.iloc[i]
            sell_signals = signals[signals == 0].index
            buy_signals = signals[signals == 1].index

            if len(sell_signals) > 0:
                cash += (current_pos[sell_signals] * today_prices[sell_signals]).sum()
                current_pos[sell_signals] = 0.0

            if len(buy_signals) > 0:
                total_value = (current_pos * today_prices).sum() + cash
                capital_per_asset = total_value / len(buy_signals)
                for ticker in buy_signals:
                    current_pos[ticker] = capital_per_asset / today_prices[ticker]
                cash = 0.0

            portfolio_values.iloc[i] = (current_pos * today_prices).sum() + cash

        return portfolio_values / portfolio_values.iloc[0]

    def backtest_constant_vol(self, target_vol=0.2, vol_lookback=10):
        prices = self.prices()
        index = prices.index
        current_pos = pd.Series(self.pos, index=self.tickers, dtype=float)
        portfolio_values = pd.Series(0.0, index=index)
        cash = 0.0

        for i in range(vol_lookback):
            portfolio_values.iloc[i] = (current_pos * prices.iloc[i]).sum() + cash

        for i in range(vol_lookback, len(prices)):
            vol = prices.iloc[i-vol_lookback:i].pct_change().std() * np.sqrt(252)
            signals = pd.Series(0, index=self.tickers, dtype=int)
            signals[vol < target_vol] = 1
            signals[vol > target_vol] = -1

            today_prices = prices.iloc[i]
            sell_signals = signals[signals == -1].index
            buy_signals = signals[signals == 1].index

            if len(sell_signals) > 0:
                cash += (current_pos[sell_signals] * today_prices[sell_signals]).sum()
                current_pos[sell_signals] = 0.0

            if len(buy_signals) > 0:
                total_value = (current_pos * today_prices).sum() + cash
                capital_per_asset = total_value / len(buy_signals)
                for ticker in buy_signals:
                    current_pos[ticker] = capital_per_asset / today_prices[ticker]
                cash = 0.0

            portfolio_values.iloc[i] = (current_pos * today_prices).sum() + cash

        return portfolio_values / portfolio_values.iloc[vol_lookback]

# ----------------- FASTAPI APP -----------------
app = FastAPI(title="Portfolio API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://portfoliomanegement-production-e103.up.railway.app",
    "https://portfolio-manegement.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # <-- ça autorise les requêtes depuis ton frontend
    allow_credentials=True,
    allow_methods=["*"],          # <-- autorise GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],          # <-- autorise tous les headers
)

# Pydantic model
class PortfolioRequest(BaseModel):
    tickers: List[str]
    positions: List[int]
    start: str
    end: Optional[str] = datetime.date.today().isoformat()
    window: Optional[int] = 252

# ---- Endpoints ----
@app.post("/metrics")
def get_metrics(req: PortfolioRequest):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=req.window)
    
    portfolio = Portfolio(start.isoformat(), end.isoformat(), req.tickers, req.positions, req.window)
    
    return {
        "CAGR": portfolio.CAGR()*100,
        "AnnualVolatility": portfolio.annual_volatility()*100,
        "SharpeRatio": portfolio.sharpe_ratio(),
        "MaxDrawdown": portfolio.max_drawdown()*100,
        "VaR_Historical": portfolio.var_historical(),
        "CVaR_Historical": portfolio.cvar_historical(),
        "portfolio_values": portfolio.portfolio_values().tolist(),
        "dates": portfolio.portfolio_values().index.strftime("%Y-%m-%d").tolist(),
        "latest_prices": portfolio.prices().iloc[-1].tolist()
    }


@app.post("/optimize_weights")
def optimize_weights(req: PortfolioRequest):
    portfolio = Portfolio(req.start, req.end, req.tickers, req.positions, req.window)
    return portfolio.optimize_weights()

@app.post("/optimize_pos")
def optimize_pos(req: PortfolioRequest, total_capital: Optional[float] = None):
    portfolio = Portfolio(req.start, req.end, req.tickers, req.positions, req.window)
    return portfolio.optimize_pos(total_capital).to_dict()

@app.post("/backtest")
def backtest(data: dict = Body(...)):
    portfolio_data = data.get("portfolio")
    momentum_params = data.get("momentumParams", {})
    const_vol_params = data.get("constVolParams", {})
    start = data.get("start")
    end = data.get("end")

    portfolio = Portfolio(
        start,
        end,
        portfolio_data['tickers'],
        portfolio_data['positions'],
        portfolio_data.get('window', 252)
    )

    momentum = portfolio.backtest_momentum(
        short_window=momentum_params.get('short_window',5),
        long_window=momentum_params.get('long_window',30)
    )
    const_vol = portfolio.backtest_constant_vol(
        target_vol=const_vol_params.get('target_vol',0.2),
        vol_lookback=const_vol_params.get('vol_lookback',10)
    )
    buy_and_hold = portfolio.portfolio_values()

    df = pd.DataFrame({
        'date': buy_and_hold.index,
        'Buy&Hold': (buy_and_hold / buy_and_hold.iloc[0]).values,
        'Momentum': momentum.values,
        'ConstVol': (const_vol / const_vol.iloc[0]).values
    })

    return {"nav": df.to_dict(orient="records")}
@app.post("/prices")
def get_prices(req: PortfolioRequest):
    portfolio = Portfolio(None, None, req.tickers, req.positions, req.window)
    
    # Définir start et end à partir de window
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=req.window)
    portfolio.start = start.strftime("%Y-%m-%d")
    portfolio.end = end.strftime("%Y-%m-%d")
    
    df = portfolio.prices()
    # transformer en liste de dicts [{date, prices:[...]}]
    data = [{"date": str(idx.date()), "prices": row.tolist()} for idx, row in df.iterrows()]
    return {"data": data}

