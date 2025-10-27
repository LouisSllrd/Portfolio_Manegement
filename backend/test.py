import yfinance as yf
from datetime import datetime

ticker_symbol = "TSLA"

try:
    # 1. Récupérer le prix actuel avec fast_info
    ticker = yf.Ticker(ticker_symbol)
    fast_info = ticker.fast_info
    current_price_fast_info = fast_info.last_price
    currency = fast_info.currency

    # 2. Récupérer le dernier prix de clôture avec history
    data = ticker.history(period="1d")  # Dernier jour disponible
    if data.empty:
        raise ValueError("Aucune donnée historique disponible.")

    last_close_price = data["Close"].iloc[-1]  # Dernier prix de clôture

    # 3. Afficher les résultats
    print(f"Ticker: {ticker_symbol}")
    print(f"Prix actuel (fast_info): {current_price_fast_info} {currency}")
    print(f"Dernier prix de clôture (history): {last_close_price} {currency}")
    print(f"Date de la dernière clôture: {data.index[-1].strftime('%Y-%m-%d')}")

    # 4. Vérifier si les prix sont identiques
    if abs(current_price_fast_info - last_close_price) < 0.01:
        print("✅ Les prix sont identiques.")
    else:
        print("❌ Les prix sont différents.")
        print(f"Différence: {abs(current_price_fast_info - last_close_price)} {currency}")

except Exception as e:
    print(f"Erreur: {e}")
    print("Vérifiez le ticker ou la connexion à Yahoo Finance.")
