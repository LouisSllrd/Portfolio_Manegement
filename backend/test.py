import yfinance as yf
from datetime import datetime, timedelta

ticker_symbol = "AAPL"

try:
    # Calculer la date d'il y a une semaine
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    # Télécharger les données pour cette date précise
    data = yf.download(
        ticker_symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=(start_date + timedelta(days=1)).strftime("%Y-%m-%d"),  # Fin = lendemain pour s'assurer d'avoir la date
        progress=False,
        auto_adjust=True
    )

    if data.empty:
        raise ValueError(f"Aucune donnée disponible pour {ticker_symbol} à la date {start_date}.")

    # Extraire le prix de clôture d'il y a une semaine
    last_week_close_price = data["Close"].iloc[0]  # Premier (et seul) élément
    last_week_close_date = data.index[0].strftime("%Y-%m-%d")

    print(f"Ticker: {ticker_symbol}")
    print(f"Prix de clôture d'il y a une semaine ({last_week_close_date}): {last_week_close_price}")

except Exception as e:
    print(f"Erreur: {e}")
    print("Vérifiez le ticker ou la connexion à Yahoo Finance.")
