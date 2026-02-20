import yfinance as yf
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Wybór aktywa
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Pobieranie danych historycznych
print("Pobieranie danych...")
data = yf.download(tickers, start="2022-01-01", end="2025-01-01")['Adj Close']

# Obliczenia 
# Średni historyczny zwrot
mu = expected_returns.mean_historical_return(data)
# Macierz kowariancji 
S = risk_models.sample_cov(data)

# Optymalizacja portfela 
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe() # Szukamy punktu o najlepszym stosunku zysku do ryzyka
cleaned_weights = ef.clean_weights()

# Wyświetlenie wyników
print("\n--- Optymalne Wagi Portfela ---")
for ticker, weight in cleaned_weights.items():
    print(f"{ticker}: {weight*100:.2f}%")

print("\n--- Statystyki Portfela ---")
ef.portfolio_performance(verbose=True)
