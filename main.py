import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Wybór aktywa
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "PKO.WA", "KGH.WA", "ALE.WA", "PKN.WA", "DNP.WA"]

# Pobieranie danych historycznych
print("Pobieranie danych...")
data = yf.download(tickers, start="2022-01-01", end="2025-01-01", auto_adjust=True)['Close']

# Obliczenia 
# Średni historyczny zwrot
mu = expected_returns.mean_historical_return(data)
# Macierz kowariancji 
S = risk_models.sample_cov(data)

# Optymalizacja portfela 
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe() # Szukamy punktu o najlepszym stosunku zysku do ryzyka
cleaned_weights = ef.clean_weights()

# Wyświetlenie wyników w terminalu
print("\n--- Optymalne Wagi Portfela ---")
for ticker, weight in cleaned_weights.items():
    print(f"{ticker}: {weight*100:.2f}%")

print("\n--- Statystyki Portfela ---")
ef.portfolio_performance(verbose=True)

# Wykres kołowy przy czym filtrujemy tylko spółki które mają udział większy niż 0%
labels = [ticker for ticker, weight in cleaned_weights.items() if weight > 0]
sizes = [weight for ticker, weight in cleaned_weights.items() if weight > 0]

if sizes:
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
    plt.title('Optymalna Alokacja Portfela (Max Sharpe Ratio)')
    plt.savefig('portfolio_plot.png')
    print("\n[SUKCES] Wykres został zapisany jako portfolio_plot.png")
else:
    print("\n[BŁĄD] Brak danych do wygenerowania wykresu.")
