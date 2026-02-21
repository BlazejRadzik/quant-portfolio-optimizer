import streamlit as st
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
st.set_page_config(page_title="Quant Optimizer", layout="wide")

# Bok panelu
st.sidebar.header("⚙️ Konfiguracja Portfela")
tickers_input = st.sidebar.text_input("Wpisz tickery (rozdziel przecinkiem)", "AAPL, PKO.WA, MSFT")
start_date = st.sidebar.date_input("Data początkowa", value=pd.to_datetime("2022-01-01"))
risk_free_rate = st.sidebar.slider("Stopa wolna od ryzyka (%)", 0.0, 10.0, 2.0) / 100

# Panel
st.title("📊 Quant Portfolio Optimizer")
st.markdown(f"Optymalizacja portfela dla: **{tickers_input}**")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Wyniki Optymalizacji")
    # wywołaj funkcję optymalizującą i pobierz wagi
    # Przykład wyświetlania metryk na bazie Twoich wyników:
    st.metric("Oczekiwany zwrot", "15.7%") 
    st.metric("Sharpe Ratio", "0.66")
    st.metric("Roczna zmienność", "23.8%")

with col2:
    st.subheader("🥧 Alokacja Aktywów")
    # wykres kołowy (st.plotly_chart)
    st.info("Tutaj pojawi się wykres kołowy wag portfela.")

st.divider()
st.subheader("📉 Backtesting (Wyniki Historyczne)")
# wykres liniowy skumulowanych zwrotów
st.line_chart([1, 1.1, 1.05, 1.2, 1.3]) # Placeholder
