import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
import plotly.express as px

# --- 1. FUNKCJE POMOCNICZE (Muszą być na początku, aby uniknąć błędów) ---

@st.cache_data
def get_sp500_tickers():
    """Dynamicznie pobiera listę spółek S&P 500 z Wikipedii."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    return table[0]['Symbol'].tolist()

def calculate_var(data, weights, alpha=0.05):
    """Oblicza Historyczny Value at Risk (VaR) na poziomie 95%."""
    # zwroty dzienne
    portfolio_returns = (data.pct_change().dropna() * pd.Series(weights)).sum(axis=1)
    # percentyle 95 najgorszych
    var_95 = portfolio_returns.quantile(alpha)
    return var_95

# --- 2. KONFIGURACJA STRONY ---

st.set_page_config(page_title="Pro Quant Terminal", layout="wide")

# aktywa
GPW = ["PKO.WA", "PKN.WA", "PZU.WA", "KGH.WA", "DNP.WA", "ALE.WA", "LPP.WA", "CDR.WA", "PEO.WA", "SPL.WA"]
try:
    SP500_DYNAMIC = get_sp500_tickers()
except:
    SP500_DYNAMIC = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"] # Fallback

ETFS = ["SPY", "QQQ", "GLD", "VGT", "EEM"]

# Łączymy listy w jedną bazę do wyszukiwania
ALL_OPTIONS = sorted(list(set(GPW + SP500_DYNAMIC + ETFS)))

# bok
st.sidebar.header("🕹️ Panel Sterowania")

# cel liczenia
st.sidebar.subheader("Cel Inwestycyjny")
strategy = st.sidebar.radio(
    "Wybierz strategię:",
    ["Max Sharpe Ratio", "Minimum Volatility", "Target Return"]
)

target_return = 0
if strategy == "Target Return":
    target_return = st.sidebar.slider("Oczekiwany zwrot (%)", 5, 50, 15) / 100

# parametry techniczne
st.sidebar.divider()
start_date = st.sidebar.date_input("Data początkowa", value=pd.to_datetime("2021-01-01"))
risk_free_rate = st.sidebar.slider("Stopa wolna od ryzyka (%)", 0.0, 10.0, 2.0) / 100

# Panel główny
st.title("🏦 Pro Quant Asset Management Terminal")

# Wybór akcji - Interaktywna lista na górze
selected_assets = st.multiselect(
    "Wybierz akcje do portfela (możesz wpisać własne tickery, np. TSLA):",
    options=ALL_OPTIONS,
    default=["PKO.WA", "AAPL", "MSFT", "CDR.WA"]
)

# przycisk startu
calculate = st.button("🚀 OBLICZ OPTYMALNY PORTFEL", use_container_width=True)

if calculate and selected_assets:
    with st.spinner('Pobieranie danych rynkowych...'):
        try:
            # pobieranie danych
            data = yf.download(selected_assets, start=start_date)['Close']
            
            # obliczenia
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, S)
            
            # wybor strategii
            if strategy == "Max Sharpe Ratio":
                weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif strategy == "Minimum Volatility":
                weights = ef.min_volatility()
            else:
                weights = ef.efficient_return(target_return=target_return)
            
            cleaned_weights = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            
            # --- DODANE: Obliczenie VaR (Tylko gdy mamy dane) ---
            var_value = calculate_var(data, cleaned_weights)

            # wizualizacja 
            c1, c2, c3 = st.columns([1, 1, 2]) # Dodano kolumnę c3 dla lepszego rozkładu
            
            with c1:
                st.subheader("📋 Statystyki")
                st.metric("Oczekiwany Zwrot", f"{perf[0]*100:.2f}%")
                st.metric("Zmienność (Ryzyko)", f"{perf[1]*100:.2f}%")
                
            with c2:
                st.subheader("🛡️ Risk Management")
                st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                st.metric("Daily VaR (95%)", f"{var_value:.2%}", help="Maksymalna oczekiwana strata dzienna z 95% prawdopodobieństwem.")
                
            with c3:
                st.subheader("🍩 Struktura Portfela")
                df_weights = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Waga'])
                df_weights = df_weights[df_weights['Waga'] > 0]
                
                fig = px.pie(
                    names=df_weights.index, 
                    values=df_weights['Waga'],
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            # backtesting
            st.divider()
            st.subheader("📉 Historia wzrostu (Backtest)")
            returns = data.pct_change().dropna()
            port_ret = (returns * pd.Series(cleaned_weights)).sum(axis=1)
            cum_ret = (1 + port_ret).cumprod()
            st.line_chart(cum_ret)

        except Exception as e:
            st.error(f"Błąd modelu: {e}. Spróbuj dodać więcej danych lub zmienić datę.")
else:
    st.info("Dodaj aktywa i kliknij przycisk powyżej, aby zobaczyć analizę.")
