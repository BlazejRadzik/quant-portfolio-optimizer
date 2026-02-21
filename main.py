import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
import plotly.express as px

# --- 1. FUNKCJE POMOCNICZE ---
@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    return table[0]['Symbol'].tolist()

def calculate_var(data, weights, alpha=0.05):
    portfolio_returns = (data.pct_change().dropna() * pd.Series(weights)).sum(axis=1)
    return portfolio_returns.quantile(alpha)

# --- 2. KONFIGURACJA ---
st.set_page_config(page_title="Pro Quant Terminal", layout="wide")

GPW = ["PKO.WA", "PKN.WA", "PZU.WA", "KGH.WA", "DNP.WA", "ALE.WA", "LPP.WA", "CDR.WA", "PEO.WA", "SPL.WA"]
try:
    SP500_DYNAMIC = get_sp500_tickers()
except:
    SP500_DYNAMIC = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

ALL_OPTIONS = sorted(list(set(GPW + SP500_DYNAMIC + ["SPY", "QQQ", "GLD"])))

# Sidebar
st.sidebar.header("🕹️ Panel Sterowania")
strategy = st.sidebar.radio("Strategia:", ["Max Sharpe Ratio", "Minimum Volatility", "Target Return"])
start_date = st.sidebar.date_input("Data początkowa", value=pd.to_datetime("2021-01-01"))
rf_rate = st.sidebar.slider("Stopa wolna od ryzyka (%)", 0.0, 10.0, 2.0) / 100

# Panel główny
st.title("🏦 Pro Quant Asset Management Terminal")

selected_assets = st.multiselect(
    "Wybierz akcje:",
    options=ALL_OPTIONS,
    default=["PKO.WA", "AAPL", "MSFT", "CDR.WA"]
)

if st.button("🚀 OBLICZ OPTYMALNY PORTFEL", use_container_width=True):
    with st.spinner('Trwa analiza...'):
        try:
            data = yf.download(selected_assets, start=start_date)['Close']
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            ef = EfficientFrontier(mu, S)
            
            if strategy == "Max Sharpe Ratio":
                weights = ef.max_sharpe(risk_free_rate=rf_rate)
            elif strategy == "Minimum Volatility":
                weights = ef.min_volatility()
            else:
                weights = ef.efficient_return(target_return=0.15)
            
            w = ef.clean_weights()
            perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf_rate)
            
            # OBLICZENIE VAR
            var_value = calculate_var(data, w)

            # --- KLUCZOWA ZMIANA: TRZY KOLUMNY DLA WIDOCZNOŚCI ---
            c1, c2, c3 = st.columns(3) 
            
            with c1:
                st.subheader("📈 Performance")
                st.metric("Oczekiwany Zwrot", f"{perf[0]:.2%}")
                st.metric("Zmienność", f"{perf[1]:.2%}")
                
            with c2:
                st.subheader("🛡️ Risk Management") # TA SEKCJA BYŁA NIEWIDOCZNA
                st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                st.metric("Daily VaR (95%)", f"{var_value:.2%}", help="Max strata dzienna (95% ufności)")
                
            with c3:
                st.subheader("🍩 Struktura")
                df_w = pd.DataFrame.from_dict(w, orient='index', columns=['W']).query("W > 0")
                fig = px.pie(df_w, names=df_w.index, values='W', hole=0.4, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("📉 Historia wzrostu (Backtest)")
            cum_ret = (1 + (data.pct_change().dropna() * pd.Series(w)).sum(axis=1)).cumprod()
            st.line_chart(cum_ret)

        except Exception as e:
            st.error(f"Błąd: {e}")
