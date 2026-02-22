import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- 1. FUNKCJE POMOCNICZE ---
@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url)[0]['Symbol'].tolist()

def calculate_var(data, weights, alpha=0.05):
    portfolio_returns = (data.pct_change().dropna() * pd.Series(weights)).sum(axis=1)
    return portfolio_returns.quantile(alpha)

# --- 2. KONFIGURACJA I KOLORYSTYKA ---
st.set_page_config(page_title="Pro Quant Terminal", layout="wide")

# Definicja palety: Biel -> Bursztyn -> Pomarańcz -> Czerwień -> Brąz
QUANT_COLORS = [
    "#FFFFFF", "#FFBF00", "#FF8C00", "#FF4500", 
    "#FF0000", "#8B0000", "#4B2C20"
]

# --- 3. LOGIKA AKTYWÓW ---
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
    with st.spinner('Trwa analiza rynkowa...'):
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
            var_value = calculate_var(data, w)

            # --- SEKCJA WYNIKÓW Z NOWĄ KOLORYSTYKĄ ---
            c1, c2, c3 = st.columns(3) 
            
            with c1:
                # Czerwony nagłówek Performance
                st.markdown("<h3 style='color: #FF4B4B;'>📈 Performance</h3>", unsafe_allow_html=True)
                st.metric("Oczekiwany Zwrot", f"{perf[0]:.2%}")
                st.metric("Zmienność", f"{perf[1]:.2%}")
                
            with c2:
                # Czerwony nagłówek Risk Management
                st.markdown("<h3 style='color: #FF4B4B;'>🛡️ Risk Management</h3>", unsafe_allow_html=True)
                st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                st.metric("Daily VaR (95%)", f"{var_value:.2%}")
                
            with c3:
                st.markdown("<h3>🍩 Struktura</h3>", unsafe_allow_html=True)
                df_w = pd.DataFrame.from_dict(w, orient='index', columns=['W']).query("W > 0")
                
                # Wykres kołowy z paletą ognia/bursztynu
                fig = px.pie(
                    df_w, names=df_w.index, values='W', 
                    hole=0.4, 
                    template="plotly_dark",
                    color_discrete_sequence=QUANT_COLORS
                )
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            # --- BACKTESTING Z PODZIAŁEM NA LATA ---
            st.divider()
            st.markdown("<h3 style='color: #FF4B4B;'>📉 Historia wzrostu (Backtest)</h3>", unsafe_allow_html=True)
            
            port_ret = (data.pct_change().dropna() * pd.Series(w)).sum(axis=1)
            cum_ret = (1 + port_ret).cumprod()

            fig_backtest = px.line(cum_ret, labels={'value': 'Kapitał', 'index': 'Oś czasu'})
            fig_backtest.update_xaxes(
                dtick="M12", 
                tickformat="%Y", 
                rangeslider_visible=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            )
            fig_backtest.update_traces(line_color='#FF4B4B') # Czerwona linia trendu
            fig_backtest.update_layout(template="plotly_dark", hovermode="x unified")
            
            st.plotly_chart(fig_backtest, use_container_width=True)

        except Exception as e:
            st.error(f"Błąd: {e}")
