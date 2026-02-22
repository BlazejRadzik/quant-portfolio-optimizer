import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- 1. DEFINICJA PALETY BARW (Bordo/Czerwień/Krem) ---
FIRE_PALETTE = ["#4A0404", "#8B0000", "#B22222", "#E37222", "#E3AFBC", "#FDE2E4"]

# --- 2. FUNKCJE POMOCNICZE ---
def calculate_var(data, weights, alpha=0.05):
    portfolio_returns = (data.pct_change().dropna() * pd.Series(weights)).sum(axis=1)
    return portfolio_returns.quantile(alpha)

@st.cache_data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(url)[0]['Symbol'].tolist()

# --- 3. KONFIGURACJA STRONY ---
st.set_page_config(page_title="Pro Quant Terminal", layout="wide")

GPW = ["PKO.WA", "PKN.WA", "PZU.WA", "KGH.WA", "DNP.WA", "ALE.WA", "LPP.WA", "CDR.WA", "PEO.WA", "SPL.WA"]
try:
    SP500_DYNAMIC = get_sp500_tickers()
except:
    SP500_DYNAMIC = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

ALL_OPTIONS = sorted(list(set(GPW + SP500_DYNAMIC + ["SPY", "QQQ", "GLD"])))

# Sidebar
st.sidebar.header("🕹️ Panel Sterowania")
strategy = st.sidebar.radio("Wybierz strategię:", ["Max Sharpe Ratio", "Minimum Volatility", "Target Return"])
start_date = st.sidebar.date_input("Data początkowa", value=pd.to_datetime("2021-01-01"))
rf_rate = st.sidebar.slider("Stopa wolna od ryzyka (%)", 0.0, 10.0, 2.0) / 100

st.title("🏦 Pro Quant Asset Management Terminal")

selected_assets = st.multiselect(
    "Wybierz akcje do portfela (możesz wpisać własne tickery, np. TSLA):",
    options=ALL_OPTIONS,
    default=["PKO.WA", "AAPL", "MSFT", "CDR.WA"]
)

calculate = st.button("🚀 OBLICZ OPTYMALNY PORTFEL", use_container_width=True)

if calculate and selected_assets:
    with st.spinner('Trwa pobieranie danych...'):
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

            # --- SEKCJA WYNIKÓW ---
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.subheader("📋 Statystyki Strategii")
                st.metric("Oczekiwany Zwrot", f"{perf[0]:.2%}")
                st.metric("Zmienność (Ryzyko)", f"{perf[1]:.2%}")
                st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                st.metric("Daily VaR (95%)", f"{var_value:.2%}")
                
                # --- POPRAWKA: SORTOWANIE TABELI WAG ---
                df_w = pd.DataFrame.from_dict(w, orient='index', columns=['Waga']).query("Waga > 0")
                df_w = df_w.sort_values(by='Waga', ascending=False) # Sortowanie malejąco
                st.dataframe(df_w.style.format("{:.2%}"), use_container_width=True)

            with c2:
                st.subheader("🍩 Struktura Portfela")
                fig_pie = px.pie(
                    df_w, names=df_w.index, values='Waga', 
                    hole=0.4, 
                    template="plotly_dark",
                    color_discrete_sequence=FIRE_PALETTE
                )
                fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            # --- BACKTESTING (CZERWONA LINIA I CZYSTY HOVER) ---
            st.divider()
            st.subheader("📉 Historia wzrostu (Backtest)")
            
            port_ret = (data.pct_change().dropna() * pd.Series(w)).sum(axis=1)
            cum_ret = (1 + port_ret).cumprod()

            fig_bt = px.line(cum_ret, labels={'value': 'Kapitał', 'index': 'Oś czasu'})
            
            # Konfiguracja osi i wyglądu
            fig_bt.update_traces(
                line_color='#B22222', # Czerwona linia
                hovertemplate="Kapitał: %{y:.2f}<extra></extra>" # Tylko kapitał na hoverze
            )
            
            fig_bt.update_xaxes(
                dtick="M12", 
                tickformat="%Y", 
                rangeslider_visible=True,
                gridcolor="rgba(255, 255, 255, 0.1)"
            )
            
            fig_bt.update_layout(
                template="plotly_dark", 
                hovermode="x unified",
                xaxis_title="",
                yaxis_title="Wartość"
            )
            
            # Wyświetlenie bez paska narzędzi Plotly (aparat itp.)
            st.plotly_chart(fig_bt, use_container_width=True, config={'displayModeBar': False})

        except Exception as e:
            st.error(f"Błąd modelu: {e}")
else:
    st.info("Dodaj aktywa i kliknij przycisk powyżej.")
