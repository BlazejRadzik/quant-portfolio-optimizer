import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# 1. Konfiguracja strony musi być na samym początku
st.set_page_config(page_title="Quant Optimizer", layout="wide")

# --- SIDEBAR (PANEL BOCZNY) ---
st.sidebar.header("⚙️ Konfiguracja Portfela")
# Zamieniamy input na listę
tickers_raw = st.sidebar.text_input("Wpisz tickery (rozdziel przecinkiem)", "AAPL, MSFT, GOOGL, PKO.WA, KGH.WA")
tickers = [t.strip().upper() for t in tickers_raw.split(",")]

start_date = st.sidebar.date_input("Data początkowa", value=pd.to_datetime("2022-01-01"))
risk_free_rate = st.sidebar.slider("Stopa wolna od ryzyka (%)", 0.0, 10.0, 2.0) / 100

# --- LOGIKA OBLICZENIOWA ---
@st.cache_data # Cache, żeby nie pobierać danych przy każdej zmianie suwaka
def get_data(tickers, start):
    data = yf.download(tickers, start=start)['Close']
    return data

try:
    data = get_data(tickers, start_date)
    
    # Obliczenia optymalizacji
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    
    # Używamy stopy wolnej od ryzyka z suwaka
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)

    # --- PANEL GŁÓWNY ---
    st.title("📊 Quant Portfolio Optimizer")
    st.markdown(f"Analiza dla: **{', '.join(tickers)}**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📈 Wyniki Optymalizacji")
        st.metric("Oczekiwany roczny zwrot", f"{perf[0]*100:.2f}%")
        st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
        st.metric("Roczna zmienność (Ryzyko)", f"{perf[1]*100:.2f}%")

    with col2:
        st.subheader("🥧 Alokacja Aktywów")
        labels = [t for t, w in cleaned_weights.items() if w > 0]
        sizes = [w for t, w in cleaned_weights.items() if w > 0]
        
        if sizes:
            fig, ax = plt.subplots(figsize=(6, 4))
            # Ustawienie ciemnego tła wykresu, by pasowało do stylu "Quant"
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'color':"w"})
            plt.setp(autotexts, size=8, weight="bold")
            st.pyplot(fig)
        else:
            st.error("Model nie mógł wyznaczyć wag. Spróbuj zmienić datę lub tickery.")

    st.divider()

    # --- KROK PO KROKU: BACKTESTING ---
    st.subheader("📉 Backtesting (Skumulowana Stopa Zwrotu)")
    
    # 1. Obliczamy dzienne stopy zwrotu akcji
    returns = data.pct_change()
    
    # 2. Mnożymy zwroty przez wyliczone wagi portfela
    portfolio_weights = pd.Series(cleaned_weights)
    weighted_returns = returns.mul(portfolio_weights, axis=1).sum(axis=1)
    
    # 3. Obliczamy skumulowany zwrot (kapitał początkowy = 1.0)
    cumulative_returns = (1 + weighted_returns).cumprod()
    
    # Wyświetlamy wykres interaktywny Streamlit
    st.line_chart(cumulative_returns)
    
    st.caption("Wykres pokazuje, jak zmieniałaby się wartość 1 PLN zainwestowanego w ten portfel w wybranym okresie.")

except Exception as e:
    st.error(f"Wystąpił błąd podczas pobierania danych: {e}")
    st.info("Upewnij się, że wpisane tickery są poprawne (np. AAPL dla Apple, PKO.WA dla PKO BP).")
