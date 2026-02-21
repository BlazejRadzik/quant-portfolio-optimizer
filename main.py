import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns

st.set_page_config(page_title="Quant Terminal", layout="wide")

class PortfolioEngine:
    """Klasa obsługująca logikę obliczeniową portfela."""
    @staticmethod
    @st.cache_data
    def get_market_data(tickers, start):
        return yf.download(tickers, start=start)['Close']

    @staticmethod
    def optimize(data, rf_rate, strategy="Max Sharpe"):
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        
        if strategy == "Max Sharpe":
            weights = ef.max_sharpe(risk_free_rate=rf_rate)
        else:
            weights = ef.min_volatility()
            
        return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=rf_rate)

# UI INTERFACE
st.title("🏦 Pro Quant Asset Management Terminal")

with st.sidebar:
    st.header("Settings")
    assets = st.multiselect("Assets", ["PKO.WA", "AAPL", "MSFT", "KGH.WA", "TSLA", "SPY"], default=["AAPL", "PKO.WA"])
    strategy = st.selectbox("Strategy", ["Max Sharpe", "Min Volatility"])
    rf = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0) / 100
    run_btn = st.button("Compute Optimization", use_container_width=True)

if run_btn and assets:
    data = PortfolioEngine.get_market_data(assets, "2021-01-01")
    w, perf = PortfolioEngine.optimize(data, rf, strategy)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Expected Return", f"{perf[0]:.2%}")
        st.metric("Volatility", f"{perf[1]:.2%}")
        st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
    
    with c2:
        df_w = pd.DataFrame.from_dict(w, orient='index', columns=['Weight']).query("Weight > 0")
        fig = px.pie(df_w, values='Weight', names=df_w.index, hole=0.4, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
