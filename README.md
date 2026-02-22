# 🏦 Institutional Quant Terminal

An interactive analytical tool for constructing and evaluating investment portfolios based on **Modern Portfolio Theory** and advanced risk metrics.

[👉 **Live Application Link**](https://quant-portfolio-optimizer-gztfcpxfanog22mgvuyxz6.streamlit.app/)

## 🛠 Core Modules
* **Dynamic Asset Selection:** Real-time fetching of **S&P 500** components, **WIG20** stocks, and global **ETFs** via Yahoo Finance API.
* **Optimization Engine:** Full implementation of Modern Portfolio Theory (MPT) using `PyPortfolioOpt` to solve convex optimization problems.
* **Risk Analytics:** Integration of a historical **Value at Risk (VaR)** model at a 95% confidence level.
* **Advanced Backtesting:** Historical performance simulation with yearly data segregation and interactive time-series analysis.

## 📈 Quantitative Methodology

### 1. Portfolio Optimization
The system calculates optimal asset weights ($\omega$) across three primary objective functions:
* **Max Sharpe Ratio:** Maximizing the risk-adjusted return.
* **Minimum Volatility:** Constructing the portfolio with the lowest possible variance.
* **Target Return:** Optimizing for a specific required rate of return.

$$SR = \frac{R_p - R_f}{\sigma_p}$$

### 2. Value at Risk (VaR)
To quantify downside risk, the terminal estimates the **Daily VaR (95%)**. This metric represents the maximum expected loss over a one-day horizon that will not be exceeded with 95% certainty.



### 3. Visualization Standards
* **Portfolio Structure:** Custom donut chart using a sequential **Deep Red & Cream** palette for clear weight distribution.
* **Sorted Holdings:** Asset weights are automatically sorted in **descending order** in the data table for immediate exposure analysis.
* **Backtest Engine:** High-fidelity Plotly charts with a **red trendline**, annual x-axis ticks, and an integrated range slider for granular analysis.

## 💻 Technical Stack
* **Language:** Python 3.x
* **Libraries:** `PyPortfolioOpt`, `Pandas`, `NumPy`
* **Data Sourcing:** `yfinance`
* **UI/Deployment:** `Streamlit` & `Streamlit Cloud`
* **Visuals:** `Plotly Express`
