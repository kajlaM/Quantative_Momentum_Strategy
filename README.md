# 📈 Quantitative Momentum Strategy using Machine Learning

This project implements and backtests a machine learning-driven directional momentum strategy on India's benchmark indices: **NSE Nifty 50** (`^NSEI`) and **BSE Sensex** (`^BSESN`) from **January 2010 to Present**.

The primary objective is to verify if machine learning models can accurately forecast short-term (5-day) index direction to serve as a **regime filter**—allowing investors/traders to remain long in positive regimes and move to cash during high-risk regimes, thereby mitigating market crashes (like the 2020 COVID-19 crash).

---

## 🚀 Key Findings

* **Downside Protection**: While a buy-and-hold index strategy generates strong absolute returns during structural bull markets, it suffers from heavy drawdowns. Overlaying the ML strategy threshold filters out high-risk periods, **slashing the maximum drawdown from ~38% to just ~10%**.
* **Risk-Adjusted Performance**: By avoiding major market crashes, the ML regime filter improves the Sharpe ratio and stability of the portfolio returns during regime shifts.

---

## 📂 Project Structure

```
Quantative_Momuntum_strategy/
├── data/
│   ├── nifty50.csv             # Cached historical data for Nifty 50 (^NSEI)
│   └── sensex.csv              # Cached historical data for Sensex (^BSESN)
├── Quantitative_Momentum_Strategy.ipynb   # Main Jupyter Notebook
├── requirements.txt            # Python library dependencies
└── .gitignore                  # Git ignore file for checkpoints and data
```

---

## ⚙️ Methodology & Strategy Workflow

### 1. Data Ingestion & Cache
Historical daily price data is retrieved starting from **2010-01-01** using the `yfinance` API. The ingestion pipeline caches the CSV data locally in `data/` to facilitate faster execution and offline development.

### 2. Feature Engineering
We construct **16 predictive technical indicators** grouped across four main dimensions:
1. **Momentum (Returns)**: Multi-horizon rolling returns over $5, 10, 21, 63, 126,$ and $252$ trading days.
2. **Trend (Moving Averages)**: Distance of the close price from the 20, 50, and 200 SMA, and the golden cross ratio (50 SMA / 200 SMA).
3. **Oscillators**: Relative Strength Index (RSI-14) and Bollinger Bands distance.
4. **Volatility & Volume**: Normalized Average True Range (ATR) and Volume Moving Average ratios.

### 3. Target Definition & Walk-Forward Validation
* **Target Variable**: A binary classifier $Y_t \in \{0, 1\}$ predicting whether the index close price $5$ trading days in the future is higher than today's close:
  $$Y_t = \begin{cases} 1 & \text{if } Close_{t+5} > Close_t \\ 0 & \text{otherwise} \end{cases}$$
* **Validation Scheme**: Standard cross-validation causes lookahead bias in time-series data. Instead, we use a **Walk-Forward (Rolling) Validation** split:
  * Out-of-sample testing begins in **2019** and goes up to **2026**.
  * The models are retrained annually, incorporating all historical data up to that year (producing 8 distinct testing folds).

### 4. Machine Learning Models
We compare three distinct classification models:
* **Logistic Regression** (L2 Regularization baseline)
* **Random Forest Classifier**
* **XGBoost Classifier**

### 5. Backtesting & Metrics
We run a daily simulation matching predicted probabilities against a probability threshold ($P_{\text{threshold}} = 0.50$):
* **Long Position**: When predicted probability of an upward move is $\ge P_{\text{threshold}}$.
* **Cash Position**: When predicted probability of an upward move is $< P_{\text{threshold}}$ (earning 0% return, minus minor transaction costs).
* We compute performance metrics: Annualized Return, Sharpe Ratio, and Maximum Drawdown.

---

## 🛠️ Setup & Execution

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook
Start the Jupyter Notebook environment and open the workspace file:
```bash
jupyter notebook Quantitative_Momentum_Strategy.ipynb
```
Run all cells to download/ingest the data, train the models, and view the diagnostic backtest curves and confusion matrices.
