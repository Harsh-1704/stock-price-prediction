# 📈 Stock Market Prediction and Trend Analysis Using Machine Learning and LSTM

An end-to-end, educational stock market prediction system built with **Python**,
**scikit-learn**, **TensorFlow/Keras (LSTM)**, and an interactive **Streamlit**
web dashboard.

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**.
> It is **not** financial advice, and predictions should never be used as the sole
> basis for real investment decisions. Stock markets are influenced by countless
> unpredictable factors that no model can fully capture.

---

## ✨ Features

- Fetches live historical stock data via the **yfinance** API (any global ticker,
  e.g. `AAPL`, `TSLA`, `RELIANCE.NS`, `TCS.NS`)
- **Company dropdown selector** — pick from 25+ popular US & Indian (NSE) companies
  by name, or choose "Other (type manually)" to enter any custom ticker
- Cleans and preprocesses data (missing values, sorting, normalization)
- Builds 60-day sliding-window sequences for time-series learning
- Trains and compares **three models**:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - LSTM Deep Learning model (2 LSTM layers + Dropout)
- Predicts the **next-day closing price**
- Evaluates models using **MAE, MSE, RMSE, and R² Score**
- Calculates technical indicators: **MA20, MA50, RSI, MACD**
- Interactive Plotly visualizations: price trends, candlesticks, actual vs.
  predicted, indicator charts, and model comparison bar charts
- Clean, professional, tabbed **Streamlit UI**

---

## 🗂️ Project Structure

```
stock_market_prediction/
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # (optional) place to cache downloaded data
├── models/                    # (optional) place to save trained models
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # yfinance data fetching & validation
│   ├── preprocessing.py       # cleaning, scaling, sequence building
│   ├── indicators.py          # MA20, MA50, RSI, MACD calculations
│   ├── ml_models.py           # Linear Regression, Random Forest, LSTM
│   └── evaluation.py          # MAE, MSE, RMSE, R² metrics
│
└── notebooks/
    └── stock_analysis.ipynb   # Exploratory analysis / experimentation notebook
```

---

## 🚀 Installation & Setup

### 1. Clone or download the project
Place the `stock_market_prediction/` folder anywhere on your machine.

### 2. Create a virtual environment (recommended)

```bash
cd stock_market_prediction
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow can take a few minutes to install and may require a
> 64-bit Python version (3.9–3.11 recommended for best compatibility).

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

This will open the app automatically in your browser at `http://localhost:8501`.

---

## 🖥️ How to Use the App

1. **Choose a company** from the sidebar dropdown (e.g. Apple, Tesla, Reliance,
   TCS), or select **"Other (type manually)"** to type any custom ticker symbol.
2. **Choose a date range** for historical data.
3. **Select which model(s)** to train: Linear Regression, Random Forest, and/or LSTM.
4. Click **🚀 Train Models & Predict** — the lookback window and LSTM epochs are
   picked automatically (no tuning needed), and shrink safely for short date ranges.
5. See the **predicted next-day price** and a rule-based **BUY / SELL / HOLD
   trend signal** (from RSI + MACD + Moving Averages) immediately at the top.
6. Explore the tabs:
   - **Stock Data** – raw historical table + CSV download
   - **Price Trends** – line chart & candlestick chart
   - **Technical Indicators** – MA20/MA50, RSI, MACD
   - **Predictions** – actual vs. predicted charts + next-day price forecast
   - **Model Comparison** – MAE / MSE / RMSE / R² across models

---

## 🧠 Model Details

| Model              | Type                  | Notes                                      |
|--------------------|-----------------------|---------------------------------------------|
| Linear Regression  | Classic ML baseline   | Uses last N days' closes as linear features |
| Random Forest      | Ensemble ML           | 200 trees, captures non-linear patterns      |
| LSTM               | Deep Learning (RNN)   | 2 LSTM layers + Dropout, Adam optimizer, MSE loss |

**LSTM Architecture:**
```
Input (60, 1)
  → LSTM(50, return_sequences=True)
  → Dropout(0.2)
  → LSTM(50)
  → Dropout(0.2)
  → Dense(25, relu)
  → Dense(1)
```

---

## ⚠️ Error Handling

The app gracefully handles:
- Invalid or unavailable stock symbols
- Insufficient historical data for the chosen lookback window
- Network/API errors from yfinance
- Empty or malformed data returned by the API

---

## 📌 Important Notes

- **Indian stocks**: Add `.NS` (NSE) or `.BO` (BSE) suffix, e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`.
- **US stocks**: Use the plain ticker, e.g. `AAPL`, `TSLA`, `MSFT`.
- LSTM training time depends on the amount of data and epochs selected — larger
  date ranges and higher epoch counts will take longer.
- Results are **not deterministic** across runs for LSTM due to random weight
  initialization, unless a fixed random seed is set.

---

## 📜 License & Disclaimer

This project is provided **as-is**, for educational purposes only. It does not
constitute financial, investment, or trading advice of any kind. The authors
are not responsible for any financial decisions made using this tool. Always
consult a qualified financial advisor before investing.
