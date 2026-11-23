"""
app.py
--------
Stock Market Prediction and Trend Analysis Using Machine Learning and LSTM
-----------------------------------------------------------------------------
A Streamlit web application that:
    - Fetches historical stock data via yfinance
    - Calculates technical indicators (MA20, MA50, RSI, MACD)
    - Trains and compares Linear Regression, Random Forest, and LSTM models
    - Predicts the next-day closing price
    - Visualizes results with interactive Plotly charts
"""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import fetch_stock_data, validate_symbol


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch_stock_data(symbol: str, start_date, end_date):
    """Cached wrapper so re-running the same symbol/date range is instant."""
    return fetch_stock_data(symbol, start_date, end_date)
from utils.preprocessing import (
    clean_data,
    scale_close_prices,
    create_sequences,
    train_test_split_sequence,
    prepare_tabular_features,
)
from utils.indicators import calculate_all_indicators, get_trend_signal
from utils.ml_models import (
    train_linear_regression,
    train_random_forest,
    build_lstm_model,
    train_lstm_model,
)
from utils.evaluation import evaluate_predictions, format_metrics_table


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Stock Market Prediction | ML & LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom styling for a cleaner, more professional look
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 1rem;
            color: #6b7280;
            margin-top: 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        div.stButton > button {
            width: 100%;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HEADER
# =============================================================================
st.markdown('<p class="main-title">📈 Stock Market Prediction & Trend Analysis</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Using Machine Learning (Linear Regression, Random Forest) and LSTM Deep Learning</p>',
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR — USER INPUTS
# =============================================================================
st.sidebar.header("⚙️ Configuration")

# -----------------------------------------------------------------------
# Predefined company list -> ticker mapping.
# Users can pick a company by name from the dropdown, or choose
# "Other (type manually)" to enter any custom ticker symbol.
# -----------------------------------------------------------------------
COMPANY_TICKER_MAP = {
    # ---- US Companies ----
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Alphabet Inc. / Google (GOOGL)": "GOOGL",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Tesla Inc. (TSLA)": "TSLA",
    "Meta Platforms Inc. (META)": "META",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Netflix Inc. (NFLX)": "NFLX",
    "Intel Corp. (INTC)": "INTC",
    "Advanced Micro Devices (AMD)": "AMD",
    "Coca-Cola Co. (KO)": "KO",
    "Walmart Inc. (WMT)": "WMT",
    "JPMorgan Chase & Co. (JPM)": "JPM",
    # ---- Indian Companies (NSE) ----
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys Ltd. (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Wipro Ltd. (WIPRO.NS)": "WIPRO.NS",
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "ITC Ltd. (ITC.NS)": "ITC.NS",
    "Adani Enterprises (ADANIENT.NS)": "ADANIENT.NS",
    "Larsen & Toubro (LT.NS)": "LT.NS",
}

st.sidebar.subheader("🏢 Company / Stock Selection")

company_display_names = ["Other (type manually)"] + list(COMPANY_TICKER_MAP.keys())

selected_company = st.sidebar.selectbox(
    "Choose a company",
    options=company_display_names,
    index=1,  # defaults to the first real company (Apple)
    help="Pick a company from the list, or select 'Other' to enter a custom ticker.",
)

if selected_company == "Other (type manually)":
    symbol = st.sidebar.text_input(
        "Enter custom stock symbol",
        value="AAPL",
        help="Examples: AAPL, TSLA, MSFT (US) | RELIANCE.NS, TCS.NS (India, NSE)",
    ).strip().upper()
else:
    symbol = COMPANY_TICKER_MAP[selected_company]
    st.sidebar.caption(f"Selected ticker: **{symbol}**")

st.sidebar.subheader("📅 Date Range")
default_start = dt.date.today() - dt.timedelta(days=5 * 365)
default_end = dt.date.today()

start_date = st.sidebar.date_input("Start Date", value=default_start, max_value=default_end)
end_date = st.sidebar.date_input("End Date", value=default_end, max_value=default_end)

st.sidebar.subheader("🤖 Model Selection")
model_choice = st.sidebar.multiselect(
    "Select model(s) to train and compare",
    options=["Linear Regression", "Random Forest", "LSTM"],
    default=["Linear Regression", "Random Forest", "LSTM"],
)

run_button = st.sidebar.button("🚀 Train Models & Predict")

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Yahoo Finance (via yfinance)")

# -----------------------------------------------------------------------
# Fixed internal defaults (no longer exposed as sliders — kept sensible
# so the app "just works" without the user needing to tune anything).
# window_size auto-adjusts later if the selected date range is too short.
# -----------------------------------------------------------------------
DEFAULT_WINDOW_SIZE = 60
DEFAULT_LSTM_EPOCHS = 25


# =============================================================================
# HELPER: session state to persist results across reruns
# =============================================================================
if "results" not in st.session_state:
    st.session_state["results"] = None


# =============================================================================
# MAIN LOGIC — triggered when user clicks the button
# =============================================================================
if run_button:
    if start_date >= end_date:
        st.error("❌ Start date must be earlier than end date.")
    elif not model_choice:
        st.error("❌ Please select at least one model.")
    else:
        with st.spinner(f"Validating symbol '{symbol}'..."):
            is_valid = validate_symbol(symbol)

        if not is_valid:
            st.error(
                f"❌ Could not find data for symbol **'{symbol}'**. "
                "Please check the ticker. For NSE stocks, use a suffix like "
                "'RELIANCE.NS' or 'TCS.NS'."
            )
        else:
            try:
                with st.spinner("📡 Fetching historical stock data..."):
                    raw_df = _cached_fetch_stock_data(symbol, start_date, end_date)
                    df = clean_data(raw_df)

                MIN_REQUIRED_ROWS = 50  # absolute minimum to train anything meaningfully

                if len(df) < MIN_REQUIRED_ROWS:
                    st.error(
                        f"❌ Not enough historical data ({len(df)} rows) for symbol '{symbol}'. "
                        "Please widen the date range and try again."
                    )
                else:
                    # Auto-adjust the lookback window so short date ranges
                    # (e.g. 6 months) still work without the user tuning anything.
                    window_size = min(DEFAULT_WINDOW_SIZE, max(20, len(df) // 4))
                    lstm_epochs = DEFAULT_LSTM_EPOCHS

                    with st.spinner("📊 Calculating technical indicators..."):
                        df_indicators = calculate_all_indicators(df)

                    # ---------------------------------------------------
                    # Train/test data preparation
                    # ---------------------------------------------------
                    scaled_data, scaler = scale_close_prices(df)
                    X_seq, y_seq = create_sequences(scaled_data, window_size)
                    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split_sequence(X_seq, y_seq)

                    X_tab, y_tab = prepare_tabular_features(df, window_size)
                    split_idx = int(len(X_tab) * 0.8)
                    X_train_tab, X_test_tab = X_tab[:split_idx], X_tab[split_idx:]
                    y_train_tab, y_test_tab = y_tab[:split_idx], y_tab[split_idx:]

                    metrics_by_model = {}
                    predictions_by_model = {}
                    next_day_by_model = {}

                    # ---------------------------------------------------
                    # Linear Regression
                    # ---------------------------------------------------
                    if "Linear Regression" in model_choice:
                        with st.spinner("Training Linear Regression..."):
                            lr_model = train_linear_regression(X_train_tab, y_train_tab)
                            lr_pred = lr_model.predict(X_test_tab)
                            metrics_by_model["Linear Regression"] = evaluate_predictions(y_test_tab, lr_pred)
                            predictions_by_model["Linear Regression"] = lr_pred
                            next_day_by_model["Linear Regression"] = float(
                                lr_model.predict(X_tab[-1].reshape(1, -1))[0]
                            )

                    # ---------------------------------------------------
                    # Random Forest
                    # ---------------------------------------------------
                    if "Random Forest" in model_choice:
                        with st.spinner("Training Random Forest Regressor..."):
                            rf_model = train_random_forest(X_train_tab, y_train_tab)
                            rf_pred = rf_model.predict(X_test_tab)
                            metrics_by_model["Random Forest"] = evaluate_predictions(y_test_tab, rf_pred)
                            predictions_by_model["Random Forest"] = rf_pred
                            next_day_by_model["Random Forest"] = float(
                                rf_model.predict(X_tab[-1].reshape(1, -1))[0]
                            )

                    # ---------------------------------------------------
                    # LSTM
                    # ---------------------------------------------------
                    lstm_test_dates = None
                    if "LSTM" in model_choice:
                        with st.spinner("Training LSTM neural network (this may take a minute)..."):
                            lstm_model = build_lstm_model(input_shape=(X_train_seq.shape[1], 1))
                            train_lstm_model(lstm_model, X_train_seq, y_train_seq, epochs=lstm_epochs)

                            lstm_pred_scaled = lstm_model.predict(X_test_seq, verbose=0)
                            lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
                            y_test_actual = scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

                            metrics_by_model["LSTM"] = evaluate_predictions(y_test_actual, lstm_pred)
                            predictions_by_model["LSTM"] = lstm_pred

                            # Predict next day using the most recent window
                            last_window = scaled_data[-window_size:].reshape(1, window_size, 1)
                            next_scaled = lstm_model.predict(last_window, verbose=0)
                            next_day_by_model["LSTM"] = float(scaler.inverse_transform(next_scaled)[0][0])

                            lstm_test_dates = df["Date"].iloc[-len(y_test_actual):].reset_index(drop=True)

                    # Store everything needed for display
                    st.session_state["results"] = {
                        "symbol": symbol,
                        "df": df,
                        "df_indicators": df_indicators,
                        "metrics_by_model": metrics_by_model,
                        "predictions_by_model": predictions_by_model,
                        "next_day_by_model": next_day_by_model,
                        "y_test_tab": y_test_tab,
                        "test_dates_tab": df["Date"].iloc[-len(y_test_tab):].reset_index(drop=True) if len(y_test_tab) else None,
                        "lstm_test_dates": lstm_test_dates,
                        "last_close": float(df["Close"].iloc[-1]),
                        "last_date": df["Date"].iloc[-1],
                    }
                    st.success("✅ Models trained successfully! Scroll down to view results.")

            except ValueError as ve:
                st.error(f"❌ {ve}")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")


# =============================================================================
# DISPLAY RESULTS (persisted in session_state so tabs/interactions don't reset)
# =============================================================================
results = st.session_state["results"]

if results is None:
    st.info("👈 Enter a stock symbol in the sidebar and click **Train Models & Predict** to get started.")
else:
    df = results["df"]
    df_indicators = results["df_indicators"]
    symbol = results["symbol"]

    # ---------------------------------------------------------------
    # 🔮 PROMINENT NEXT-DAY PRICE — visible immediately, no tab click needed
    # ---------------------------------------------------------------
    st.markdown(
        """
        <style>
            .next-day-box {
                background: linear-gradient(135deg, #1e3a8a, #2563eb);
                padding: 1.3rem 1.6rem;
                border-radius: 14px;
                color: white;
                margin-bottom: 1.2rem;
            }
            .next-day-label {
                font-size: 0.95rem;
                opacity: 0.85;
                margin-bottom: 0.2rem;
            }
            .next-day-price {
                font-size: 2.6rem;
                font-weight: 800;
                line-height: 1.1;
            }
            .next-day-sub {
                font-size: 0.9rem;
                opacity: 0.9;
                margin-top: 0.3rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    next_day_preds = results["next_day_by_model"]
    last_close = results["last_close"]
    last_date = results["last_date"]

    # Use the average across all trained models as the headline number
    avg_next_day = sum(next_day_preds.values()) / len(next_day_preds)
    change = avg_next_day - last_close
    pct = (change / last_close) * 100
    direction = "📈" if change >= 0 else "📉"

    # Rule-based Buy/Sell/Hold signal from RSI + MACD + Moving Averages
    signal_info = get_trend_signal(results["df_indicators"])
    signal_colors = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#d97706"}
    signal_color = signal_colors[signal_info["signal"]]

    col_price, col_signal = st.columns([2.2, 1])

    with col_price:
        st.markdown(
            f"""
            <div class="next-day-box">
                <div class="next-day-label">🔮 Predicted Next-Day Closing Price — {symbol}</div>
                <div class="next-day-price">{avg_next_day:.2f} {direction}</div>
                <div class="next-day-sub">
                    Change vs last close ({last_close:.2f} on {last_date.strftime('%Y-%m-%d')}):
                    <strong>{change:+.2f} ({pct:+.2f}%)</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_signal:
        st.markdown(
            f"""
            <div class="next-day-box" style="background: {signal_color}; text-align:center;">
                <div class="next-day-label">📊 Trend Signal</div>
                <div class="next-day-price" style="font-size:2.1rem;">{signal_info['signal']}</div>
                <div class="next-day-sub">RSI: {signal_info['rsi']} | MACD vs Signal: {'▲' if signal_info['macd'] > signal_info['macd_signal'] else '▼'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Per-model breakdown, shown as compact metric cards right below the headline
    model_cols = st.columns(len(next_day_preds))
    for i, (model_name, pred_val) in enumerate(next_day_preds.items()):
        m_change = pred_val - last_close
        m_pct = (m_change / last_close) * 100
        model_cols[i].metric(
            label=model_name,
            value=f"{pred_val:.2f}",
            delta=f"{m_change:+.2f} ({m_pct:+.2f}%)",
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📄 Stock Data", "📈 Price Trends", "📊 Technical Indicators", "🤖 Predictions", "🏆 Model Comparison"]
    )

    # -------------------------------------------------------------------
    # TAB 1: Stock Data Table
    # -------------------------------------------------------------------
    with tab1:
        st.subheader(f"Historical Data — {symbol}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", f"{results['last_close']:.2f}")
        col2.metric("Total Records", f"{len(df)}")
        col3.metric("Start Date", df["Date"].min().strftime("%Y-%m-%d"))
        col4.metric("End Date", df["Date"].max().strftime("%Y-%m-%d"))

        st.dataframe(df.sort_values("Date", ascending=False), use_container_width=True, height=400)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Data as CSV", data=csv, file_name=f"{symbol}_data.csv", mime="text/csv")

    # -------------------------------------------------------------------
    # TAB 2: Price Trend Chart
    # -------------------------------------------------------------------
    with tab2:
        st.subheader("Historical Closing Price Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close Price", line=dict(color="#2563eb")))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Price",
            template="plotly_white", height=450,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Candlestick Chart")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
        )])
        fig_candle.update_layout(template="plotly_white", height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 3: Technical Indicators
    # -------------------------------------------------------------------
    with tab3:
        st.subheader("Moving Averages (20-day & 50-day)")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=df_indicators["Date"], y=df_indicators["Close"], name="Close", line=dict(color="#111827")))
        fig_ma.add_trace(go.Scatter(x=df_indicators["Date"], y=df_indicators["MA20"], name="MA 20", line=dict(color="#f59e0b")))
        fig_ma.add_trace(go.Scatter(x=df_indicators["Date"], y=df_indicators["MA50"], name="MA 50", line=dict(color="#dc2626")))
        fig_ma.update_layout(template="plotly_white", height=420, hovermode="x unified")
        st.plotly_chart(fig_ma, use_container_width=True)

        col_rsi, col_macd = st.columns(2)

        with col_rsi:
            st.subheader("RSI (Relative Strength Index)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_indicators["Date"], y=df_indicators["RSI"], name="RSI", line=dict(color="#7c3aed")))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(template="plotly_white", height=380, yaxis_range=[0, 100])
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col_macd:
            st.subheader("MACD")
            fig_macd = make_subplots(specs=[[{"secondary_y": False}]])
            fig_macd.add_trace(go.Scatter(x=df_indicators["Date"], y=df_indicators["MACD"], name="MACD", line=dict(color="#2563eb")))
            fig_macd.add_trace(go.Scatter(x=df_indicators["Date"], y=df_indicators["MACD_Signal"], name="Signal", line=dict(color="#dc2626")))
            fig_macd.add_trace(go.Bar(x=df_indicators["Date"], y=df_indicators["MACD_Histogram"], name="Histogram", marker_color="#9ca3af"))
            fig_macd.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_macd, use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 4: Predictions
    # -------------------------------------------------------------------
    with tab4:
        st.subheader("Actual vs Predicted Closing Prices")

        for model_name, preds in results["predictions_by_model"].items():
            st.markdown(f"**{model_name}**")
            fig_pred = go.Figure()

            if model_name == "LSTM" and results["lstm_test_dates"] is not None:
                dates = results["lstm_test_dates"]
                actual = df.set_index("Date").loc[dates, "Close"].values if False else None
                # Use scaler-inverse actuals already aligned by index length
                y_actual_plot = df["Close"].iloc[-len(preds):].values
            else:
                dates = results["test_dates_tab"]
                y_actual_plot = results["y_test_tab"]

            fig_pred.add_trace(go.Scatter(x=dates, y=y_actual_plot, mode="lines", name="Actual", line=dict(color="#111827")))
            fig_pred.add_trace(go.Scatter(x=dates, y=preds, mode="lines", name="Predicted", line=dict(color="#dc2626", dash="dot")))
            fig_pred.update_layout(template="plotly_white", height=380, hovermode="x unified")
            st.plotly_chart(fig_pred, use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 5: Model Comparison
    # -------------------------------------------------------------------
    with tab5:
        st.subheader("Model Performance Comparison")
        metrics_df = format_metrics_table(results["metrics_by_model"])
        st.dataframe(metrics_df, use_container_width=True)

        metric_to_plot = st.selectbox("Compare models by metric:", ["MAE", "MSE", "RMSE", "R2"])
        fig_bar = go.Figure(data=[
            go.Bar(x=metrics_df["Model"], y=metrics_df[metric_to_plot], marker_color="#2563eb")
        ])
        fig_bar.update_layout(template="plotly_white", height=420, yaxis_title=metric_to_plot)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(
            "Lower MAE / MSE / RMSE indicate better accuracy. "
            "Higher R² (closer to 1.0) indicates the model explains more of the price variance."
        )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Built with Streamlit, scikit-learn, and TensorFlow/Keras.")
