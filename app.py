import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import datetime


from src.data_fetcher import fetch_stock_data
from src.indicators import add_indicators
from src.signals import generate_signals
from src.predictors import run_ml_pipeline, run_classification_pipeline
from src.decision import get_final_decision
from streamlit_autorefresh import st_autorefresh
from src.predictors import run_arima_model

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Stock AI Dashboard",
    page_icon="📈",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(145deg, #1c1f26, #111318);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.buy { color: #00ff9f; font-weight: bold; }
.sell { color: #ff4b4b; font-weight: bold; }
.hold { color: #f1c40f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Stock Market Dashboard")

# ================= SIDEBAR =================
st.sidebar.title("Settings")

stocks = [
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS",
    "ITC.NS","HINDUNILVR.NS",
    "RELIANCE.NS","ONGC.NS",
    "TATAMOTORS.NS","MARUTI.NS","BAJAJ-AUTO.NS",
    "ADANIENT.NS","ADANIPORTS.NS","LT.NS","ULTRACEMCO.NS"
]

ticker = st.sidebar.selectbox("Select Stock", stocks)
period = st.sidebar.selectbox("Select Period", ["6mo", "1y", "2y", "5y"])

# ================= AUTO REFRESH CONTROL =================
refresh_time = st.sidebar.slider("🔄 Refresh Interval (sec)", 10, 120, 60)

count = st_autorefresh(
    interval=refresh_time * 1000,
    limit=None,
    key="refresh"
)

st.caption(f"🔄 Auto-refresh count: {count}")

# ================= CACHED DATA LOADER =================
@st.cache_data(ttl=60)
def load_data(ticker, period):
    return fetch_stock_data(ticker, period)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()

now = datetime.datetime.now()

if now.hour >= 9 and now.hour <= 15:
    st.success("🟢 Market Open")
else:
    st.error("🔴 Market Closed")

# ================= FETCH DATA =================
data = load_data(ticker, period)

if data.empty:
    st.error("No data fetched. Check ticker or internet.")
    st.stop()

# ================= DATA =================
data = fetch_stock_data(ticker, period)

if data.empty:
    st.error("No data fetched. Check ticker or internet.")
    st.stop()

data = add_indicators(data)
data = generate_signals(data)

clean_data = data.dropna(subset=["Close"])

# ================= KPI =================
st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

if len(clean_data) > 1:
    current_price = clean_data["Close"].iloc[-1]
    prev_price = clean_data["Close"].iloc[-2]

    change = current_price - prev_price
    percent_change = (change / prev_price) * 100

    col1.metric("Current Price", f"₹{current_price:.2f}")
    col2.metric("Change", f"₹{change:.2f}")
    col3.metric("Change %", f"{percent_change:.2f}%")

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "🤖 ML", "🧠 Decision", "📡 Signals"])

# ================= CHARTS =================
with tab1:
    st.subheader("Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )])

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Moving Averages")
    st.line_chart(data[["Close", "MA20", "MA50"]])

    # RSI
    st.subheader("RSI Indicator")

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))

    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

    fig_rsi.update_layout(template="plotly_dark")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    st.subheader("MACD Indicator")

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["Signal_Line"], name="Signal"))

    fig_macd.update_layout(template="plotly_dark")
    st.plotly_chart(fig_macd, use_container_width=True)

# ================= ML =================
with tab2:
    st.subheader("Model Performance")

    result = run_ml_pipeline(data)

    if result[0] is None:
        st.warning("⚠️ Not enough data for ML. Select 6mo or 1y.")
    else:
        best_model, results, predictions_dict, y_test, prediction = result

        st.metric("Predicted Price", f"₹{prediction:.2f}")
        st.write(f"Best Model: {best_model}")
        st.write(f"R² Score: {results[best_model]:.4f}")

        st.subheader("Model Comparison")

        st.bar_chart(pd.DataFrame({
            "Model": list(results.keys()),
            "R² Score": list(results.values())
        }).set_index("Model"))

        st.subheader("Actual vs Predicted")

        best_preds = predictions_dict[best_model]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=y_test, name="Actual"))
        fig2.add_trace(go.Scatter(y=best_preds, name="Predicted"))

        st.plotly_chart(fig2)

# ================= DECISION ENGINE =================
with tab3:
    st.subheader("🧠 Final Decision Engine")

    # ML + Regression
    result = run_ml_pipeline(data)
    _, _, _, _, predicted_price = result

    model_name, acc, pred, confidence = run_classification_pipeline(data)
    ml_signal = "BUY" if pred == 1 else "SELL"

    latest_row = data.iloc[-1]
    current_price = data["Close"].iloc[-1]

    final_signal, reg_signal, score = get_final_decision(
        current_price,
        predicted_price,
        ml_signal,
        confidence,
        latest_row
    )

    # UI styling
    color_class = "hold"
    if final_signal == "BUY":
        color_class = "buy"
    elif final_signal == "SELL":
        color_class = "sell"

    st.markdown(f"""
    <div class="metric-card">
    <h2>Final Decision: <span class="{color_class}">{final_signal}</span></h2>
    <p>Regression: {reg_signal}</p>
    <p>ML Signal: {ml_signal}</p>
    <p>Confidence: {confidence:.2f}</p>
    <p>Score: {score}</p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence"},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

# ================= SIGNALS =================
with tab4:
    st.subheader("Trading Signals")

    st.dataframe(data[["Close", "MA20", "MA50", "Signal"]].tail(10))

    latest_signal = data["Signal"].iloc[-1]

    if latest_signal == "BUY":
        st.success("📈 BUY Signal Detected")
    elif latest_signal == "SELL":
        st.error("📉 SELL Signal Detected")
    else:
        st.warning("⚖️ HOLD")

# ================= DEBUG =================
st.write("Latest Indicators")
st.write(data[["RSI", "MACD", "Signal_Line"]].tail(5))