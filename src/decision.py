# src/decision_engine.py

import pandas as pd


def get_final_decision(current_price, predicted_price, ml_signal, confidence, latest_row):
    
    # 🔥 Extract values safely
    rsi = latest_row.get("RSI")
    macd = latest_row.get("MACD")
    signal_line = latest_row.get("Signal_Line")

    # 🔥 Convert None → NaN
    if rsi is None:
        rsi = float("nan")
    if macd is None:
        macd = float("nan")
    if signal_line is None:
        signal_line = float("nan")

    # 🔹 Regression signal
    if predicted_price is not None and current_price is not None:
        reg_signal = "BUY" if predicted_price > current_price else "SELL"
    else:
        reg_signal = "HOLD"

    score = 0

    # ================= ML (weight = 2) =================
    if ml_signal == "BUY":
        score += 2
    elif ml_signal == "SELL":
        score -= 2

    # ================= RSI (weight = 1) =================
    if not pd.isna(rsi):
        if rsi < 40:
            score += 1
        elif rsi > 60:
            score -= 1

    # ================= MACD (weight = 1) =================
    if not pd.isna(macd) and not pd.isna(signal_line):
        if macd > signal_line:
            score += 1
        elif macd < signal_line:
            score -= 1

    # ================= REGRESSION (weight = 1) =================
    if reg_signal == "BUY":
        score += 1
    elif reg_signal == "SELL":
        score -= 1

    # ================= FINAL DECISION =================
    if score >= 2:
        final_signal = "BUY"
    elif score <= -2:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    return final_signal, reg_signal, score