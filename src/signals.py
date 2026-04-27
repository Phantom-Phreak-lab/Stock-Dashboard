import pandas as pd

def generate_signals(df):
    df = df.copy()
    signals = ["HOLD"]

    for i in range(1, len(df)):
        rsi = df["RSI"].iloc[i]
        macd = df["MACD"].iloc[i]
        signal_line = df["Signal_Line"].iloc[i]

        ma20 = df["MA20"].iloc[i]
        ma50 = df["MA50"].iloc[i]

        if pd.isna(rsi) or pd.isna(macd) or pd.isna(signal_line):
            signals.append("HOLD")

        # 🔥 BUY (more flexible)
        elif rsi < 40 or (macd > signal_line and ma20 > ma50):
            signals.append("BUY")

        # 🔥 SELL (more flexible)
        elif rsi > 60 or (macd < signal_line and ma20 < ma50):
            signals.append("SELL")

        else:
            signals.append("HOLD")

    df["Signal"] = signals
    return df