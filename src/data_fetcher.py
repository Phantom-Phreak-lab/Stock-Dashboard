import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker, period="6mo", interval="1d"):
    """
    Fetch stock data from Yahoo Finance (robust version)
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )

        # 🔥 Fix MultiIndex issue (IMPORTANT)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 🔥 Drop completely empty rows
        data = data.dropna(how="all")

        # 🔥 Final validation
        if data.empty or len(data) == 0:
            raise ValueError("No data found for this ticker")

        return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()