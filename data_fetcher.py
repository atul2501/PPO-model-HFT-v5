#data_fetcher.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from config import client
from indicators import preprocess_and_generate_signals, calculate_rsi, calculate_ema, calculate_bollinger_bands, calculate_macd, get_order_book_depth, get_funding_rate  # Importing additional functions

def fetch_historical_data(symbol, interval='1m', limit=1000):
    """
    Fetch historical data from Binance and return a DataFrame.
    :param symbol: The trading pair (e.g., 'BTCUSDT')
    :param interval: The interval between data points (e.g., '1m', '5m', '1h')
    :param limit: The number of data points to retrieve
    :return: A DataFrame with Open, High, Low, Close, and Volume columns
    """
    try:
        # Fetch the data from Binance API
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
            'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
            'Taker buy quote asset volume', 'Ignore'
        ])
        
        # Convert numerical columns from string to float
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        
        # Ensure no missing columns or values
        if df.empty:
            raise ValueError("Empty DataFrame returned from Binance.")

        # Handle any missing values (optional: forward fill missing data)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].ffill()


        return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()


def fetch_live_data(symbol, seq_length=10):
    """
    Fetch live data and calculate indicators for the RL agent.
    :param symbol: The trading pair (e.g., 'BTCUSDT')
    :param seq_length: The number of recent data points to retrieve
    :return: The most recent live data with indicators, ready for RL agent processing
    """
    # Fetch recent historical data for the symbol (used as live data)
    live_data = fetch_historical_data(symbol, limit=seq_length)

    # Check if live data is available and non-empty
    if live_data is None or live_data.empty:
        print(f"Error: No live data available for {symbol}.")
        return None

    # Preprocess the live data by calculating indicators (RSI, EMA, Bollinger Bands, MACD)
    live_data = preprocess_and_generate_signals(live_data)

    # Return the live data processed with indicators
    return live_data

def generate_combined_signal(symbol='BTCUSDT'):
    """
    Generate a combined trading signal based on various indicators and the order book depth.
    :param symbol: The trading pair (e.g., 'BTCUSDT')
    :return: A trading signal indicating 'Bullish', 'Bearish', or 'Neutral'
    """
    try:
        # Get real-time data for order book depth and funding rate
        bid_depth, ask_depth = get_order_book_depth(symbol=symbol)
        funding_rate = get_funding_rate(symbol=symbol)
        
        # Fetch historical price data for technical indicators
        df = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "100 minutes ago UTC")
        df = pd.DataFrame(df, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
        df = preprocess_and_generate_signals(df)

        if df is None or df.empty:
            print(f"Error: No valid data to process for {symbol}.")
            return None

        # Trading signal logic based on indicators and order book depth
        if bid_depth > ask_depth and funding_rate > 0:
            signal = "Bullish - Market sentiment is bullish (Order book and funding rate)"
        elif ask_depth > bid_depth and funding_rate < 0:
            signal = "Bearish - Market sentiment is bearish (Order book and funding rate)"
        else:
            signal = "Neutral - Mixed signals from order book and funding rate"
        
        return signal
    except Exception as e:
        print(f"Error generating combined signal for {symbol}: {e}")
        return None
