import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from binance.client import Client

# Import the config.py for API keys
from config import api_key, api_secret, client

# Function to calculate RSI with customizable length
def calculate_rsi(df, length=14):
    """Calculate RSI using pandas_ta for the specified length."""
    df['RSI'] = ta.rsi(df['Close'], length=length)
    return df

# Function to calculate EMA for a specified period
def calculate_ema(df, window):
    """Calculate EMA for the DataFrame."""
    df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20, num_std=2):
    """Calculate Bollinger Bands for the given DataFrame."""
    df['SMA'] = df['Close'].rolling(window=window).mean()  # Simple Moving Average
    rolling_std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['SMA'] + (rolling_std * num_std)  # Upper Band
    df['BB_Lower'] = df['SMA'] - (rolling_std * num_std)  # Lower Band

    # Fill any NaN values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

# Function to calculate MACD
def calculate_macd(df):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()  # Signal line

    df['MACD'] = macd
    df['MACD_Signal'] = signal
    return df

# Function to get Order Book Depth data
def get_order_book_depth(symbol='BTCUSDT', limit=100):
    """Get order book depth from Binance API."""
    order_book = client.get_order_book(symbol=symbol, limit=limit)
    bids = order_book['bids']  # Buy orders
    asks = order_book['asks']  # Sell orders

    bid_depth = sum([float(bid[1]) for bid in bids])  # Total buy order volume
    ask_depth = sum([float(ask[1]) for ask in asks])  # Total sell order volume

    return bid_depth, ask_depth

# Function to get Funding Rate
def get_funding_rate(symbol='BTCUSDT'):
    """Get funding rate from Binance Futures."""
    funding_info = client.futures_funding_rate(symbol=symbol, limit=1)
    funding_rate = float(funding_info[0]['fundingRate'])
    return funding_rate

# Function to preprocess and generate signals
def preprocess_and_generate_signals(data):
    # Convert necessary columns to numeric
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

    # Calculate RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)

    # Calculate EMAs
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Calculate Bollinger Bands
    bb = ta.bbands(data['Close'], length=20, std=2)
    data['BB_Upper'] = bb['BBU_20_2.0']
    data['BB_Lower'] = bb['BBL_20_2.0']

    # Calculate MACD
    macd = ta.macd(data['Close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_Signal'] = macd['MACDs_12_26_9']

    # Forward and backward fill any NaN values
    data.ffill(inplace=True)  # Forward fill NaN values
    data.bfill(inplace=True)  # Backward fill NaN values (as a fallback)

    return data


# Function to generate combined trading signal
def combined_trading_signal(symbol='BTCUSDT'):
    bid_depth, ask_depth = get_order_book_depth(symbol=symbol)
    funding_rate = get_funding_rate(symbol=symbol)

    df = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "100 minutes ago UTC")
    df = pd.DataFrame(df, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 
                                   'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 
                                   'Taker_buy_quote_asset_volume', 'Ignore'])
    df = preprocess_and_generate_signals(df)

    if df is not None:
        rsi = df['RSI'].iloc[-1]  # Last RSI value

        # Trading signal logic based on indicators and thresholds
        if bid_depth > ask_depth and funding_rate > 0 and rsi < 30:
            signal = "Bullish - Market sentiment is bullish (Order book, funding rate, and RSI)"
        elif ask_depth > bid_depth and funding_rate < 0 and rsi > 70:
            signal = "Bearish - Market sentiment is bearish (Order book, funding rate, and RSI)"
        else:
            signal = "Neutral - Mixed signals from order book, funding rate, and RSI"
    else:
        signal = "Data insufficient for analysis"

    return signal

# Optional: Plotting some of the calculated indicators
def plot_indicators(df):
    """Plot the indicators like RSI, EMA, Bollinger Bands, and MACD."""
    plt.figure(figsize=(12, 10))

    # Plot RSI
    plt.subplot(4, 1, 1)
    plt.plot(df['RSI'], label='RSI', color='orange')
    plt.title('RSI (Relative Strength Index)')
    plt.legend()

    # Plot EMA
    plt.subplot(4, 1, 2)
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['EMA_21'], label='EMA 21', color='red')
    plt.plot(df['EMA_50'], label='EMA 50', color='green')
    plt.plot(df['EMA_200'], label='EMA 200', color='purple')
    plt.title('Exponential Moving Averages (EMA)')
    plt.legend()

    # Plot Bollinger Bands
    plt.subplot(4, 1, 3)
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['BB_Upper'], label='Upper Bollinger Band', color='red')
    plt.plot(df['BB_Lower'], label='Lower Bollinger Band', color='green')
    plt.title('Bollinger Bands')
    plt.legend()

    # Plot MACD
    plt.subplot(4, 1, 4)
    plt.plot(df['MACD'], label='MACD', color='blue')
    plt.plot(df['MACD_Signal'], label='MACD Signal', color='red')
    plt.title('MACD (Moving Average Convergence Divergence)')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Function to get the indicators from the DataFrame
def get_indicators(df):
    """Return calculated indicators from the DataFrame."""
    df = calculate_indicators(df)
    return df[['RSI', 'EMA_21', 'EMA_50', 'EMA_200', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal']]
