import time
from data_fetcher import fetch_live_data  # Fetch live data
from trade_execution import execute_trade, close_position, set_take_profit, set_stop_loss
from model import PPOAgent  # Use PPO agent for trading decisions
from indicators import preprocess_and_generate_signals  # Import the signal generation function

# Function for live trading
def start_live_trading(symbol, agent, leverage=6, trade_amount=1, seq_length=60):
    balance = 20  # Initial balance for trading
    position = None
    total_profit = 0
    stop_loss = None
    take_profit = None
    entry_price = None

    print(f"Live trading started for {symbol}...")

    while True:
        try:
            # Fetch live data and get the most recent price
            live_data = fetch_live_data(symbol, seq_length=seq_length)

            if live_data is None:
                print(f"No valid live data for {symbol}. Retrying in 1 minute...")
                time.sleep(60)
                continue

            # Preprocess and generate signals
            live_data = preprocess_and_generate_signals(live_data)
            current_market_price = live_data['Close'].iloc[-1]

            # PPO agent acts on the latest live data
            action = agent.act(live_data)
            print(f"Current Price: {current_market_price}, Action: {action}")

            # If no position is open, decide to buy or sell
            if position is None:
                if action == 1:  # BUY
                    position = "BUY"
                    entry_price = current_market_price
                    print(f"Executing BUY at {entry_price}")
                    execute_trade("BUY", trade_amount, leverage, symbol)
                    stop_loss = set_stop_loss(entry_price, side="BUY", stop_loss_percent=0.02)
                    take_profit = set_take_profit(entry_price, side="BUY", take_profit_percent=0.05)

                elif action == 2:  # SELL
                    position = "SELL"
                    entry_price = current_market_price
                    print(f"Executing SELL at {entry_price}")
                    execute_trade("SELL", trade_amount, leverage, symbol)
                    stop_loss = set_stop_loss(entry_price, side="SELL", stop_loss_percent=0.02)
                    take_profit = set_take_profit(entry_price, side="SELL", take_profit_percent=0.05)

            # Check stop loss and take profit conditions
            elif position == "BUY":
                if current_market_price <= stop_loss:
                    profit = (current_market_price - entry_price) * leverage
                    total_profit += profit
                    balance += profit
                    print(f"Closed BUY at {current_market_price} (Stop Loss). Profit: {profit}")
                    close_position(symbol, "BUY")
                    position = None
                elif current_market_price >= take_profit:
                    profit = (current_market_price - entry_price) * leverage
                    total_profit += profit
                    balance += profit
                    print(f"Closed BUY at {current_market_price} (Take Profit). Profit: {profit}")
                    close_position(symbol, "BUY")
                    position = None

            elif position == "SELL":
                if current_market_price >= stop_loss:
                    profit = (entry_price - current_market_price) * leverage
                    total_profit += profit
                    balance += profit
                    print(f"Closed SELL at {current_market_price} (Stop Loss). Profit: {profit}")
                    close_position(symbol, "SELL")
                    position = None
                elif current_market_price <= take_profit:
                    profit = (entry_price - current_market_price) * leverage
                    total_profit += profit
                    balance += profit
                    print(f"Closed SELL at {current_market_price} (Take Profit). Profit: {profit}")
                    close_position(symbol, "SELL")
                    position = None

            # Print the current balance and total profit
            print(f"Balance: {balance}, Total Profit: {total_profit}")

            # Wait for 1 minute before fetching the next live data
            time.sleep(60)

        except Exception as e:
            print(f"Error in live trading loop: {e}")
            time.sleep(60)  # Wait before retrying to avoid flooding requests

# Example usage
if __name__ == "__main__":
    symbol = input("Enter trading pair (e.g., BTCUSDT): ").strip().upper()
    leverage = float(input("Enter leverage (e.g., 10 for 10x leverage): "))
    trade_amount = float(input("Enter trade amount in USDT: "))

    # Initialize your PPO agent (load from saved model if available)
    agent = PPOAgent.load("ppo_trading_model")  # Assuming the model is saved
    
    # Start live trading
    start_live_trading(symbol, agent, leverage=leverage, trade_amount=trade_amount)
