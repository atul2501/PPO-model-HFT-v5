#main.py
import time
import os
import numpy as np
import pandas as pd
from indicators import preprocess_and_generate_signals  # Import from indicators correctly
from model import PPOAgent
from data_fetcher import fetch_live_data, fetch_historical_data  # Fetch live data function
from trade_execution import execute_trade, close_position, set_stop_loss, set_take_profit, check_open_position_and_exit, set_leverage  # Trade execution functions
from config import EPISODES, BATCH_SIZE  # Hyperparameters and configurations
import logging
from live_trading import start_live_trading



# Set up logging configuration
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message):
    """Logs a message to both console and log file."""
    print(f"Log: {message}")
    logging.info(message)

def get_ppo_state(df):
    """Convert the preprocessed live data into a state for the PPO agent."""
    state = df[['Close', 'Volume', 'RSI', 'EMA_21', 'EMA_50', 'EMA_200', 'BB_Upper', 'BB_Lower']].values[-1]

    return state


def manage_open_position(symbol, position_opened, position_side, entry_price, stop_loss, take_profit):
    """Check if stop loss or take profit conditions are met and close the position accordingly."""
    try:
        # Fetch the latest market data to compare with stop loss/take profit
        live_data = fetch_live_data(symbol)
        current_price = live_data['Close'].values[-1]

        log_message(f"Checking position: {position_side} at price {current_price}")

        if position_side == "BUY":
            if current_price <= stop_loss:
                log_message(f"Stop loss hit for Long position. Closing position at {current_price}")
                close_position(symbol, position_side)
                position_opened = False
            elif current_price >= take_profit:
                log_message(f"Take profit hit for Long position. Closing position at {current_price}")
                close_position(symbol, position_side)
                position_opened = False
        elif position_side == "SELL":
            if current_price >= stop_loss:
                log_message(f"Stop loss hit for Short position. Closing position at {current_price}")
                close_position(symbol, position_side)
                position_opened = False
            elif current_price <= take_profit:
                log_message(f"Take profit hit for Short position. Closing position at {current_price}")
                close_position(symbol, position_side)
                position_opened = False

        return position_opened  # Return the position status to update the loop

    except Exception as e:
        log_message(f"Error during position management: {e}")
        return position_opened  # Return the current status if an error occurs

def live_trading_loop(symbol, leverage, amount_in_usdt, model):
    """Main live trading loop with PPO model deciding actions."""
    try:
        log_message(f"Starting live trading for {symbol} with leverage {leverage} and amount {amount_in_usdt}")

        # Set leverage before starting trading
        set_leverage(symbol=symbol, leverage=leverage)

        # Fetch initial live data
        live_data = fetch_live_data(symbol)

        # Initialize placeholders for take_profit, stop_loss, and position status
        entry_price = None
        take_profit = None
        stop_loss = None
        position_opened = False  # Track whether there's an active position
        position_side = None  # Track whether the position is BUY or SELL

        while True:
            # Preprocess the live data for state generation
            live_data = preprocess_and_generate_signals(live_data)
            state = get_ppo_state(live_data)  # Most recent state for PPO model

            # Fetch the current price of the asset
            current_price = live_data['Close'].values[-1]

            # Log current market indicators and price for debugging
            log_message(f"Current State Indicators: RSI={live_data['RSI'].values[-1]}, EMA_21={live_data['EMA_21'].values[-1]}, Price={current_price}")


            # If no position is currently opened, proceed with trading decisions
            if not position_opened:
                
                # PPO model decides an action (buy, sell, hold)
                action = model.choose_action(state)
                log_message(f"Current Price: {current_price}, Action: {action}")


                # Execute the action: 0 -> Buy (Long), 1 -> Sell (Short), 2 -> Hold
                if action == 0:  # Buy/Long
                    log_message("Executing Long (Buy) Trade...")
                    execute_trade("BUY", amount_in_usdt, leverage, symbol)
                    entry_price = current_price
                    stop_loss = set_stop_loss(entry_price, "BUY",
                     stop_loss_percent=0.02)
                    take_profit = set_take_profit(entry_price, "BUY", take_profit_percent=0.05)
                    log_message(f"Long trade with entry {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                    position_opened = True  # Mark position as opened
                    position_side = "BUY"  # Mark the position side as Buy

                elif action == 1:  # Sell/Short
                    log_message("Executing Short (Sell) Trade...")
                    execute_trade("SELL", amount_in_usdt, leverage, symbol)
                    entry_price = current_price
                    stop_loss = set_stop_loss(entry_price, "SELL", stop_loss_percent=0.02)
                    take_profit = set_take_profit(entry_price, "SELL", take_profit_percent=0.05)
                    log_message(f"Short trade with entry {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                    position_opened = True  # Mark position as opened
                    position_side = "SELL"  # Mark the position side as Sell

                else:
                    log_message("Holding Position, no trade executed.")
            
            # Monitoring and position management will be handled in the next part.

            time.sleep(60)  # Delay before the next fetch

    except Exception as e:
        log_message(f"Error during live trading: {e}")

# main.py

if __name__ == "__main__":
    # Prompt user for input
    symbol = input("Enter the trading pair (e.g., BTCUSDT): ").strip().upper()

    leverage = float(input("Enter the leverage (e.g., 10 for 10x leverage): "))
    amount_in_usdt = float(input("Enter the amount in USDT: "))

    log_message(f"User inputs: symbol={symbol}, leverage={leverage}, amount_in_usdt={amount_in_usdt}")

    # Initialize PPO agent (check if model exists, otherwise create a new one)
    if os.path.exists(f"ppo_trading_model_{symbol}.h5"):
        model = PPOAgent.load(f"ppo_trading_model_{symbol}.h5")
        print("Model loaded successfully.")
    else:
        print(f"No saved model found for {symbol}. Initializing a new PPO agent and training with {symbol} data.")
        state_size = 10  # Ensure this matches your state feature size
        action_size = 3  # Example action size (buy, sell, hold)
        model = PPOAgent(state_size, action_size)

        # Train the PPO agent with the specified token's data
        train_ppo_agent(symbol, model)

    # Start live trading with the user-selected symbol
    try:
        live_trading_loop(symbol, leverage, amount_in_usdt, model)
    except Exception as e:
        print(f"An error occurred during live trading: {e}")
