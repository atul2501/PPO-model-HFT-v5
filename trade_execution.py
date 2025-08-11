#trade_execuation.py
from config import client
import time

# Function to set leverage for futures trading
def set_leverage(symbol, leverage):
    try:
        response = client.futures_change_leverage(symbol=symbol, leverage=int(leverage))
        print(f"Leverage set to {leverage}x for {symbol}")
    except Exception as e:
        print(f"Error setting leverage: {e}")

# Validate action outside of execute_trade
def validate_action(action):
    if isinstance(action, str):
        print(f"Invalid action value received: {action}. Converting to integer.")
        action = 0 if action == "BUY" else 1 if action == "SELL" else None
    if action not in [0, 1]:
        raise ValueError("Invalid action passed")
    return action

# Function to execute a trade
def execute_trade(action, amount_in_usdt, leverage, symbol):
    try:
        # Ensure action is valid
        action = validate_action(action)

        print(f"Action passed to execute_trade: {action}")

        # Set leverage for the symbol
        set_leverage(symbol, leverage)

        # Fetch the current price of the asset
        price_data = client.futures_symbol_ticker(symbol=symbol)
        current_price = float(price_data['price'])
        quantity = amount_in_usdt / current_price  # Calculate quantity based on USDT amount

        # Fetch the asset precision from Binance Futures
        exchange_info = client.futures_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                price_precision = symbol_info['pricePrecision']
                precision = symbol_info['quantityPrecision']
                break

        # Round the quantity based on the asset's precision
        quantity = round(quantity, precision)

        # Determine the side based on the action
        side = "BUY" if action == 0 else "SELL"

        print(f"Preparing to execute: {side} {quantity} contracts for {symbol} at leverage {leverage}")

        # Create market order
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )

        # Log the order details after execution
        print(f"Trade executed: {side} {amount_in_usdt} USDT ({quantity} contracts) for {symbol} at leverage {leverage}. Order ID: {order['orderId']}")

        # Now, calculate stop-loss and take-profit
        entry_price = current_price
        stop_loss = set_stop_loss(entry_price, side, stop_loss_percent=0.02)
        take_profit = set_take_profit(entry_price, side, take_profit_percent=0.05)

        # Ensure stop-loss and take-profit prices conform to the asset's price precision
        stop_loss = round(stop_loss, price_precision)
        take_profit = round(take_profit, price_precision)

        # Place Stop-Loss Order
        stop_loss_order = client.futures_create_order(
            symbol=symbol,
            side='BUY' if side == 'SELL' else 'SELL',  # Opposite side to close
            type='STOP_MARKET',
            stopPrice=stop_loss,
            quantity=quantity,
            reduceOnly=True  # Ensures the order only reduces the position
        )
        print(f"Stop-loss order placed at: {stop_loss}")

        # Place Take-Profit Order
        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side='BUY' if side == 'SELL' else 'SELL',  # Opposite side to close
            type='TAKE_PROFIT_MARKET',
            stopPrice=take_profit,
            quantity=quantity,
            reduceOnly=True  # Ensures the order only reduces the position
        )
        print(f"Take-profit order placed at: {take_profit}")

        return order

    except Exception as e:
        print(f"Error executing trade: {e}")
        return None

def set_stop_loss(entry_price, side, stop_loss_percent=0.02):
    """
    Set stop-loss based on the entry price and desired percentage.
    :param entry_price: Entry price of the trade
    :param side: 'BUY' or 'SELL'
    :param stop_loss_percent: Percentage for stop-loss
    :return: Stop-loss price
    """
    if side == "BUY":
        stop_loss_price = entry_price * (1 - stop_loss_percent)
    elif side == "SELL":
        stop_loss_price = entry_price * (1 + stop_loss_percent)
    print(f"Stop-loss set at {stop_loss_price} for {side} position.")
    return stop_loss_price


def set_take_profit(entry_price, side, take_profit_percent=0.05):
    """
    Set take-profit based on the entry price and desired percentage.
    :param entry_price: Entry price of the trade
    :param side: 'BUY' or 'SELL'
    :param take_profit_percent: Percentage for take-profit
    :return: Take-profit price
    """
    if side == "BUY":
        take_profit_price = entry_price * (1 + take_profit_percent)
    elif side == "SELL":
        take_profit_price = entry_price * (1 - take_profit_percent)
    print(f"Take-profit set at {take_profit_price} for {side} position.")
    return take_profit_price


def close_position(symbol, side, amount):
    """
    Closes an open position by executing the opposite trade.
    :param symbol: The trading pair (e.g., 'BTCUSDT')
    :param side: Current side of the position, 'BUY' or 'SELL'
    :param amount: Amount to close
    """
    opposite_side = 'SELL' if side == 'BUY' else 'BUY'
    try:
        # Create a market order to close the position
        close_order = client.futures_create_order(
            symbol=symbol,
            side=opposite_side,
            type='MARKET',
            quantity=amount,
            reduceOnly=True  # Ensures this order only reduces the position
        )
        print(f"Position closed: {opposite_side} {amount} {symbol}. Order ID: {close_order['orderId']}")
        return close_order
    except Exception as e:
        print(f"Error closing position: {e}")
        return None

def auto_trade(symbol, leverage, amount_in_usdt, starting_action=0):
    """
    Auto-trade loop that opens a new trade after closing the previous one.
    It adjusts the next trade based on the outcome of the previous trade.
    :param symbol: Trading pair (e.g., 'BTCUSDT')
    :param leverage: Leverage for the trade
    :param amount_in_usdt: Amount in USDT for each trade
    :param starting_action: 0 for BUY, 1 for SELL
    """
    action = starting_action
    previous_profit = 0

    while True:
        # Execute a trade
        entry_price, quantity = execute_trade(action, amount_in_usdt, leverage, symbol)

        if entry_price is None or quantity is None:
            print("Error executing trade. Exiting auto-trade loop.")
            break

        # Set stop-loss and take-profit
        stop_loss = set_stop_loss(entry_price, "BUY" if action == 0 else "SELL")
        take_profit = set_take_profit(entry_price, "BUY" if action == 0 else "SELL")

        # Monitor the trade until it closes
        print("Monitoring trade...")
        while True:
            profit = check_open_position_and_exit(symbol, "BUY" if action == 0 else "SELL", take_profit, stop_loss, leverage)
            if profit is not None:
                previous_profit = profit
                break
            time.sleep(10)  # Check every 10 seconds

        # Adjust action for the next trade based on the previous profit (simple learning)
        if previous_profit > 0:
            action = 0  # Open BUY next
        else:
            action = 1  # Open SELL next

        print(f"Completed one trade. Preparing to open the next trade. Last profit: {previous_profit}")

def get_open_positions(symbol):
    """
    Fetches the current open positions for a given symbol from Binance Futures.
    :param symbol: The trading pair (e.g., 'BERAUSDT')
    :return: The current open position details (if any)
    """
    try:
        positions = client.futures_position_information(symbol=symbol)
        for position in positions:
            if float(position['positionAmt']) != 0:  # A position is open if the amount is non-zero
                return position
        return None
    except Exception as e:
        print(f"Error fetching open positions: {e}")
        return None


def check_open_position_and_exit(symbol, side, take_profit, stop_loss, leverage):
    """
    Check if an open position should be closed based on the current price relative to stop-loss and take-profit.
    :param symbol: The trading pair (e.g., 'BERAUSDT')
    :param side: The side of the open position ('BUY' or 'SELL')
    :param take_profit: Take-profit price
    :param stop_loss: Stop-loss price
    :param leverage: Leverage used in the trade
    :return: Profit made or lost, or None if the position remains open
    """
    # Retrieve open position for the symbol
    position = get_open_positions(symbol)
    
    # If there is no position open, return None
    if not position:
        return None

    # Fetch the current price and position details
    current_price = float(position['markPrice'])
    entry_price = float(position['entryPrice'])
    position_amt = abs(float(position['positionAmt']))  # Quantity of the position
    
    # Handle 'BUY' (long) positions
    if side == 'BUY':
        # Stop-loss triggered
        if current_price <= stop_loss:
            profit = (current_price - entry_price) * leverage * position_amt
            print(f"Closing BUY position at {current_price} (Stop-Loss triggered). Profit: {profit}")
            close_position(symbol, side, position_amt)
            return profit
        
        # Take-profit triggered
        elif current_price >= take_profit:
            profit = (current_price - entry_price) * leverage * position_amt
            print(f"Closing BUY position at {current_price} (Take-Profit triggered). Profit: {profit}")
            close_position(symbol, side, position_amt)
            return profit
    
    # Handle 'SELL' (short) positions
    elif side == 'SELL':
        # Stop-loss triggered
        if current_price >= stop_loss:
            profit = (entry_price - current_price) * leverage * position_amt
            print(f"Closing SELL position at {current_price} (Stop-Loss triggered). Profit: {profit}")
            close_position(symbol, side, position_amt)
            return profit
        
        # Take-profit triggered
        elif current_price <= take_profit:
            profit = (entry_price - current_price) * leverage * position_amt
            print(f"Closing SELL position at {current_price} (Take-Profit triggered). Profit: {profit}")
            close_position(symbol, side, position_amt)
            return profit

    # If the position is still open, return None to indicate that no exit happened
    return None












