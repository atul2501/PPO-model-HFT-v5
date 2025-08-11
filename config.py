# config.py


# Binance API Credentials (replace with environment variables in production)
api_key = "1RHip0xIFG6ERjcixOFmRjohMnMKZzV1QkHxoFwK6sJYZdpoSY8psaWCoI3eITkK"
api_secret = "0MCVQjZ3ptAhqu5J405FTWRI947gDqrrZDBE8R9tBmLlOAR17uAqvNbsPxunUDOi"

# Base URL for Binance USDⓂ-Margined Futures API
base_url = "https://fapi.binance.com"

# Binance client setup
from binance.client import Client
client = Client(api_key, api_secret)

# RL Agent Hyperparameters
EPISODES = 1000  # Number of episodes for training the agent
BATCH_SIZE = 2  # Batch size for experience replay

# Other hyperparameters can be added here, such as learning rate, discount factor, etc.
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
