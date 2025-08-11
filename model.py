#model.py
import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
from indicators import get_indicators
from data_fetcher import fetch_live_data, fetch_historical_data
from indicators import preprocess_and_generate_signals


# Hyperparameters
EPISODES = 60
LEARNING_RATE = 0.0003
DISCOUNT_RATE = 0.95
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
BATCH_SIZE = 2
CLIP_RATIO = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
RETRAIN_EVERY = 50  # Retrain model every 50 episodes

# PPO Agent class using the indicator data
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.experience_buffer = []  # Store live trading experiences for retraining

        # Build the PPO model
        self.model = self._build_model()

    def _build_model(self):
        """Builds the PPO model with policy and value networks."""
        state_input = layers.Input(shape=(self.state_size,))
        
        # Shared network layers 128 layers
        x = layers.Dense(128, activation='relu')(state_input)
        x = layers.Dense(128, activation='relu')(x)
        
        # Policy output (action probabilities)
        policy_output = layers.Dense(self.action_size, activation='softmax')(x)
        
        # Value output (state value estimation)
        value_output = layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=state_input, outputs=[policy_output, value_output])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def choose_action(self, state):
        """Select an action based on the state using the policy network."""
        state = np.reshape(state, [1, self.state_size])  # Ensure the state has batch dimension
        policy, _ = self.model.predict(state)
        
        # Check if the policy output contains NaN
        if np.isnan(policy).any():
            raise ValueError(f"NaN values detected in policy output: {policy}")

        # Normalize the policy to ensure it sums to 1
        policy = policy / np.sum(policy[0])

        action = np.random.choice(self.action_size, p=policy[0])  # Choose action based on policy
        return action


    def save(self, model_path):
        """Save the trained model."""
        self.model.save(model_path)
        print(f"Model saved at {model_path}")

    #classmethod
    def load(cls, model_path):
        """Load the saved model."""
        loaded_model = tf.keras.models.load_model(model_path)
        instance = cls.__new__(cls)  # Create an uninitialized instance of PPOAgent
        instance.model = loaded_model
        return instance

    # Other methods like choose_action, train, etc.


    def choose_action(self, state):
        """Select an action based on the state using the policy network."""
        state = np.reshape(state, [1, self.state_size])  # Ensure the state has batch dimension
        policy, _ = self.model.predict(state)
        action = np.random.choice(self.action_size, p=policy[0])  # Choose action based on policy
        return action

    def store_experience(self, state, action, reward, next_state, done):
        """Store live trading experiences."""
        self.experience_buffer.append((state, action, reward, next_state, done))

    def train(self, states, actions, rewards, next_states, dones):
        """Train the model using PPO."""
        
        # Convert states and other inputs to NumPy arrays and ensure proper shape
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Reshape states to remove extra dimensions if necessary
        if len(states.shape) == 3 and states.shape[1] == 1:  # Remove the extra 1 dimension
            states = np.squeeze(states, axis=1)

        # Ensure that states are of shape (batch_size, state_size)
        assert len(states.shape) == 2, f"Expected states shape (batch_size, state_size), but got {states.shape}"

        # Predict the old policy and values using the current model
        old_policy, old_values = self.model.predict(states)

        # Compute advantages and target values
        advantages, target_values = self._compute_advantages(rewards, next_states, dones, old_values)

        # Training the model using PPO loss
        with tf.GradientTape() as tape:
            # Get the current policy and value
            policy, value = self.model(states)

            # Gather action probabilities
            action_probs = tf.gather(policy, actions, batch_dims=1, axis=1)
            old_action_probs = tf.gather(old_policy, actions, batch_dims=1, axis=1)

            # Compute the PPO ratio
            ratio = action_probs / (old_action_probs + 1e-10)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)

            # Compute the loss components
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
            value_loss = tf.reduce_mean((target_values - value) ** 2)
            entropy_loss = -tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

            # Total loss
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

        # Compute gradients and apply them
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))






    def _compute_advantages(self, rewards, next_states, dones, values):
        """Calculate the advantages using GAE (Generalized Advantage Estimation)."""
        advantages = np.zeros_like(rewards)
        target_values = np.zeros_like(rewards)
        
        next_value = 0
        for t in range(len(rewards) - 1, -1, -1):
            if dones[t]:
                next_value = 0
            delta = rewards[t] + DISCOUNT_RATE * next_value - values[t]
            advantages[t] = delta + DISCOUNT_RATE * advantages[t + 1] if t < len(rewards) - 1 else delta
            target_values[t] = advantages[t] + values[t]
            next_value = values[t]

        return advantages, target_values

    def periodic_retrain(self):
        """Periodically retrain the model using the stored experiences."""
        if len(self.experience_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = zip(*self.experience_buffer)
            self.train(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
            print("Retrained the model using live experiences.")
            self.experience_buffer.clear()


# Reward function based on trade outcomes
def calculate_reward(profit_or_loss, hit_stop_loss=False, transaction_cost=0.001):
    reward = profit_or_loss - transaction_cost  # Deduct transaction cost
    if hit_stop_loss:
        return -abs(reward) * 2  # Larger penalty for hitting stop-loss
    return reward



def prepare_data():
    symbol = 'BTCUSDT'  # Replace with your desired symbol
    df = fetch_live_data(symbol)
    # Further processing of df if necessary
    return df


# Main training loop with live trading and periodic retraining
def train_ppo_agent():
    """Training PPO agent with market data and live experiences for a specific token."""
    
    # Prompt the user to input the token name
    symbol = input("Enter the trading pair for training (e.g., BTCUSDT): ").strip().upper()

    state_size = 10  # Define based on indicators
    action_size = 3  # Actions: buy, hold, sell

    # Check if a pre-trained model for the token exists, otherwise create a new agent
    model_path = f"ppo_trading_model_{symbol}.h5"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model for {symbol}...")
        agent = PPOAgent.load(model_path)
    else:
        print(f"No pre-trained model found for {symbol}. Creating a new PPO agent.")
        agent = PPOAgent(state_size, action_size)

    for episode in range(EPISODES):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        # Fetch data for the user-specified symbol
        data = fetch_historical_data(symbol)  # Use symbol to fetch relevant data
        
        if data is None or data.empty:
            print(f"Skipping training: Data for {symbol} is empty or None.")
            continue  # Skip this episode

        # Step 1: Preprocess the data and calculate the signals
        data = preprocess_and_generate_signals(data)
        print(data.columns)  # Check if all necessary columns are present after processing

        # Define the exact columns that match the expected state size
        selected_features = ['Close', 'Volume', 'RSI', 'EMA_21', 'EMA_50', 
                             'EMA_200', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_Signal']

        # Step 2: Checking for missing columns
        missing_columns = [col for col in selected_features if col not in data.columns]
        if missing_columns:
            print(f"Error: Missing columns in data: {missing_columns}")
            continue  # Skip this episode if essential columns are missing

        for step in range(len(data) - 1):
            state = data[selected_features].iloc[step].values.reshape(1, -1)
            next_state = data[selected_features].iloc[step + 1].values.reshape(1, -1)

            # Check if state contains NaN values
            if np.isnan(state).any():
                print(f"Skipping step {step}: State contains NaN values.")
                continue

            if state.shape[1] != state_size:
                print(f"Skipping step {step}: Incorrect state size. Expected {state_size}, but got {state.shape[1]}")
                continue

            action = agent.choose_action(state)
            reward = calculate_reward(profit_or_loss=1.0)
            done = False

            agent.store_experience(state, action, reward, next_state, done)


            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        if episode % RETRAIN_EVERY == 0:
            agent.periodic_retrain()

        # Save the model periodically
        if episode % 50 == 0:
            agent.model.save(f"ppo_model_episode_{episode}_{symbol}.keras")

        print(f"Episode {episode}/{EPISODES} completed for {symbol}.")


# Run the training process
train_ppo_agent()  # No arguments are needed here since token will be asked within the function


