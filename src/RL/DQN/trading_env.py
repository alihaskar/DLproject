import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Trading environment for DQN agent to learn trading with focus on transaction cost reduction.
    
    This environment simulates a trading scenario with transaction costs and provides
    state observations, rewards, and done signals to the RL agent.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        window_size: int = 10,
        transaction_cost: float = 0.0001, # 0.1 pip per trade
        initial_balance: float = 10000.0,
        reward_scaling: float = 1.0,
        trade_penalty: float = 0.01,  # Reduced from 0.05 to 0.01
        correct_direction_reward: float = 0.02,  # Added reward for correct directional trades
        min_trades_per_episode: int = 0,  # Minimum trades target
        min_trades_penalty: float = 0.2,  # Penalty for not meeting min trades at end of episode
        target_trade_frequency: float = 0.1,  # Target trade frequency (10% of original)
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame containing the trading data with required columns
            feature_columns: List of column names to use as state features
            window_size: Number of time steps to include in the state observation
            transaction_cost: Cost per trade as a decimal percentage
            initial_balance: Initial account balance
            reward_scaling: Scaling factor for rewards
            trade_penalty: Additional penalty for trading to discourage excessive trading
            correct_direction_reward: Reward for making a trade in the correct direction
            min_trades_per_episode: Minimum number of trades target for the episode
            min_trades_penalty: Penalty for not meeting min_trades_per_episode
            target_trade_frequency: Target trade frequency as a ratio
        """
        super().__init__()
        
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reward_scaling = reward_scaling
        self.trade_penalty = trade_penalty
        self.correct_direction_reward = correct_direction_reward
        self.min_trades_per_episode = min_trades_per_episode
        self.min_trades_penalty = min_trades_penalty
        self.target_trade_frequency = target_trade_frequency
        
        # Extract price columns if they exist
        self.price_columns = {
            'open': 'open' if 'open' in data.columns else None,
            'high': 'high' if 'high' in data.columns else None,
            'low': 'low' if 'low' in data.columns else None,
            'close': 'close' if 'close' in data.columns else None,
        }
        
        # Validate that we have at least close prices
        if self.price_columns['close'] is None:
            raise ValueError("Data must contain 'close' column for trading simulation")
        
        # Set up the action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Calculate observation shape based on features and window size
        obs_shape = (len(self.feature_columns) + 3,)  # +3 for position, balance, and trade count
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        # Initialize trading simulation variables
        self._reset_sim_vars()
        
        # Calculate total steps in the episode
        self.total_steps = len(data) - window_size - 1
        
        # If min_trades is not specified, use the target_trade_frequency
        if self.min_trades_per_episode == 0:
            self.min_trades_per_episode = int(self.total_steps * self.target_trade_frequency / 10)
            logger.info(f"Setting min trades per episode to {self.min_trades_per_episode}")
        
    def _reset_sim_vars(self):
        """Reset simulation variables to initial state."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.trades_executed = 0
        self.total_costs = 0.0
        self.last_trade_price = None
        self.done = False
        self.correct_trades = 0
        self.all_rewards = []
        
    def reset(self):
        """Reset the environment to initial state and return the initial observation."""
        self._reset_sim_vars()
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current state observation.
        
        Returns:
            numpy array of state features including current position and normalized balance
        """
        # Extract current features
        features = self.data.iloc[self.current_step][self.feature_columns].values
        
        # Add position, normalized balance, and normalized trade count to state
        position_onehot = np.array([self.position])
        balance_normalized = np.array([self.balance / self.initial_balance - 1.0])
        trades_normalized = np.array([self.trades_executed / max(1, self.min_trades_per_episode)])
        
        # Combine into state
        observation = np.concatenate([features, position_onehot, balance_normalized, trades_normalized])
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action: 0 (hold), 1 (buy), 2 (sell)
            
        Returns:
            observation, reward, done, info
        """
        # Get current price and next price for calculating returns
        current_price = self.data.iloc[self.current_step][self.price_columns['close']]
        
        # Calculate old position to check if trade is executed
        old_position = self.position
        
        # Determine new position based on action
        if action == 0:  # Hold
            new_position = self.position
        elif action == 1:  # Buy
            new_position = 1
        elif action == 2:  # Sell
            new_position = -1
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Calculate transaction cost if position changes
        transaction_cost = 0.0
        executed_trade = False
        correct_direction = False
        
        # Get the market direction for directional reward
        next_step = min(self.current_step + 1, len(self.data) - 1)
        next_price = self.data.iloc[next_step][self.price_columns['close']]
        price_change = (next_price - current_price) / current_price
        market_direction = 1 if price_change > 0 else -1 if price_change < 0 else 0
        
        if new_position != old_position and old_position != 0:
            # Closing existing position
            transaction_cost += current_price * self.transaction_cost
            executed_trade = True
            
        if new_position != old_position and new_position != 0:
            # Opening new position
            transaction_cost += current_price * self.transaction_cost
            executed_trade = True
            
            # Check if trade direction matches market direction for extra reward
            if new_position == market_direction:
                correct_direction = True
                self.correct_trades += 1
            
        # Apply trade penalty if a trade is executed, but scale it down for correct direction
        trade_penalty = 0.0
        directional_reward = 0.0
        
        if executed_trade:
            trade_penalty = self.trade_penalty
            if correct_direction:
                directional_reward = self.correct_direction_reward
                # Reduce penalty for correct direction trades
                trade_penalty *= 0.5
        
        # Record trading activity
        if executed_trade:
            self.trades_executed += 1
            self.total_costs += transaction_cost
            self.last_trade_price = current_price
            
        # Apply new position
        self.position = new_position
        
        # Move to next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Calculate returns based on position, price change, and costs
        position_return = old_position * price_change  # Use old position for returns
        
        # Apply transaction costs
        net_return = position_return - transaction_cost / current_price
        
        # Update balance
        self.balance *= (1 + net_return)
        
        # Calculate reward: returns after costs + directional reward - trade penalty
        reward = net_return * self.reward_scaling + directional_reward - trade_penalty
        
        # Add min trades penalty if episode is done and we didn't reach minimum trades
        if self.done and self.trades_executed < self.min_trades_per_episode:
            trade_deficit_ratio = (self.min_trades_per_episode - self.trades_executed) / self.min_trades_per_episode
            min_trades_penalty = trade_deficit_ratio * self.min_trades_penalty
            reward -= min_trades_penalty
            
        self.all_rewards.append(reward)
            
        # Get new observation
        new_observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'step': self.current_step,
            'price': current_price,
            'position': self.position,
            'return': position_return,
            'cost': transaction_cost,
            'net_return': net_return,
            'balance': self.balance,
            'trades_executed': self.trades_executed,
            'total_costs': self.total_costs,
            'directional_reward': directional_reward,
            'trade_penalty': trade_penalty,
            'correct_trades': self.correct_trades
        }
        
        return new_observation, reward, self.done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode != 'human':
            raise ValueError(f"Unsupported render mode: {mode}")
        
        print(f"Step: {self.current_step}, "
              f"Position: {self.position}, "
              f"Balance: {self.balance:.2f}, "
              f"Trades: {self.trades_executed}, "
              f"Costs: {self.total_costs:.4f}")
    
    def get_portfolio_value(self):
        """Get current portfolio value."""
        return self.balance 