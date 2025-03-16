import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Trading environment for SAC agent to learn trading with focus on transaction cost reduction.
    
    This environment simulates a trading scenario with transaction costs and provides
    state observations, rewards, and done signals to the RL agent. The SAC implementation
    is specifically designed to reduce transaction costs and optimize trading frequency.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        window_size: int = 10,
        transaction_cost: float = 0.0001, # 0.1 pip per trade
        initial_balance: float = 10000.0,
        reward_scaling: float = 1.0,
        trade_penalty: float = 0.01,  # Penalty for each trade
        holding_penalty: float = 0.001,  # Small penalty for holding positions over time
        correct_direction_reward: float = 0.03,  # Added reward for correct directional trades
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
            holding_penalty: Small penalty for holding positions over time
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
        self.holding_penalty = holding_penalty
        self.correct_direction_reward = correct_direction_reward
        self.min_trades_per_episode = min_trades_per_episode
        self.min_trades_penalty = min_trades_penalty
        self.target_trade_frequency = target_trade_frequency
        
        # For tracking cumulative variables
        self.current_step = 0
        self.current_position = 0
        self.current_balance = initial_balance
        self.start_balance = initial_balance
        self.trade_count = 0
        self.transaction_costs = 0
        self.position_history = []
        self.returns_history = []
        self.trade_history = []
        self.trade_durations = []
        self.last_trade_step = 0
        self.current_trade_duration = 0
        self.portfolio_values = []
        
        # Check if the data has the required columns
        required_cols = ['close', 'returns', 'meta_position', 'meta_confidence']
        for col in required_cols:
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found in data. This might cause issues.")
        
        # Define action and observation spaces
        # Action space: Continuous value between -1 and 1
        # -1 represents maximum short position, 1 represents maximum long position
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: Features + position + trade duration 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(feature_columns) + 2,),
            dtype=np.float32
        )
        
        # Calculate meta strategy trades for reference
        if 'meta_position' in self.data.columns:
            self.meta_position_changes = (self.data['meta_position'].diff() != 0).sum()
            self.min_trades_per_episode = max(
                self.min_trades_per_episode, 
                int(self.meta_position_changes * target_trade_frequency)
            )
            logger.info(f"Meta strategy has {self.meta_position_changes} position changes")
            logger.info(f"Minimum trades target set to {self.min_trades_per_episode}")
        
        self._reset_sim_vars()
    
    def _reset_sim_vars(self):
        """Reset simulation variables."""
        self.current_step = 0
        self.current_position = 0
        self.current_balance = self.initial_balance
        self.start_balance = self.initial_balance
        self.trade_count = 0
        self.transaction_costs = 0
        self.position_history = []
        self.returns_history = []
        self.trade_history = []
        self.trade_durations = []
        self.last_trade_step = 0
        self.current_trade_duration = 0
        self.portfolio_values = [self.current_balance]
        
    def reset(self):
        """Reset the environment to initial state."""
        self._reset_sim_vars()
        return self._get_observation()
    
    def _get_observation(self):
        """
        Construct the state observation.
        
        Returns:
            Numpy array of state features
        """
        # Get window of feature data
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.window_size + 1)
        
        # Extract features from data
        try:
            # Get original features
            features = self.data.iloc[end_idx][self.feature_columns].values.astype(np.float32)
            
            # Check if we need to handle feature count mismatch based on observation space dimension
            expected_feature_count = self.observation_space.shape[0] - 2  # Subtract position and duration features
            
            if len(features) < expected_feature_count:
                # Pad with zeros if we have fewer features than expected
                logger.warning(f"Padding features from {len(features)} to {expected_feature_count}")
                padding = np.zeros(expected_feature_count - len(features), dtype=np.float32)
                features = np.concatenate([features, padding])
            elif len(features) > expected_feature_count:
                # Truncate if we have more features than expected
                logger.warning(f"Truncating features from {len(features)} to {expected_feature_count}")
                features = features[:expected_feature_count]
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Fallback to zeros if feature extraction fails
            features = np.zeros(self.observation_space.shape[0] - 2, dtype=np.float32)
        
        # Add current position and trade duration as additional state features
        position_feature = np.array([self.current_position], dtype=np.float32)
        duration_feature = np.array([self.current_trade_duration], dtype=np.float32)
        
        # Combine all features
        state = np.concatenate([features, position_feature, duration_feature])
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action from the agent (continuous value between -1 and 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        
        # Convert continuous action to position
        # Action is between -1 and 1, we interpret it as position directly
        new_position = float(action[0])  # Extract scalar from action array
        
        # Record the position before the action
        previous_position = self.current_position
        
        # Check if a trade occurred (position changed)
        trade_occurred = abs(new_position - previous_position) > 0.1  # Use threshold to avoid tiny position changes
        
        # Update position
        self.current_position = new_position
        
        # Get market return for this step
        market_return = current_data['returns'] if 'returns' in current_data else 0
        
        # Calculate strategy return (position * market return)
        strategy_return = previous_position * market_return
        
        # Calculate transaction cost if trade occurred
        transaction_cost = 0
        if trade_occurred:
            # Cost is proportional to the size of the position change
            transaction_cost = abs(new_position - previous_position) * self.transaction_cost
            self.transaction_costs += transaction_cost
            self.trade_count += 1
            self.last_trade_step = self.current_step
            
            # Record trade duration if closing/changing a position
            if abs(previous_position) > 0.1:
                self.trade_durations.append(self.current_trade_duration)
            
            # Reset trade duration for new trade
            self.current_trade_duration = 0
            
            # Record trade information
            self.trade_history.append({
                'step': self.current_step,
                'date': self.data.index[self.current_step] if hasattr(self.data, 'index') else self.current_step,
                'previous_position': previous_position,
                'new_position': new_position,
                'market_return': market_return,
                'transaction_cost': transaction_cost
            })
        else:
            # Increment trade duration if in a position
            if abs(self.current_position) > 0.1:
                self.current_trade_duration += 1
        
        # Calculate reward components
        
        # 1. Strategy return minus transaction cost
        profit_reward = strategy_return - transaction_cost
        
        # 2. Penalty for trading (to reduce excessive trading)
        trading_penalty = self.trade_penalty * transaction_cost * 10 if trade_occurred else 0
        
        # 3. Small penalty for holding positions (to encourage efficient use of capital)
        holding_penalty = self.holding_penalty * abs(self.current_position) if abs(self.current_position) > 0.1 else 0
        
        # 4. Reward for trading in the correct direction (if we can determine it)
        direction_reward = 0
        if trade_occurred and 'meta_position' in current_data:
            meta_position = current_data['meta_position']
            # If our position matches the meta-labeler direction, give a small reward
            if (new_position > 0 and meta_position > 0) or (new_position < 0 and meta_position < 0):
                # Scale by meta confidence if available
                confidence_multiplier = current_data['meta_confidence'] if 'meta_confidence' in current_data else 1.0
                direction_reward = self.correct_direction_reward * confidence_multiplier
            # If we go against the meta-labeler with high confidence, give a small penalty
            elif (new_position > 0 and meta_position < 0) or (new_position < 0 and meta_position > 0):
                confidence_multiplier = current_data['meta_confidence'] if 'meta_confidence' in current_data else 1.0
                if confidence_multiplier > 0.7:  # Only penalize going against high confidence signals
                    direction_reward = -self.correct_direction_reward * confidence_multiplier
        
        # Combine reward components
        reward = profit_reward - trading_penalty - holding_penalty + direction_reward
        
        # Scale reward
        reward = reward * self.reward_scaling
        
        # Update account balance
        self.current_balance += profit_reward * self.initial_balance
        
        # Update histories
        self.position_history.append(self.current_position)
        self.returns_history.append(strategy_return)
        self.portfolio_values.append(self.current_balance)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Apply minimum trades penalty if done and below minimum
        if done and self.min_trades_per_episode > 0 and self.trade_count < self.min_trades_per_episode:
            # Penalty proportional to how far we are from minimum
            shortfall_ratio = (self.min_trades_per_episode - self.trade_count) / self.min_trades_per_episode
            min_trades_penalty = self.min_trades_penalty * shortfall_ratio
            reward -= min_trades_penalty
        
        # Get next observation
        next_observation = self._get_observation()
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'position': self.current_position,
            'balance': self.current_balance,
            'trade_count': self.trade_count,
            'transaction_costs': self.transaction_costs,
            'portfolio_value': self.current_balance,
            'return': profit_reward,
            'strategy_return': strategy_return
        }
        
        return next_observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode != 'human':
            return
        
        print(f"Step: {self.current_step}, Position: {self.current_position:.2f}, "
              f"Balance: {self.current_balance:.2f}, Trades: {self.trade_count}, "
              f"Transaction Costs: {self.transaction_costs:.4f}")
    
    def get_portfolio_value(self):
        """Get current portfolio value."""
        return self.current_balance
    
    def get_results(self):
        """Get results of the simulation."""
        return {
            'balance': self.current_balance,
            'trade_count': self.trade_count,
            'transaction_costs': self.transaction_costs,
            'returns': self.returns_history,
            'positions': self.position_history,
            'portfolio_values': self.portfolio_values,
            'trades': self.trade_history,
            'trade_durations': self.trade_durations
        } 