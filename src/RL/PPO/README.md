# PPO for Trading with Transaction Cost Reduction

This module implements Proximal Policy Optimization (PPO) for trading with a focus on reducing transaction costs while maintaining returns similar to the original meta-labeled strategy.

## Overview

PPO is a state-of-the-art reinforcement learning algorithm that offers a good balance between sample efficiency, ease of implementation, and performance. This implementation is specifically designed to:

1. Learn a trading policy that reduces transaction costs by minimizing unnecessary trades
2. Maintain performance comparable to the original meta-labeled strategy
3. Produce more stable and consistent returns

## Key Features

- Actor-Critic architecture with separate policy and value networks
- Generalized Advantage Estimation (GAE) for more stable policy updates
- Clipped surrogate objective to prevent destructively large policy updates
- Entropy bonus to encourage exploration
- Batch training for more stable gradients
- Automatic model selection based on test set Sharpe ratio

## Implementation Details

### Files

- `ppo_model.py`: Contains the PPO model implementation (actor network, critic network, and PPO agent)
- `trading_env.py`: Trading environment that simulates trading with transaction costs
- `train_ppo.py`: Script for training and evaluating PPO models
- `run_ppo_backtest.py`: Runner script for executing the training and backtesting

### Environment

The trading environment is designed to:

- Penalize excessive trading through a trade penalty
- Reward correct directional trades
- Apply position duration penalties to discourage holding positions too long
- Track transaction costs and trading activity
- Ensure a minimum level of trading activity to maintain liquidity

### Key Hyperparameters

- `trade_penalty`: Penalty applied when making a trade (default: 0.01)
- `transaction_cost`: Cost per trade as a percentage (default: 0.0001, i.e., 0.1 pip)
- `min_trades_ratio`: Target minimum trades as a ratio of meta-labeled strategy trades (default: 0.1)
- `gamma`: Discount factor for future rewards (default: 0.99)
- `gae_lambda`: Lambda parameter for GAE (default: 0.95)
- `policy_clip`: Clipping parameter for PPO (default: 0.2)

## Usage

### Running the Backtest

```bash
# Navigate to the PPO directory
cd src/RL/PPO

# Run the backtest with default settings
python run_ppo_backtest.py
```

### Customizing Parameters

You can modify `run_ppo_backtest.py` to adjust various parameters such as:

- Number of training episodes
- Transaction cost
- Trade penalty
- Network architecture
- etc.

### Output

The training process will generate several outputs in the `reports/rl/ppo` directory:

1. Training metrics plot showing returns, costs, trades, and losses over episodes
2. Model comparison plot comparing PPO models with the meta-labeled strategy
3. In-sample vs out-of-sample returns plot
4. CSV files with detailed results

## Results Interpretation

The main goal of this implementation is to reduce transaction costs by minimizing unnecessary trades while maintaining comparable returns to the original meta-labeled strategy.

Key metrics to evaluate success:
- Trade reduction: Percentage reduction in number of trades compared to meta-labeled strategy
- Return comparison: Final returns compared to meta-labeled strategy
- Sharpe ratio: Risk-adjusted return metric

## Extending and Customizing

To customize this implementation:

1. Modify reward function in `trading_env.py` to emphasize different aspects
2. Adjust network architecture in `ppo_model.py`
3. Change hyperparameters in `run_ppo_backtest.py`

## References

- PPO Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- GAE Paper: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) 