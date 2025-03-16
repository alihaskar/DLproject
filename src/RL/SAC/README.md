# Soft Actor-Critic (SAC) for Trading with Transaction Cost Reduction

This implementation of Soft Actor-Critic (SAC) is designed specifically for optimizing trading strategies with a focus on transaction cost reduction.

## Overview

SAC is an off-policy actor-critic deep reinforcement learning algorithm that:
1. Uses maximum entropy reinforcement learning
2. Maintains a good balance between exploration and exploitation 
3. Is sample-efficient as an off-policy algorithm
4. Works well with continuous action spaces

The implementation is tailored to trading with the following key features:
- Continuous action space representing position size (-1 to 1)
- Custom reward function that balances returns against transaction costs
- Entropy maximization to explore diverse trading strategies
- Built-in mechanisms to reduce trading frequency

## Components

- `sac_model.py`: Core SAC algorithm implementation with actor-critic networks
- `trading_env.py`: Custom trading environment with transaction cost modeling
- `train_sac.py`: Script for training the SAC model on metalabeled data
- `run_sac_backtest.py`: Script for backtesting a trained SAC model

## Transaction Cost Reduction Approach

The SAC implementation is designed to address the issue of excessive trading in traditional strategies:

1. **Reward Structure**:
   - Base reward comes from trading returns
   - Trading penalty proportional to transaction cost 
   - Small holding penalty to discourage unnecessarily long positions
   - Bonus reward for trading in the same direction as the metalabeled strategy

2. **Continuous Position Sizing**:
   - Unlike discrete actions (buy/sell/hold), SAC outputs continuous position sizes
   - This allows for partial position adjustments that can reduce transaction costs
   - Position size range from -1 (full short) to 1 (full long)

3. **Entropy Regularization**:
   - Encourages exploration of diverse trading strategies
   - Helps avoid getting stuck in suboptimal strategies that trade too frequently

## Results

The model is trained to optimize returns while reducing the number of trades compared to the original metalabeled strategy. The evaluation metrics include:

- Total return
- Number of trades
- Transaction costs
- Sharpe ratio
- Maximum drawdown

## Usage

### Training

To train the SAC model on metalabeled data:

```bash
python src/RL/SAC/train_sac.py \
    --data_path data/cmma_metalabeled_atr_lag5_regime_filtered.csv \
    --output_dir reports/rl/SAC \
    --model_dir models/rl/SAC \
    --num_episodes 100 \
    --transaction_cost 0.0001 \
    --trade_penalty 0.01
```

### Backtesting

To run a backtest with a trained model:

```bash
python src/RL/SAC/run_sac_backtest.py \
    --data_path data/cmma_metalabeled_atr_lag5_regime_filtered.csv \
    --model_path models/rl/SAC/best_sac_model.pt \
    --output_dir reports/rl/SAC
```

## Configuration Parameters

The SAC implementation has several parameters that can be tuned:

### Environment Parameters:
- `transaction_cost`: Cost per trade 
- `trade_penalty`: Additional penalty for trading to reduce frequency
- `holding_penalty`: Small penalty for holding positions
- `correct_direction_reward`: Reward for making trades in the same direction as metalabeled strategy
- `min_trades_ratio`: Target minimum trades as a ratio of metalabeled strategy trades

### SAC Parameters:
- `hidden_dims`: Hidden dimensions for networks
- `lr_actor`: Learning rate for actor
- `lr_critic`: Learning rate for critic
- `gamma`: Discount factor
- `tau`: Target network update rate
- `alpha`: Entropy coefficient
- `automatic_entropy_tuning`: Whether to automatically tune entropy coefficient

## Output

All results, metrics, and visualizations are saved to the specified output directory:

- Training metrics plots
- Backtest cumulative returns comparison
- Trade frequency comparison
- Strategy comparison metrics
- Evaluation reports 