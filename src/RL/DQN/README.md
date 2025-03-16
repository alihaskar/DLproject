# Deep Q-Network (DQN) for Trading with Transaction Cost Reduction

This module implements a Deep Q-Network (DQN) for trading with a focus on reducing transaction costs while maintaining performance.

## Overview

The DQN trading agent is designed to:

1. Learn optimal trading strategies from metalabeled market data
2. Minimize transaction costs (0.1 pip per trade)
3. Reduce excessive trading by including a trade penalty in the reward function
4. Provide performance comparisons with the original metalabeled strategy

## Components

The implementation consists of the following components:

- `trading_env.py`: A custom trading environment based on OpenAI Gym that simulates trading with transaction costs
- `dqn_model.py`: The DQN model implementation with experience replay and target network
- `train_dqn.py`: A script for training and evaluating the DQN model
- `run_dqn_backtest.py`: A simple script to run the DQN training and backtesting

## Features

- Transaction cost awareness: The agent is trained with transaction costs included in the reward function
- Trade penalty: Additional penalty for trading to discourage excessive trading
- Experience replay: Memory buffer to store and sample experiences for more efficient learning
- Target network: Separate network for stable learning
- Epsilon-greedy exploration: Balance between exploration and exploitation

## Usage

To train and backtest the DQN agent, simply run:

```bash
python src/RL/DQN/run_dqn_backtest.py
```

This will:

1. Train a DQN agent on the metalabeled data
2. Evaluate its performance on a test set
3. Compare it with the original metalabeled strategy
4. Save results in `reports/rl/dqn`

## Results

The DQN agent is trained to optimize trading decisions while considering transaction costs. The results are saved in the following formats:

1. `dqn_training_progress.png`: A plot showing episode rewards and losses during training
2. `dqn_vs_metalabeled.png`: A comparison of cumulative returns and positions between DQN and metalabeled strategies
3. `dqn_backtest_results.csv`: Detailed backtest results for the DQN strategy
4. `strategy_comparison.csv`: Performance metrics comparing DQN and metalabeled strategies

## Configuration

The DQN agent can be configured with various hyperparameters through command-line arguments in `run_dqn_backtest.py`, including:

- Number of training episodes
- Transaction cost
- Trade penalty
- Network architecture
- Learning rate
- Discount factor
- etc.

## Implementation Details

The agent uses state-of-the-art reinforcement learning techniques:

1. **DQN Architecture**: Uses a neural network to approximate the Q-function
2. **Experience Replay**: Stores and randomly samples past experiences to break correlations between consecutive samples
3. **Target Network**: Uses a separate network for more stable learning
4. **Dueling DQN**: Separates state value and action advantage estimation for more efficient learning
5. **Transaction Cost Awareness**: Incorporates transaction costs directly into the reward function
6. **Trade Frequency Penalty**: Discourages excessive trading to reduce overall costs 