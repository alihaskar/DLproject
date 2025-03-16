# Deep Reinforcement Learning Trading System

A comprehensive Python trading system that combines:
- Market regime detection (trend, mean reversion, volatile)
- Strategy metalabeling
- Deep Reinforcement Learning (DQN, PPO, SAC)

## Installation

This project uses Poetry for dependency management. To install dependencies:

```bash
poetry install
```

## Usage

### From Command Line

Run the entire system or specific components:

```bash
# Run everything (regimes, metalabeling, and all RL algorithms)
poetry run python run_me.py --data_path data/cmma.csv

# Run only regime detection
poetry run python run_me.py --data_path data/cmma.csv --mode regimes

# Run only metalabeling
poetry run python run_me.py --data_path data/cmma.csv --mode metalabel

# Run specific RL algorithm (dqn, ppo, or sac)
poetry run python run_me.py --data_path data/cmma.csv --mode dqn
```

### From Python

```python
from src.drl import DRL

# Initialize the system
drl = DRL(data_path='data/cmma.csv')

# Get market regimes
regimes_df = drl.regimes()

# Get metalabels
metalabel_df = drl.metalabel()

# Run RL algorithms
drl.rl.dqn()  # Deep Q-Network
drl.rl.ppo()  # Proximal Policy Optimization
drl.rl.sac()  # Soft Actor-Critic
```

## Input Format

The CSV file should contain:
- DateTime: timestamp
- Open/Close/High/Low price columns
- position: trading position 
- stg: strategy returns

## Output

### Regimes Detection
- feature_regime: Rule-based detection
- hmm_regime: Hidden Markov Model detection
- transformer_regime: Transformer-based detection

Regime types:
- uptrend
- downtrend
- mean_reversion
- volatile
- neutral

### Metalabeling
DataFrame with strategy signals enhanced with ML predictions

### RL Models
Trained models and performance metrics for each algorithm (DQN, PPO, SAC)
