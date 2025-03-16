#!/usr/bin/env python
"""
Run PPO training and backtesting on metalabeled data.
This script provides a simple way to run the PPO training and evaluation.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path for proper imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.RL.PPO.train_ppo import main as train_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run PPO training and backtesting."""
    # Create necessary directories
    os.makedirs('models/rl/ppo', exist_ok=True)
    os.makedirs('reports/rl/ppo', exist_ok=True)
    
    logger.info("Starting PPO training and backtesting...")
    
    # Call training main function with arguments
    sys.argv = [
        'train_ppo.py',
        '--data_path', 'data/cmma_metalabeled_atr_lag5_regime_filtered.csv',
        '--output_dir', 'reports/rl/ppo',
        '--model_dir', 'models/rl/ppo',
        '--num_episodes', '30',  # Adjust based on your computational resources
        '--transaction_cost', '0.0001',  # 0.1 pip as specified
        '--trade_penalty', '0.01',  # Reduced penalty to encourage more selective trading
        '--trade_correction_reward', '0.02',  # Reward for correct directional trades
        '--min_trades_ratio', '0.1',  # Target at least 10% of metalabeled strategy trades
        '--min_trades_penalty', '0.2',  # Penalty for not meeting minimum trades target
        '--save_top_k', '3',  # Save top 3 models
        '--plot_is_oos',  # Plot in-sample and out-of-sample results
        '--test_size', '0.2',  # Use 20% of data for testing
        '--hidden_dims', '128', '64',  # Network architecture
        '--batch_size', '64',  # Batch size for training
        '--gamma', '0.99',  # Discount factor
        '--gae_lambda', '0.95',  # GAE lambda parameter
        '--policy_clip', '0.2',  # PPO clipping parameter
        '--n_epochs', '10',  # Number of epochs per PPO update
        '--lr_actor', '0.0003',  # Actor learning rate
        '--lr_critic', '0.001',  # Critic learning rate
        '--seed', '42'  # Random seed
    ]
    
    # Run training
    train_main()
    
    logger.info("PPO training and backtesting completed!")
    logger.info("Results saved in reports/rl/ppo")

if __name__ == "__main__":
    main() 