#!/usr/bin/env python
"""
Run DQN training and backtesting on metalabeled data.
This script provides a simple way to run the DQN training and evaluation.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path for proper imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.RL.DQN.train_dqn import main as train_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run DQN training and backtesting."""
    # Create necessary directories
    os.makedirs('models/rl/dqn', exist_ok=True)
    os.makedirs('reports/rl/dqn', exist_ok=True)
    
    logger.info("Starting DQN training and backtesting...")
    
    # Call training main function with arguments
    sys.argv = [
        'train_dqn.py',
        '--data_path', 'data/cmma_metalabeled_atr_lag5_regime_filtered.csv',
        '--output_dir', 'reports/rl/dqn',
        '--model_dir', 'models/rl/dqn',
        '--num_episodes', '20',  # Fewer episodes for quicker results
        '--transaction_cost', '0.0001',  # 0.1 pip as specified
        '--trade_penalty', '0.01',  # Reduced from 0.05 to encourage more trading
        '--trade_correction_reward', '0.02',  # Reward for correct directional trades
        '--min_trades_ratio', '0.1',  # Target at least 10% of metalabeled strategy trades
        '--min_trades_penalty', '0.2',  # Penalty for not meeting minimum trades target
        '--save_top_k', '3',  # Save top 3 models
        '--plot_is_oos', 'True',  # Plot in-sample and out-of-sample results
        '--test_size', '0.2',
        '--hidden_dims', '128', '64',
        '--batch_size', '64',
        '--gamma', '0.99',
        '--lr', '0.001',
        '--seed', '42'
    ]
    
    # Run training
    train_main()
    
    logger.info("DQN training and backtesting completed!")
    logger.info("Results saved in reports/rl/dqn")

if __name__ == "__main__":
    main() 