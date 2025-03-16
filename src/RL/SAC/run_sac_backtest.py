import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add the project root to the path for proper imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.RL.SAC.trading_env import TradingEnvironment
from src.RL.SAC.sac_model import SACAgent
from src.RL.SAC.utils import (
    prepare_data, select_features, evaluate_sac, plot_results, 
    compare_with_metalabeled, calculate_metalabeled_returns,
    prepare_results_for_json, PandasJSONEncoder
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest SAC for Trading')
    
    parser.add_argument('--data_path', type=str, default='data/cmma_metalabeled_atr_lag5_regime_filtered.csv',
                      help='Path to metalabeled data CSV')
    parser.add_argument('--model_path', type=str, default='models/rl/SAC/best_sac_model.pt',
                      help='Path to saved SAC model')
    parser.add_argument('--output_dir', type=str, default='reports/rl/SAC',
                      help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--transaction_cost', type=float, default=0.0001,
                      help='Transaction cost per trade (0.1 pip)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256],
                      help='Hidden dimensions for SAC networks')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def backtest_sac(args):
    """Backtest the SAC agent."""
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare data
    train_data, test_data, feature_cols, _ = prepare_data(
        args.data_path, 
        test_size=args.test_size, 
        seed=args.seed
    )
    
    # Get state dimension from either provided parameter, model checkpoint, or feature count
    if hasattr(args, 'original_state_dim') and args.original_state_dim is not None:
        state_dim = args.original_state_dim
        logger.info(f"Using provided original_state_dim={state_dim}")
    else:
        # Try to get from checkpoint
        try:
            checkpoint = torch.load(args.model_path)
            if 'state_dim' in checkpoint:
                state_dim = checkpoint['state_dim']
                logger.info(f"Loaded state_dim={state_dim} from model checkpoint")
            else:
                # Use the first layer's weight shape to determine the input size
                state_dim = checkpoint['actor']['layers.0.weight'].shape[1]
                logger.info(f"Inferred state_dim={state_dim} from model weights")
        except Exception as e:
            logger.warning(f"Could not determine state_dim from checkpoint: {e}")
            # Fallback to using the feature columns
            state_dim = len(feature_cols)
            logger.info(f"Using feature count as state_dim={state_dim}")
    
    action_dim = 1  # Position size
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        device=device
    )
    
    # Load the best model
    logger.info(f"Loading model from {args.model_path}")
    agent.load(args.model_path)
    
    # Run backtest on test data
    logger.info("Running backtest on test data")
    test_results = evaluate_sac(agent, test_data, feature_cols, args)
    
    # Save test results
    backtest_path = os.path.join(args.output_dir, 'backtest_results.json')
    import json
    with open(backtest_path, 'w') as f:
        # Use the custom JSON encoder to handle Timestamp objects
        json_safe_results = prepare_results_for_json(test_results)
        json.dump(json_safe_results, f, indent=4)
    
    # Plot backtest results
    plot_path = os.path.join(args.output_dir, 'backtest_results.png')
    plot_results(test_results, train_data, test_data, plot_path)
    
    # Compare with metalabeled strategy
    logger.info("Comparing with original metalabeled strategy")
    compare_with_metalabeled(test_results, train_data, test_data, args)
    
    # Log performance metrics
    logger.info(f"Backtest Results:")
    logger.info(f"  Total Return: {test_results['total_return']:.4f}")
    logger.info(f"  Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {test_results['max_drawdown']:.4f}")
    logger.info(f"  Win Rate: {test_results['win_rate']:.2f}")
    logger.info(f"  Number of Trades: {test_results['trade_count']}")
    logger.info(f"  Transaction Costs: {test_results['transaction_costs']:.4f}")
    
    # Calculate trade efficiency
    if test_results['trade_count'] > 0:
        logger.info(f"  Return per Trade: {test_results['total_return'] / test_results['trade_count']:.6f}")
        logger.info(f"  Cost per Trade: {test_results['transaction_costs'] / test_results['trade_count']:.6f}")
    
    # Compare with buy and hold if price data is available
    if 'close' in test_data.columns:
        price_start = test_data['close'].iloc[0]
        price_end = test_data['close'].iloc[-1]
        buy_hold_return = (price_end / price_start) - 1
        logger.info(f"  Buy & Hold Return: {buy_hold_return:.4f}")
        logger.info(f"  Excess Return vs Buy & Hold: {test_results['total_return'] - buy_hold_return:.4f}")
    
    return test_results

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Run backtest
    backtest_sac(args)

if __name__ == "__main__":
    main() 