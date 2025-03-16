import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from .trading_env import TradingEnvironment
from .sac_model import SACAgent
from .utils import (
    prepare_data, select_features, evaluate_sac, plot_results, 
    compare_with_metalabeled, plot_training_metrics, 
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
    parser = argparse.ArgumentParser(description='Train SAC for Trading with Transaction Cost Reduction')
    
    parser.add_argument('--data_path', type=str, default='data/cmma_metalabeled_atr_lag5_regime_filtered.csv',
                      help='Path to metalabeled data CSV')
    parser.add_argument('--output_dir', type=str, default='reports/rl/SAC',
                      help='Output directory for results')
    parser.add_argument('--model_dir', type=str, default='models/rl/SAC',
                      help='Directory to save model checkpoints')
    parser.add_argument('--num_episodes', type=int, default=100,
                      help='Number of training episodes')
    parser.add_argument('--transaction_cost', type=float, default=0.0001,
                      help='Transaction cost per trade (0.1 pip)')
    parser.add_argument('--trade_penalty', type=float, default=0.01,
                      help='Penalty for trading to reduce frequency')
    parser.add_argument('--holding_penalty', type=float, default=0.001,
                      help='Small penalty for holding positions over time')
    parser.add_argument('--trade_correction_reward', type=float, default=0.03,
                      help='Reward for correct directional trades')
    parser.add_argument('--min_trades_ratio', type=float, default=0.1,
                      help='Target minimum trades as a ratio of metalabeled strategy trades')
    parser.add_argument('--min_trades_penalty', type=float, default=0.2,
                      help='Penalty for not meeting minimum trades target')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256],
                      help='Hidden dimensions for SAC networks')
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                      help='Learning rate for actor network')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                      help='Learning rate for critic networks')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for SAC training')
    parser.add_argument('--buffer_capacity', type=int, default=100000,
                      help='Replay buffer capacity')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor for future rewards')
    parser.add_argument('--tau', type=float, default=0.005,
                      help='Target network update rate')
    parser.add_argument('--alpha', type=float, default=0.2,
                      help='Initial entropy coefficient')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True,
                      help='Whether to automatically tune entropy coefficient')
    parser.add_argument('--steps_per_episode', type=int, default=None,
                      help='Maximum steps per episode (None for entire dataset)')
    parser.add_argument('--eval_frequency', type=int, default=5,
                      help='Evaluate model every N episodes')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    return parser.parse_args()

def train_sac(args):
    """Train the SAC agent."""
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare data
    train_data, test_data, feature_cols, meta_trades = prepare_data(
        args.data_path, 
        test_size=args.test_size, 
        seed=args.seed
    )
    
    # Calculate minimum trades target
    min_trades = max(1, int(meta_trades * args.min_trades_ratio))
    logger.info(f"Setting minimum trades target to {min_trades}")
    
    # Create environment for training
    env = TradingEnvironment(
        data=train_data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=args.trade_penalty,
        holding_penalty=args.holding_penalty,
        correct_direction_reward=args.trade_correction_reward,
        min_trades_per_episode=min_trades,
        min_trades_penalty=args.min_trades_penalty,
        target_trade_frequency=args.min_trades_ratio
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        automatic_entropy_tuning=args.automatic_entropy_tuning,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        device=device
    )
    
    # Training loop
    best_reward = -np.inf
    best_model_path = os.path.join(args.model_dir, 'best_sac_model.pt')
    
    # Arrays to store metrics
    episode_rewards = []
    episode_trades = []
    episode_costs = []
    episode_returns = []
    episode_sharpes = []
    actor_losses = []
    critic_losses = []
    alpha_losses = []
    
    logger.info(f"Starting training for {args.num_episodes} episodes")
    
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Set max steps for this episode
        max_steps = len(train_data) if args.steps_per_episode is None else min(args.steps_per_episode, len(train_data))
        
        # Store losses for this episode
        ep_actor_losses = []
        ep_critic_losses = []
        ep_alpha_losses = []
        
        # Run episode
        pbar = tqdm(total=max_steps, desc=f"Episode {episode+1}/{args.num_episodes}")
        
        while not done and episode_steps < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            actor_loss, critic_loss, alpha_loss = agent.train()
            
            # Record losses
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)
            ep_alpha_losses.append(alpha_loss)
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            
            # Update state
            state = next_state
            
            # Update progress bar
            pbar.update(1)
        
        pbar.close()
        
        # Calculate episode metrics
        results = env.get_results()
        returns = np.array(results['returns'])
        total_return = np.sum(returns)
        
        # Calculate Sharpe ratio (if possible)
        if len(returns) > 0 and np.std(returns) != 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_trades.append(results['trade_count'])
        episode_costs.append(results['transaction_costs'])
        episode_returns.append(total_return)
        episode_sharpes.append(sharpe)
        
        # Store average losses
        if ep_actor_losses:
            actor_losses.append(np.mean(ep_actor_losses))
        if ep_critic_losses:
            critic_losses.append(np.mean(ep_critic_losses))
        if ep_alpha_losses:
            alpha_losses.append(np.mean(ep_alpha_losses))
        
        # Log progress
        logger.info(f"Episode {episode+1}: Reward={episode_reward:.4f}, Return={total_return:.4f}, "
                   f"Trades={results['trade_count']}, Costs={results['transaction_costs']:.4f}, "
                   f"Sharpe={sharpe:.2f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(best_model_path)
            logger.info(f"New best model saved with reward {best_reward:.4f}")
        
        # Periodically evaluate on test data
        if (episode + 1) % args.eval_frequency == 0 or episode == args.num_episodes - 1:
            logger.info(f"Evaluating model after episode {episode+1}")
            eval_results = evaluate_sac(agent, test_data, feature_cols, args, min_trades)
            
            # Save evaluation results
            eval_path = os.path.join(args.output_dir, f'eval_episode_{episode+1}.json')
            with open(eval_path, 'w') as f:
                # Use the custom JSON encoder to handle Timestamp objects
                json_safe_results = prepare_results_for_json(eval_results)
                json.dump(json_safe_results, f, indent=4)
            
            # Plot evaluation results
            plot_path = os.path.join(args.output_dir, f'eval_episode_{episode+1}.png')
            plot_results(eval_results, train_data, test_data, plot_path)
    
    # Save training metrics
    metrics = {
        'rewards': episode_rewards,
        'trades': episode_trades,
        'costs': episode_costs,
        'returns': episode_returns,
        'sharpes': episode_sharpes,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'alpha_losses': alpha_losses
    }
    
    metrics_path = os.path.join(args.output_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, cls=PandasJSONEncoder, indent=4)
    
    # Plot training metrics
    plot_training_metrics(metrics, os.path.join(args.output_dir, 'training_metrics.png'))
    
    # Final evaluation with best model
    logger.info("Performing final evaluation with best model")
    agent.load(best_model_path)
    final_results = evaluate_sac(agent, test_data, feature_cols, args, min_trades)
    
    # Save final evaluation results
    final_path = os.path.join(args.output_dir, 'final_evaluation.json')
    with open(final_path, 'w') as f:
        # Use the custom JSON encoder to handle Timestamp objects
        json_safe_results = prepare_results_for_json(final_results)
        json.dump(json_safe_results, f, indent=4)
    
    # Plot final results
    plot_path = os.path.join(args.output_dir, 'final_evaluation.png')
    plot_results(final_results, train_data, test_data, plot_path)
    
    # Compare with metalabeled strategy
    compare_with_metalabeled(final_results, train_data, test_data, args)
    
    return agent, final_results

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    
    # Train model
    agent, results = train_sac(args)
    
    # Log final results
    logger.info(f"Final results: Total return={results['total_return']:.4f}, "
               f"Sharpe ratio={results['sharpe_ratio']:.2f}, "
               f"Max drawdown={results['max_drawdown']:.4f}, "
               f"Trade count={results['trade_count']}, "
               f"Transaction costs={results['transaction_costs']:.4f}")
    
    logger.info("Training and evaluation completed")
    
    # Automatically run backtest
    logger.info("Starting automatic backtest...")
    
    # Import backtest function here to avoid circular dependency
    from src.RL.SAC.run_sac_backtest import backtest_sac
    
    # Create a namespace with the same args structure for backtest
    backtest_args = argparse.Namespace(
        data_path=args.data_path,
        model_path=os.path.join(args.model_dir, 'best_sac_model.pt'),
        output_dir=args.output_dir,
        test_size=args.test_size,
        transaction_cost=args.transaction_cost,
        hidden_dims=args.hidden_dims,
        seed=args.seed,
        # Add the original state_dim to ensure consistent feature handling
        original_state_dim=agent.state_dim if hasattr(agent, 'state_dim') else None
    )
    
    # Run backtest
    backtest_results = backtest_sac(backtest_args)
    
    logger.info("Backtest completed automatically")
    logger.info(f"Backtest results: Total return={backtest_results['total_return']:.4f}, "
               f"Sharpe ratio={backtest_results['sharpe_ratio']:.2f}, "
               f"Trade count={backtest_results['trade_count']}, "
               f"Transaction costs={backtest_results['transaction_costs']:.4f}")

if __name__ == "__main__":
    main() 