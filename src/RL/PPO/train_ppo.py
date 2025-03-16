import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import argparse
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import seaborn as sns

from .trading_env import TradingEnvironment
from .ppo_model import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PPO for Trading')
    
    parser.add_argument('--data_path', type=str, default='data/cmma_metalabeled_atr_lag5_regime_filtered.csv',
                      help='Path to metalabeled data CSV')
    parser.add_argument('--output_dir', type=str, default='reports/rl/ppo',
                      help='Output directory for results')
    parser.add_argument('--model_dir', type=str, default='models/rl/ppo',
                      help='Directory to save model checkpoints')
    parser.add_argument('--num_episodes', type=int, default=100,
                      help='Number of training episodes')
    parser.add_argument('--transaction_cost', type=float, default=0.0001,
                      help='Transaction cost per trade (0.1 pip)')
    parser.add_argument('--trade_penalty', type=float, default=0.01,
                      help='Penalty for trading to reduce frequency')
    parser.add_argument('--trade_correction_reward', type=float, default=0.02,
                      help='Reward for correct directional trades')
    parser.add_argument('--min_trades_ratio', type=float, default=0.1,
                      help='Target minimum trades as a ratio of metalabeled strategy trades')
    parser.add_argument('--min_trades_penalty', type=float, default=0.2,
                      help='Penalty for not meeting minimum trades target')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Proportion of data to use for testing')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                      help='Hidden dimensions of PPO networks')
    parser.add_argument('--batch_size', type=int, default=64, 
                      help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                      help='GAE lambda parameter')
    parser.add_argument('--policy_clip', type=float, default=0.2,
                      help='PPO policy clipping parameter')
    parser.add_argument('--n_epochs', type=int, default=10,
                      help='Number of policy update epochs per episode')
    parser.add_argument('--lr_actor', type=float, default=0.0003,
                      help='Learning rate for actor network')
    parser.add_argument('--lr_critic', type=float, default=0.001,
                      help='Learning rate for critic network')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--save_top_k', type=int, default=3,
                      help='Number of top models to save')
    parser.add_argument('--plot_is_oos', action='store_true',
                      help='Whether to plot in-sample and out-of-sample results')
    
    return parser.parse_args()

def select_features(df):
    """Select relevant features for the PPO model."""
    # Basic features that don't include future information
    feature_cols = [
        # Base features
        'atr_lag1', 'rsi_lag1', 'close_ma_diff_lag1', 
        # Price-based features
        'price_range_lag1', 'high_low_ratio_lag1', 'price_ma_ratio_lag1',
        # Regime information 
        'hmm_downtrend', 'hmm_uptrend',
        'transformer_downtrend', 'transformer_neutral', 'transformer_uptrend',
    ]
    
    # Add some lagged features if available
    for lag in range(1, 4):  # Use lags 1, 2, 3
        for base_feature in ['atr', 'rsi', 'returns']:
            feature_name = f'{base_feature}_lag{lag}'
            if feature_name in df.columns:
                if feature_name not in feature_cols:
                    feature_cols.append(feature_name)
    
    # Validate columns exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")
    return feature_cols

def prepare_data(data_path, test_size=0.2, seed=42):
    """Load and prepare data for training and testing."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert DateTime to datetime and set as index
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
    
    # Split data into train and test sets
    np.random.seed(seed)
    test_idx = int(len(df) * (1 - test_size))
    
    train_data = df.iloc[:test_idx].copy()
    test_data = df.iloc[test_idx:].copy()
    
    logger.info(f"Data split: train={len(train_data)}, test={len(test_data)}")
    
    return train_data, test_data

def train_ppo(args):
    """Train PPO agent on trading data."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Prepare data
    train_data, test_data = prepare_data(args.data_path, args.test_size, args.seed)
    
    # Get feature columns
    feature_cols = select_features(train_data)
    
    # Count original strategy trades
    metalabel_trades = sum(abs(train_data['meta_position'].diff().fillna(0)) != 0)
    min_trades = int(metalabel_trades * args.min_trades_ratio)
    logger.info(f"Original strategy trades: {metalabel_trades}, min trades target: {min_trades}")
    
    # Create trading environment
    env = TradingEnvironment(
        data=train_data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=args.trade_penalty,
        correct_direction_reward=args.trade_correction_reward,
        min_trades_per_episode=min_trades,
        min_trades_penalty=args.min_trades_penalty
    )
    
    # Initialize PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_clip=args.policy_clip,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs
    )
    
    # Set up model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.model_dir, f"ppo_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    best_returns = []
    episode_returns = []
    episode_costs = []
    episode_trades = []
    episode_rewards = []
    episode_losses = []
    
    N_PRINT = max(1, args.num_episodes // 10)  # Print every 10% of episodes
    
    logger.info("Starting PPO training...")
    progress_bar = tqdm(range(args.num_episodes), desc="Training PPO")
    
    for episode in progress_bar:
        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0
        episode_loss = 0
        
        # Run episode
        while not done:
            # Select action
            action, prob, val = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition in PPO memory
            agent.store_transition(state, action, prob, val, reward, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
        # Train the agent after completing the episode
        loss_metrics = agent.train()
        episode_loss = loss_metrics["total_loss"]
        
        # Record episode metrics
        final_portfolio = env.get_portfolio_value()
        episode_return = final_portfolio / env.initial_balance - 1
        episode_returns.append(episode_return)
        episode_costs.append(env.total_costs)
        episode_trades.append(env.trades_executed)
        episode_rewards.append(total_reward)
        episode_losses.append(episode_loss)
        
        # Update progress bar
        progress_bar.set_postfix({
            'return': f"{episode_return:.4f}",
            'trades': env.trades_executed,
            'costs': f"{env.total_costs:.4f}",
            'loss': f"{episode_loss:.4f}"
        })
        
        # Print episode metrics periodically
        if (episode + 1) % N_PRINT == 0 or episode == 0 or episode == args.num_episodes - 1:
            logger.info(f"Episode {episode + 1}/{args.num_episodes}: "
                       f"Return={episode_return:.4f}, "
                       f"Trades={env.trades_executed}, "
                       f"Costs={env.total_costs:.4f}, "
                       f"Reward={total_reward:.4f}, "
                       f"Loss={episode_loss:.4f}")
        
        # Save model if it's in the top k performers
        if len(best_returns) < args.save_top_k or episode_return > min(best_returns):
            model_path = os.path.join(model_dir, f"ppo_episode_{episode + 1}.pt")
            agent.save(model_path)
            logger.info(f"Saved model at episode {episode + 1} with return {episode_return:.4f}")
            
            # Update best returns
            if len(best_returns) < args.save_top_k:
                best_returns.append(episode_return)
            else:
                min_idx = best_returns.index(min(best_returns))
                best_returns[min_idx] = episode_return
    
    # Plot training metrics
    plot_training_metrics(
        episode_returns=episode_returns,
        episode_costs=episode_costs,
        episode_trades=episode_trades,
        episode_rewards=episode_rewards,
        episode_losses=episode_losses,
        output_dir=args.output_dir
    )
    
    # Get top k model paths
    model_files = os.listdir(model_dir)
    model_paths = [os.path.join(model_dir, f) for f in model_files if f.endswith('.pt')]
    model_episodes = [int(f.split('_')[-1].split('.')[0]) for f in model_files if f.endswith('.pt')]
    model_returns = [episode_returns[ep-1] for ep in model_episodes]
    
    top_indices = sorted(range(len(model_returns)), key=lambda i: model_returns[i], reverse=True)[:args.save_top_k]
    top_models = [model_paths[i] for i in top_indices]
    
    # Evaluate top models
    logger.info(f"Evaluating top {len(top_models)} models...")
    all_results = evaluate_top_models(top_models, train_data, test_data, feature_cols, args, min_trades)
    
    # Calculate metalabeled strategy returns
    train_meta_returns = calculate_metalabeled_returns(train_data)
    test_meta_returns = calculate_metalabeled_returns(test_data)
    
    # Plot model comparison
    plot_model_comparison(
        all_results=all_results,
        train_meta_returns=train_meta_returns,
        test_meta_returns=test_meta_returns,
        train_data=train_data,
        test_data=test_data,
        args=args
    )
    
    # Return the best model path
    best_model_idx = all_results['best_model_idx']
    best_model_path = top_models[best_model_idx]
    
    return best_model_path, all_results

def evaluate_ppo(agent_or_path, test_data, feature_cols, args, min_trades=None):
    """Evaluate PPO agent on test data."""
    # Create environment
    env = TradingEnvironment(
        data=test_data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=args.trade_penalty,
        correct_direction_reward=args.trade_correction_reward,
        min_trades_per_episode=min_trades if min_trades is not None else 0,
        min_trades_penalty=args.min_trades_penalty
    )
    
    # If agent_or_path is a string, load the model
    if isinstance(agent_or_path, str):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=args.hidden_dims,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_clip=args.policy_clip
        )
        agent.load(agent_or_path)
    else:
        agent = agent_or_path
    
    # Run evaluation
    return run_evaluation(agent, env, test_data)

def calculate_metalabeled_returns(data):
    """Calculate returns from metalabeled strategy."""
    # Calculate cumulative returns from metalabeled strategy
    returns = []
    positions = data['meta_position'].values
    price_changes = data['returns'].values
    
    for i in range(len(positions)):
        if i > 0:
            ret = positions[i-1] * price_changes[i]
            returns.append(ret)
        else:
            returns.append(0.0)
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + np.array(returns)) - 1
    
    return cum_returns

def evaluate_top_models(top_models, train_data, test_data, feature_cols, args, min_trades=None):
    """Evaluate top models on both training and test data."""
    all_results = {
        'train_returns': [],
        'test_returns': [],
        'train_cum_returns': [],
        'test_cum_returns': [],
        'train_trades': [],
        'test_trades': [],
        'train_costs': [],
        'test_costs': [],
        'best_model_idx': 0
    }
    
    # Calculate sharpe ratios
    train_sharpe_ratios = []
    test_sharpe_ratios = []
    
    for i, model_path in enumerate(top_models):
        logger.info(f"Evaluating model {i+1}/{len(top_models)}: {model_path}")
        
        # Evaluate on training data
        train_results = evaluate_ppo(model_path, train_data, feature_cols, args, min_trades)
        all_results['train_returns'].append(train_results['returns'])
        all_results['train_cum_returns'].append(train_results['cum_returns'])
        all_results['train_trades'].append(train_results['trades_executed'])
        all_results['train_costs'].append(train_results['total_costs'])
        
        # Calculate Sharpe ratio for training data
        returns_mean = np.mean(train_results['returns'])
        returns_std = np.std(train_results['returns']) if np.std(train_results['returns']) > 0 else 1e-6
        train_sharpe = returns_mean / returns_std * np.sqrt(252)  # Annualized Sharpe
        train_sharpe_ratios.append(train_sharpe)
        
        # Evaluate on test data
        test_results = evaluate_ppo(model_path, test_data, feature_cols, args, min_trades)
        all_results['test_returns'].append(test_results['returns'])
        all_results['test_cum_returns'].append(test_results['cum_returns'])
        all_results['test_trades'].append(test_results['trades_executed'])
        all_results['test_costs'].append(test_results['total_costs'])
        
        # Calculate Sharpe ratio for test data
        returns_mean = np.mean(test_results['returns'])
        returns_std = np.std(test_results['returns']) if np.std(test_results['returns']) > 0 else 1e-6
        test_sharpe = returns_mean / returns_std * np.sqrt(252)  # Annualized Sharpe
        test_sharpe_ratios.append(test_sharpe)
        
        logger.info(f"Model {i+1} - Train Sharpe: {train_sharpe:.4f}, Test Sharpe: {test_sharpe:.4f}")
        logger.info(f"Model {i+1} - Train Return: {train_results['cum_returns'][-1]:.4f}, "
                   f"Test Return: {test_results['cum_returns'][-1]:.4f}")
        logger.info(f"Model {i+1} - Train Trades: {train_results['trades_executed']}, "
                   f"Test Trades: {test_results['trades_executed']}")
        
    # Choose best model by test Sharpe ratio
    best_model_idx = np.argmax(test_sharpe_ratios)
    all_results['best_model_idx'] = best_model_idx
    
    logger.info(f"Best model by test Sharpe ratio: {best_model_idx + 1} with Sharpe {test_sharpe_ratios[best_model_idx]:.4f}")
    
    return all_results

def run_evaluation(agent, env, data):
    """Run evaluation of agent on environment."""
    # Reset environment
    state = env.reset()
    done = False
    
    all_returns = []
    all_positions = []
    all_actions = []
    all_prices = []
    
    # Run episode
    while not done:
        # Select action
        action, _, _ = agent.select_action(state, training=False)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Record information
        all_returns.append(info['net_return'])
        all_positions.append(info['position'])
        all_actions.append(action)
        all_prices.append(info['price'])
        
        # Update state
        state = next_state
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + np.array(all_returns)) - 1
    
    # Get trade history
    trade_history = env.get_trade_history()
    
    # Results
    results = {
        'returns': all_returns,
        'cum_returns': cum_returns,
        'positions': all_positions,
        'actions': all_actions,
        'prices': all_prices,
        'trade_history': trade_history,
        'final_return': cum_returns[-1] if len(cum_returns) > 0 else 0.0,
        'trades_executed': env.trades_executed,
        'total_costs': env.total_costs,
        'correct_trades': env.correct_trades if hasattr(env, 'correct_trades') else 0
    }
    
    return results

def plot_training_metrics(episode_returns, episode_costs, episode_trades, episode_rewards, 
                          episode_losses, output_dir):
    """Plot training metrics."""
    plt.figure(figsize=(15, 12))
    
    # Plot episode returns
    plt.subplot(3, 2, 1)
    plt.plot(episode_returns)
    plt.title('Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    
    # Plot episode costs
    plt.subplot(3, 2, 2)
    plt.plot(episode_costs)
    plt.title('Episode Transaction Costs')
    plt.xlabel('Episode')
    plt.ylabel('Costs')
    plt.grid(True)
    
    # Plot episode trades
    plt.subplot(3, 2, 3)
    plt.plot(episode_trades)
    plt.title('Episode Trades')
    plt.xlabel('Episode')
    plt.ylabel('Number of Trades')
    plt.grid(True)
    
    # Plot episode rewards
    plt.subplot(3, 2, 4)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot episode losses
    plt.subplot(3, 2, 5)
    plt.plot(episode_losses)
    plt.title('Episode Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot returns vs trades scatter
    plt.subplot(3, 2, 6)
    plt.scatter(episode_trades, episode_returns)
    plt.title('Returns vs Trades')
    plt.xlabel('Number of Trades')
    plt.ylabel('Return')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300)
    plt.close()
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, len(episode_returns) + 1),
        'return': episode_returns,
        'costs': episode_costs,
        'trades': episode_trades,
        'rewards': episode_rewards,
        'losses': episode_losses
    })
    metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
    
    logger.info(f"Training metrics saved to {output_dir}")

def plot_model_comparison(all_results, train_meta_returns, test_meta_returns, 
                         train_data, test_data, args):
    """Plot comparison of models and metalabeled strategy."""
    # Get best model
    best_model_idx = all_results['best_model_idx']
    
    # Create figure
    plt.figure(figsize=(20, 12))
    
    # Plot cumulative returns
    plt.subplot(2, 2, 1)
    
    # Plot training returns
    plt.plot(train_meta_returns, 'k-', label='Meta-labeled (Train)')
    for i, cum_returns in enumerate(all_results['train_cum_returns']):
        if i == best_model_idx:
            plt.plot(cum_returns, 'r-', linewidth=2, label=f'PPO Best Model (Train)')
        else:
            plt.plot(cum_returns, 'b-', alpha=0.3)
    
    plt.title('Cumulative Returns (Training)')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Return')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True)
    plt.legend()
    
    # Plot test returns
    plt.subplot(2, 2, 2)
    plt.plot(test_meta_returns, 'k-', label='Meta-labeled (Test)')
    for i, cum_returns in enumerate(all_results['test_cum_returns']):
        if i == best_model_idx:
            plt.plot(cum_returns, 'r-', linewidth=2, label=f'PPO Best Model (Test)')
        else:
            plt.plot(cum_returns, 'b-', alpha=0.3)
    
    plt.title('Cumulative Returns (Test)')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Return')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True)
    plt.legend()
    
    # Plot trade count comparison
    plt.subplot(2, 2, 3)
    
    meta_train_trades = sum(abs(train_data['meta_position'].diff().fillna(0)) != 0)
    meta_test_trades = sum(abs(test_data['meta_position'].diff().fillna(0)) != 0)
    
    train_trades = [all_results['train_trades'][i] for i in range(len(all_results['train_trades']))]
    test_trades = [all_results['test_trades'][i] for i in range(len(all_results['test_trades']))]
    
    model_labels = [f'Model {i+1}' for i in range(len(train_trades))]
    best_label = f'Best (Model {best_model_idx+1})'
    
    # Highlight best model
    colors = ['blue' if i != best_model_idx else 'red' for i in range(len(train_trades))]
    
    # Create DataFrame for grouped bar chart
    trade_data = pd.DataFrame({
        'Model': model_labels + ['Meta-labeled'],
        'Train': train_trades + [meta_train_trades],
        'Test': test_trades + [meta_test_trades]
    })
    
    # Melt DataFrame for plotting
    trade_data_melted = pd.melt(trade_data, id_vars='Model', var_name='Dataset', value_name='Trades')
    
    # Create grouped bar chart
    sns.barplot(x='Model', y='Trades', hue='Dataset', data=trade_data_melted)
    plt.title('Number of Trades Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Plot final returns comparison
    plt.subplot(2, 2, 4)
    
    train_final_returns = [all_results['train_cum_returns'][i][-1] for i in range(len(all_results['train_cum_returns']))]
    test_final_returns = [all_results['test_cum_returns'][i][-1] for i in range(len(all_results['test_cum_returns']))]
    
    # Create DataFrame for grouped bar chart
    return_data = pd.DataFrame({
        'Model': model_labels + ['Meta-labeled'],
        'Train': train_final_returns + [train_meta_returns[-1]],
        'Test': test_final_returns + [test_meta_returns[-1]]
    })
    
    # Melt DataFrame for plotting
    return_data_melted = pd.melt(return_data, id_vars='Model', var_name='Dataset', value_name='Return')
    
    # Create grouped bar chart
    sns.barplot(x='Model', y='Return', hue='Dataset', data=return_data_melted)
    plt.title('Final Returns Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # Create a detailed results table
    results_data = []
    for i in range(len(train_trades)):
        results_data.append({
            'Model': f'Model {i+1}',
            'Best': i == best_model_idx,
            'Train Return': train_final_returns[i],
            'Test Return': test_final_returns[i],
            'Train Trades': train_trades[i],
            'Test Trades': test_trades[i],
            'Train Costs': all_results['train_costs'][i],
            'Test Costs': all_results['test_costs'][i],
            'Train Trades Reduction': 1 - train_trades[i] / meta_train_trades,
            'Test Trades Reduction': 1 - test_trades[i] / meta_test_trades,
        })
    
    # Add metalabeled strategy
    results_data.append({
        'Model': 'Meta-labeled',
        'Best': False,
        'Train Return': train_meta_returns[-1],
        'Test Return': test_meta_returns[-1],
        'Train Trades': meta_train_trades,
        'Test Trades': meta_test_trades,
        'Train Costs': meta_train_trades * args.transaction_cost,
        'Test Costs': meta_test_trades * args.transaction_cost,
        'Train Trades Reduction': 0.0,
        'Test Trades Reduction': 0.0,
    })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(args.output_dir, 'model_comparison_results.csv'), index=False)
    
    # If plot_is_oos flag is set, plot in-sample and out-of-sample returns
    if args.plot_is_oos:
        plot_is_oos_returns(
            train_data=train_data,
            test_data=test_data,
            ppo_train_returns=all_results['train_cum_returns'][best_model_idx],
            ppo_test_returns=all_results['test_cum_returns'][best_model_idx],
            meta_train_returns=train_meta_returns,
            meta_test_returns=test_meta_returns,
            output_dir=args.output_dir
        )
    
    logger.info(f"Model comparison results saved to {args.output_dir}")

def plot_is_oos_returns(train_data, test_data, ppo_train_returns, ppo_test_returns,
                       meta_train_returns, meta_test_returns, output_dir):
    """Plot in-sample and out-of-sample returns."""
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create a combined date index
    train_dates = train_data.index.tolist()
    test_dates = test_data.index.tolist()
    all_dates = train_dates + test_dates
    
    # Make sure lengths match for training data
    if len(ppo_train_returns) < len(train_dates):
        train_dates = train_dates[:len(ppo_train_returns)]
    if len(meta_train_returns) < len(train_dates):
        train_dates = train_dates[:len(meta_train_returns)]
    
    # Make sure lengths match for test data
    if len(ppo_test_returns) < len(test_dates):
        test_dates = test_dates[:len(ppo_test_returns)]
    if len(meta_test_returns) < len(test_dates):
        test_dates = test_dates[:len(meta_test_returns)]
    
    # Truncate returns to match dates
    ppo_train_returns = ppo_train_returns[:len(train_dates)]
    meta_train_returns = meta_train_returns[:len(train_dates)]
    ppo_test_returns = ppo_test_returns[:len(test_dates)]
    meta_test_returns = meta_test_returns[:len(test_dates)]
    
    # Combine returns
    ppo_all_returns = list(ppo_train_returns) + list(ppo_test_returns)
    meta_all_returns = list(meta_train_returns) + list(meta_test_returns)
    
    # Combine dates (now with matching lengths)
    all_dates = train_dates + test_dates
    
    # Ensure same length for all arrays
    min_len = min(len(all_dates), len(ppo_all_returns), len(meta_all_returns))
    all_dates = all_dates[:min_len]
    ppo_all_returns = ppo_all_returns[:min_len]
    meta_all_returns = meta_all_returns[:min_len]
    
    # Plot returns
    plt.plot(all_dates, ppo_all_returns, 'b-', label='PPO Strategy')
    plt.plot(all_dates, meta_all_returns, 'k-', label='Meta-labeled Strategy')
    
    # Add vertical line at train/test split
    if train_dates:
        plt.axvline(x=train_dates[-1], color='r', linestyle='--', 
                    label='Train/Test Split')
    
    # Add in-sample and out-of-sample labels
    if train_dates and meta_all_returns:
        mid_train_idx = len(train_dates)//2
        if mid_train_idx < len(train_dates):
            mid_train = train_dates[mid_train_idx]
            max_y = max(max(ppo_all_returns), max(meta_all_returns))
            plt.text(mid_train, max_y * 0.9, 'In-Sample', 
                    fontsize=12, ha='center', backgroundcolor='white')
    
    if test_dates and meta_all_returns:
        mid_test_idx = len(test_dates)//2
        if mid_test_idx < len(test_dates):
            mid_test = test_dates[mid_test_idx]
            max_y = max(max(ppo_all_returns), max(meta_all_returns))
            plt.text(mid_test, max_y * 0.9, 'Out-of-Sample', 
                    fontsize=12, ha='center', backgroundcolor='white')
    
    plt.title('Cumulative Returns - In-Sample vs. Out-of-Sample')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'is_oos_returns.png'), dpi=300)
    plt.close()
    
    logger.info(f"In-sample and out-of-sample returns plot saved to {output_dir}")

def main():
    """Main function to run PPO training and evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Train PPO agent
    best_model_path, results = train_ppo(args)
    
    logger.info(f"PPO training and evaluation completed! Best model: {best_model_path}")

if __name__ == "__main__":
    main() 