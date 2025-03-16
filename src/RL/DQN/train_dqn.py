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
from tqdm import tqdm

# Add the project root to the path for proper imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from .trading_env import TradingEnvironment
from .dqn_model import DQNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN for Trading')
    
    parser.add_argument('--data_path', type=str, default='data/cmma_metalabeled_atr_lag5_regime_filtered.csv',
                      help='Path to metalabeled data CSV')
    parser.add_argument('--output_dir', type=str, default='reports/rl/dqn',
                      help='Output directory for results')
    parser.add_argument('--model_dir', type=str, default='models/rl/dqn',
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
                      help='Hidden dimensions of DQN')
    parser.add_argument('--batch_size', type=int, default=64, 
                      help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--save_top_k', type=int, default=3,
                      help='Number of top models to save')
    parser.add_argument('--plot_is_oos', action='store_true',
                      help='Whether to plot in-sample and out-of-sample results')
    
    return parser.parse_args()

def select_features(df):
    """Select relevant features for the DQN model."""
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
    
    # Fill NaN values
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Select features
    feature_cols = select_features(df)
    
    # Split data into train and test
    np.random.seed(seed)
    train_size = int(len(df) * (1 - test_size))
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    
    logger.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Calculate minimum trades target based on metalabeled strategy
    meta_trades = len(df[df['meta_position'].diff() != 0])
    logger.info(f"Original metalabeled strategy made {meta_trades} trades")
    
    return train_data, test_data, feature_cols, meta_trades

def train_dqn(args):
    """Train the DQN agent."""
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load and prepare data
    train_data, test_data, feature_cols, meta_trades = prepare_data(
        args.data_path, args.test_size, args.seed
    )
    
    # Calculate minimum trades target
    min_trades = int(meta_trades * args.min_trades_ratio)
    logger.info(f"Setting minimum trades target to {min_trades} ({args.min_trades_ratio*100:.1f}% of original)")
    
    # Create training environment
    train_env = TradingEnvironment(
        data=train_data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=args.trade_penalty,
        correct_direction_reward=args.trade_correction_reward,
        min_trades_per_episode=min_trades,
        min_trades_penalty=args.min_trades_penalty,
        target_trade_frequency=args.min_trades_ratio
    )
    
    # Initialize agent
    state_dim = len(feature_cols) + 3  # +3 for position, balance, and trade count
    action_dim = train_env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        # Start with higher exploration to find good trading opportunities
        epsilon_start=1.0,
        epsilon_end=0.05,  # Higher than default to maintain some exploration
        epsilon_decay=0.98,  # Slower decay
    )
    
    # Training loop
    total_rewards = []
    avg_losses = []
    trade_counts = []
    best_reward = -float('inf')
    
    # Track top K models
    top_models = []  # List of (adjusted_reward, episode, model_path) tuples
    
    logger.info(f"Starting training for {args.num_episodes} episodes")
    for episode in range(args.num_episodes):
        state = train_env.reset()
        done = False
        episode_reward = 0
        episode_losses = []
        
        # Run one episode
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, info = train_env.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            if loss > 0:
                episode_losses.append(loss)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Log episode results
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        total_rewards.append(episode_reward)
        avg_losses.append(avg_loss)
        trade_counts.append(train_env.trades_executed)
        
        # Calculate percentage of correct trades
        correct_trade_pct = (train_env.correct_trades / max(1, train_env.trades_executed)) * 100
        
        logger.info(f"Episode {episode+1}/{args.num_episodes}, "
                   f"Reward: {episode_reward:.4f}, "
                   f"Avg Loss: {avg_loss:.4f}, "
                   f"Trades: {train_env.trades_executed}/{min_trades}, "
                   f"Correct Trades: {train_env.correct_trades} ({correct_trade_pct:.1f}%), "
                   f"Costs: {train_env.total_costs:.4f}, "
                   f"Final Balance: {train_env.balance:.2f}, "
                   f"Epsilon: {agent.epsilon:.4f}")
        
        # Save best model based on both reward and minimum trades
        # We want to encourage models that trade enough while having good rewards
        trade_ratio = min(1.0, train_env.trades_executed / min_trades)
        adjusted_reward = episode_reward * (0.5 + 0.5 * trade_ratio)  # Weight by trade ratio
        
        # Check if this model is good enough to be in top K
        if train_env.trades_executed >= min_trades * 0.5:
            # Create model path for this episode
            model_path = os.path.join(args.model_dir, f'dqn_episode_{episode+1}.pt')
            
            # Update top models list
            if len(top_models) < args.save_top_k or adjusted_reward > top_models[-1][0]:
                # Save the model
                agent.save(model_path)
                
                # Add to top models
                top_models.append((adjusted_reward, episode+1, model_path))
                
                # Sort by adjusted reward (descending)
                top_models.sort(reverse=True)
                
                # Keep only top K models
                if len(top_models) > args.save_top_k:
                    # Remove the worst model from disk if it's not the current one
                    worst_model_path = top_models[-1][2]
                    if worst_model_path != model_path and os.path.exists(worst_model_path):
                        os.remove(worst_model_path)
                    
                    # Remove from list
                    top_models.pop()
                
                # Log if this is the best model so far
                if adjusted_reward > best_reward:
                    best_reward = adjusted_reward
                    logger.info(f"New best model with adjusted reward {adjusted_reward:.4f} "
                               f"(raw reward: {episode_reward:.4f}, trade ratio: {trade_ratio:.2f})")
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'dqn_final.pt')
    agent.save(final_model_path)
    
    logger.info(f"Top {len(top_models)} models:")
    for i, (reward, ep, path) in enumerate(top_models):
        logger.info(f"  {i+1}. Episode {ep}: reward={reward:.4f}, path={path}")
    
    # Plot training progress
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(total_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(avg_losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.subplot(2, 2, 3)
    plt.plot(trade_counts)
    plt.axhline(y=min_trades, color='r', linestyle='--', label=f'Target ({min_trades})')
    plt.title('Number of Trades per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Trades')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.scatter(trade_counts, total_rewards)
    plt.axvline(x=min_trades, color='r', linestyle='--', label=f'Target Trades ({min_trades})')
    plt.title('Reward vs Number of Trades')
    plt.xlabel('Number of Trades')
    plt.ylabel('Episode Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'dqn_training_progress.png'))
    
    # Evaluate best model on test data
    logger.info("Evaluating top models on test data")
    best_model_path = top_models[0][2] if top_models else final_model_path
    evaluate_dqn(best_model_path, test_data, feature_cols, args, min_trades)
    
    # If requested, evaluate all top models together
    if args.save_top_k > 1:
        evaluate_top_models(top_models, train_data, test_data, feature_cols, args, min_trades)
    
    return agent, train_data, test_data, feature_cols, top_models

def evaluate_dqn(agent_or_path, test_data, feature_cols, args, min_trades=None):
    """Evaluate the trained DQN agent on test data."""
    # Determine min trades if not provided
    if min_trades is None:
        meta_trades = len(test_data[test_data['meta_position'].diff() != 0])
        min_trades = int(meta_trades * args.min_trades_ratio)
    
    # Create test environment
    test_env = TradingEnvironment(
        data=test_data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=0.0,  # No penalty during evaluation
        correct_direction_reward=0.0,  # No extra reward during evaluation
        min_trades_per_episode=0  # No minimum trades during evaluation
    )
    
    # Load best model if agent is provided as path
    if isinstance(agent_or_path, str):
        state_dim = len(feature_cols) + 3  # +3 for position, balance, trades
        action_dim = 3  # hold, buy, sell
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(agent_or_path)
    else:
        agent = agent_or_path
    
    # Run evaluation
    state = test_env.reset()
    done = False
    
    # Store data for analysis
    actions = []
    positions = []
    rewards = []
    balances = []
    timestamps = []
    trades_executed = 0
    total_costs = 0.0
    
    # Track performance
    while not done:
        # Select action (no exploration)
        action = agent.select_action(state, training=False)
        
        # Take action
        next_state, reward, done, info = test_env.step(action)
        
        # Record data
        actions.append(action)
        positions.append(info['position'])
        rewards.append(reward)
        balances.append(info['balance'])
        timestamps.append(test_data.index[test_env.current_step - 1])
        
        if info['cost'] > 0:
            trades_executed += 1
            total_costs += info['cost']
        
        # Update state
        state = next_state
    
    # Create results dataframe
    results = pd.DataFrame({
        'timestamp': timestamps,
        'action': actions,
        'position': positions,
        'reward': rewards,
        'balance': balances
    })
    
    # Calculate cumulative returns
    initial_balance = test_env.initial_balance
    results['returns'] = (results['balance'] - initial_balance) / initial_balance
    results['cum_returns'] = results['returns'].cumsum()
    
    # Set timestamp as index
    results.set_index('timestamp', inplace=True)
    
    # Compare with metalabeled strategy
    meta_returns = calculate_metalabeled_returns(test_data)
    
    # Plot comparison
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(results.index, results['cum_returns'], label='DQN Strategy')
    plt.plot(meta_returns.index, meta_returns, label='Metalabeled Strategy')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(results.index, results['position'], label='DQN Position')
    plt.plot(test_data.index, test_data['meta_position'], label='Metalabeled Position')
    plt.title('Position Comparison')
    plt.xlabel('Date')
    plt.ylabel('Position (-1=Short, 0=Neutral, 1=Long)')
    plt.legend()
    plt.grid(True)
    
    # Add a plot of trade executions
    position_changes = results['position'].diff() != 0
    meta_position_changes = test_data['meta_position'].diff() != 0
    
    plt.subplot(3, 1, 3)
    plt.scatter(results.index[position_changes], 
                np.zeros(position_changes.sum()) + 0.1, 
                marker='|', s=100, color='blue', label='DQN Trades')
    plt.scatter(test_data.index[meta_position_changes], 
                np.zeros(meta_position_changes.sum()) - 0.1, 
                marker='|', s=100, color='orange', label='Metalabeled Trades')
    plt.title('Trade Execution Comparison')
    plt.xlabel('Date')
    plt.yticks([])
    plt.legend()
    plt.grid(True)
    
    # If we have separate train/test data and plot_is_oos is enabled
    if args.plot_is_oos and 'train_end_date' in args.__dict__:
        # Add vertical line to indicate train/test split
        train_end = args.train_end_date
        for i in range(1, 4):  # For all 3 subplots
            plt.subplot(3, 1, i)
            plt.axvline(x=train_end, color='r', linestyle='--', 
                       label='Train/Test Split' if i == 1 else None)
            if i == 1:  # Only add the legend entry once
                plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'dqn_vs_metalabeled.png'))
    
    # Save detailed results to CSV
    results.to_csv(os.path.join(args.output_dir, 'dqn_backtest_results.csv'))
    
    # Calculate performance metrics
    dqn_final_return = results['cum_returns'].iloc[-1]
    meta_final_return = meta_returns.iloc[-1]
    dqn_trades = trades_executed
    meta_trades = len(test_data[test_data['meta_position'].diff() != 0])
    dqn_costs = total_costs
    
    # Calculate approximate costs for metalabeled strategy
    # This is a rough estimate assuming same transaction cost per trade
    meta_costs = meta_trades * args.transaction_cost * test_data['close'].mean()
    
    # Calculate Sharpe ratio and other metrics
    dqn_daily_returns = results['returns'].resample('D').sum()
    dqn_sharpe = dqn_daily_returns.mean() / max(0.0001, dqn_daily_returns.std()) * np.sqrt(252)
    
    meta_daily_returns = meta_returns.diff().fillna(0).resample('D').sum()
    meta_sharpe = meta_daily_returns.mean() / max(0.0001, meta_daily_returns.std()) * np.sqrt(252)
    
    # Calculate maximum drawdown
    dqn_cum_returns = results['cum_returns']
    dqn_running_max = np.maximum.accumulate(dqn_cum_returns)
    dqn_drawdown = (dqn_cum_returns - dqn_running_max) / (dqn_running_max + 1e-10)
    dqn_max_drawdown = dqn_drawdown.min()
    
    meta_running_max = np.maximum.accumulate(meta_returns)
    meta_drawdown = (meta_returns - meta_running_max) / (meta_running_max + 1e-10)
    meta_max_drawdown = meta_drawdown.min()
    
    # Log comparison
    logger.info("==== Performance Comparison ====")
    logger.info(f"DQN Final Return: {dqn_final_return:.4f}")
    logger.info(f"Metalabeled Final Return: {meta_final_return:.4f}")
    logger.info(f"DQN Number of Trades: {dqn_trades}")
    logger.info(f"Metalabeled Number of Trades: {meta_trades}")
    logger.info(f"DQN Transaction Costs: {dqn_costs:.4f}")
    logger.info(f"Metalabeled Transaction Costs (est.): {meta_costs:.4f}")
    logger.info(f"DQN Sharpe Ratio: {dqn_sharpe:.2f}")
    logger.info(f"Metalabeled Sharpe Ratio: {meta_sharpe:.2f}")
    logger.info(f"DQN Max Drawdown: {dqn_max_drawdown:.2%}")
    logger.info(f"Metalabeled Max Drawdown: {meta_max_drawdown:.2%}")
    
    if dqn_trades > 0:
        transaction_cost_reduction = (meta_costs/meta_trades - dqn_costs/dqn_trades) / (meta_costs/meta_trades)
        logger.info(f"Transaction Cost Reduction per Trade: {transaction_cost_reduction:.2%}")
    else:
        logger.info("Transaction Cost Reduction: N/A (no trades)")
    
    # Save comparison metrics to CSV
    comparison = pd.DataFrame({
        'Metric': ['Final Return', 'Number of Trades', 'Transaction Costs', 
                 'Sharpe Ratio', 'Max Drawdown',
                 'Cost per Trade', 'Return per Trade'],
        'DQN': [dqn_final_return, dqn_trades, dqn_costs, 
               dqn_sharpe, dqn_max_drawdown,
               0 if dqn_trades == 0 else dqn_costs/dqn_trades,
               0 if dqn_trades == 0 else dqn_final_return/dqn_trades],
        'Metalabeled': [meta_final_return, meta_trades, meta_costs, 
                      meta_sharpe, meta_max_drawdown,
                      meta_costs/meta_trades, 
                      meta_final_return/meta_trades]
    })
    comparison.to_csv(os.path.join(args.output_dir, 'strategy_comparison.csv'), index=False)
    
    return results

def calculate_metalabeled_returns(data):
    """Calculate cumulative returns for the metalabeled strategy."""
    # Ensure we have the needed columns
    if 'cum_strategy_returns' in data.columns:
        return data['cum_strategy_returns']
    
    # If we don't have pre-calculated returns, calculate them
    if 'meta_position' in data.columns and 'returns' in data.columns:
        # Calculate returns based on position and market returns
        strategy_returns = data['meta_position'].shift(1) * data['returns']
        cum_returns = strategy_returns.fillna(0).cumsum()
        return cum_returns
    
    # If we can't calculate returns, return zeros
    logger.warning("Could not calculate metalabeled returns, returning zeros")
    return pd.Series(0, index=data.index)

def evaluate_top_models(top_models, train_data, test_data, feature_cols, args, min_trades=None):
    """Evaluate and compare all top models."""
    if not top_models:
        logger.warning("No top models to evaluate")
        return
        
    logger.info(f"Evaluating {len(top_models)} top models together")
    
    # Determine min trades if not provided
    if min_trades is None:
        meta_trades = len(test_data[test_data['meta_position'].diff() != 0])
        min_trades = int(meta_trades * args.min_trades_ratio)
    
    # Create environments
    train_env = TradingEnvironment(
        data=train_data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=0.0,  # No penalty during evaluation
        correct_direction_reward=0.0,  # No extra reward during evaluation
        min_trades_per_episode=0  # No minimum trades during evaluation
    )
    
    test_env = TradingEnvironment(
        data=test_data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=0.0,
        correct_direction_reward=0.0,
        min_trades_per_episode=0
    )
    
    # Prepare for comparison data
    all_results = {}
    state_dim = len(feature_cols) + 3  # +3 for position, balance, trades
    action_dim = 3  # hold, buy, sell
    
    # Get metalabeled strategy returns for comparison
    train_meta_returns = calculate_metalabeled_returns(train_data)
    test_meta_returns = calculate_metalabeled_returns(test_data)
    
    # Evaluate each model on both train and test data
    for i, (reward, episode, model_path) in enumerate(top_models):
        model_name = f"Model {i+1} (Episode {episode})"
        logger.info(f"Evaluating {model_name}")
        
        # Load model
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(model_path)
        
        # Evaluate on training data (in-sample)
        train_results = run_evaluation(agent, train_env, train_data)
        train_results['dataset'] = 'train'
        
        # Evaluate on test data (out-of-sample)
        test_results = run_evaluation(agent, test_env, test_data)
        test_results['dataset'] = 'test'
        
        # Store results for this model
        all_results[model_name] = {
            'train': train_results,
            'test': test_results
        }
    
    # Plot comparison of all models
    plot_model_comparison(all_results, train_meta_returns, test_meta_returns, 
                         train_data, test_data, args)

def run_evaluation(agent, env, data):
    """Run evaluation for a single model on given environment."""
    state = env.reset()
    done = False
    
    # Store data for analysis
    actions = []
    positions = []
    rewards = []
    balances = []
    timestamps = []
    trades_executed = 0
    total_costs = 0.0
    
    # Track performance
    while not done:
        # Select action (no exploration)
        action = agent.select_action(state, training=False)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Record data
        actions.append(action)
        positions.append(info['position'])
        rewards.append(reward)
        balances.append(info['balance'])
        timestamps.append(data.index[env.current_step - 1])
        
        if info['cost'] > 0:
            trades_executed += 1
            total_costs += info['cost']
        
        # Update state
        state = next_state
    
    # Create results dataframe
    results = pd.DataFrame({
        'timestamp': timestamps,
        'action': actions,
        'position': positions,
        'reward': rewards,
        'balance': balances
    })
    
    # Calculate cumulative returns
    initial_balance = env.initial_balance
    results['returns'] = (results['balance'] - initial_balance) / initial_balance
    results['cum_returns'] = results['returns'].cumsum()
    
    # Set timestamp as index
    results.set_index('timestamp', inplace=True)
    
    # Calculate additional metrics
    results['trades_executed'] = trades_executed
    results['total_costs'] = total_costs
    
    return results

def plot_model_comparison(all_results, train_meta_returns, test_meta_returns, 
                         train_data, test_data, args):
    """Plot comparison of all models' performance."""
    plt.figure(figsize=(15, 15))
    
    # Plot cumulative returns
    plt.subplot(3, 1, 1)
    
    # Plot training period (in-sample)
    all_train_returns = {}
    for model_name, results in all_results.items():
        train_returns = results['train']['cum_returns']
        all_train_returns[model_name] = train_returns
        plt.plot(train_returns.index, train_returns, linestyle='-', alpha=0.7, 
                label=f"{model_name} (Train)")
    
    # Plot test period (out-of-sample)
    all_test_returns = {}
    for model_name, results in all_results.items():
        test_returns = results['test']['cum_returns']
        all_test_returns[model_name] = test_returns
        plt.plot(test_returns.index, test_returns, linestyle='--', alpha=0.7,
                label=f"{model_name} (Test)")
    
    # Plot metalabeled strategy
    plt.plot(train_meta_returns.index, train_meta_returns, 'k-', alpha=0.5,
            label='Metalabeled (Train)')
    plt.plot(test_meta_returns.index, test_meta_returns, 'k--', alpha=0.5,
            label='Metalabeled (Test)')
    
    # Add vertical line to separate train/test
    if args.plot_is_oos:
        train_end = train_data.index[-1]
        plt.axvline(x=train_end, color='r', linestyle='--', alpha=0.5,
                   label='Train/Test Split')
        
        # Add shaded regions for in-sample and out-of-sample
        plt.axvspan(train_data.index[0], train_end, alpha=0.1, color='green', label='In-Sample')
        plt.axvspan(train_end, test_data.index[-1], alpha=0.1, color='red', label='Out-of-Sample')
    
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    
    # Plot positions for the best model
    plt.subplot(3, 1, 2)
    best_model = list(all_results.keys())[0]  # First is best
    
    # Train positions
    train_positions = all_results[best_model]['train']['position']
    plt.plot(train_positions.index, train_positions, 'b-', alpha=0.7,
            label=f"{best_model} Position (Train)")
    
    # Test positions
    test_positions = all_results[best_model]['test']['position']
    plt.plot(test_positions.index, test_positions, 'b--', alpha=0.7,
            label=f"{best_model} Position (Test)")
    
    # Metalabeled positions
    plt.plot(train_data.index, train_data['meta_position'], 'k-', alpha=0.5,
            label='Metalabeled Position (Train)')
    plt.plot(test_data.index, test_data['meta_position'], 'k--', alpha=0.5,
            label='Metalabeled Position (Test)')
    
    # Add train/test separation
    if args.plot_is_oos:
        plt.axvline(x=train_end, color='r', linestyle='--', alpha=0.5)
        plt.axvspan(train_data.index[0], train_end, alpha=0.1, color='green')
        plt.axvspan(train_end, test_data.index[-1], alpha=0.1, color='red')
    
    plt.title(f'Position Comparison for {best_model}')
    plt.xlabel('Date')
    plt.ylabel('Position (-1=Short, 0=Neutral, 1=Long)')
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    
    # Plot performance metrics for all models
    plt.subplot(3, 1, 3)
    
    metrics = []
    for model_name, results in all_results.items():
        train_result = results['train']
        test_result = results['test']
        
        # Calculate metrics
        train_return = train_result['cum_returns'].iloc[-1]
        test_return = test_result['cum_returns'].iloc[-1]
        train_trades = train_result['trades_executed']
        test_trades = test_result['trades_executed']
        
        metrics.append({
            'Model': model_name,
            'Train Return': train_return,
            'Test Return': test_return,
            'Train Trades': train_trades,
            'Test Trades': test_trades
        })
    
    # Add metalabeled strategy
    meta_train_return = train_meta_returns.iloc[-1]
    meta_test_return = test_meta_returns.iloc[-1]
    meta_train_trades = len(train_data[train_data['meta_position'].diff() != 0])
    meta_test_trades = len(test_data[test_data['meta_position'].diff() != 0])
    
    metrics.append({
        'Model': 'Metalabeled',
        'Train Return': meta_train_return,
        'Test Return': meta_test_return,
        'Train Trades': meta_train_trades,
        'Test Trades': meta_test_trades
    })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # Plot as table
    cell_text = []
    for i in range(len(metrics_df)):
        cell_text.append([
            metrics_df.iloc[i]['Model'],
            f"{metrics_df.iloc[i]['Train Return']:.4f}",
            f"{metrics_df.iloc[i]['Test Return']:.4f}",
            f"{metrics_df.iloc[i]['Train Trades']}",
            f"{metrics_df.iloc[i]['Test Trades']}"
        ])
    
    table = plt.table(cellText=cell_text,
                     colLabels=['Model', 'Train Return', 'Test Return', 'Train Trades', 'Test Trades'],
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.axis('off')
    plt.title('Performance Metrics Comparison')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'dqn_model_comparison.png'))
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(args.output_dir, 'model_comparison_metrics.csv'), index=False)
    
    logger.info("Model comparison completed and saved")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train DQN
    agent, train_data, test_data, feature_cols, top_models = train_dqn(args)
    
    # Store train end date for visualization
    if args.plot_is_oos:
        args.train_end_date = train_data.index[-1]
        logger.info(f"Train/test split date: {args.train_end_date}")
    
    # Evaluate the final model if not already in top models
    final_model_path = os.path.join(args.model_dir, 'dqn_final.pt')
    
    # Check if final model is in top models
    final_in_top = any(final_model_path == path for _, _, path in top_models)
    
    if not final_in_top:
        logger.info("Evaluating final model on test data")
        evaluate_dqn(final_model_path, test_data, feature_cols, args)
    
    # Plot evaluation of all top models together if not done already
    if args.save_top_k > 1 and not any('dqn_model_comparison.png' in file for file in os.listdir(args.output_dir)):
        evaluate_top_models(top_models, train_data, test_data, feature_cols, args)
    
    logger.info("Training and evaluation completed!")
    logger.info(f"Saved {len(top_models)} top models")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Show top models summary
    if top_models:
        logger.info("\nTop models summary:")
        for i, (reward, episode, path) in enumerate(top_models):
            logger.info(f"{i+1}. Episode {episode}: Adjusted Reward = {reward:.4f}, Path = {path}")

if __name__ == "__main__":
    main() 