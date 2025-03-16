import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import json
from datetime import datetime

# Custom JSON encoder for handling pandas Timestamp objects
class PandasJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)

logger = logging.getLogger(__name__)

# Utility function to convert results dictionary for JSON serialization
def prepare_results_for_json(results):
    """Clean and prepare results dictionary for JSON serialization"""
    json_safe_results = {}
    
    # Process each item in the dictionary
    for key, value in results.items():
        # Handle numpy arrays and pandas series
        if isinstance(value, (np.ndarray, pd.Series)):
            json_safe_results[key] = value.tolist()
        # Handle lists containing numpy types
        elif isinstance(value, list):
            # Convert list items if needed
            json_safe_results[key] = _convert_list_items(value)
        # Handle dictionaries
        elif isinstance(value, dict):
            json_safe_results[key] = prepare_results_for_json(value)
        # Handle pandas Timestamp
        elif isinstance(value, pd.Timestamp):
            json_safe_results[key] = value.strftime('%Y-%m-%d %H:%M:%S')
        # Handle numpy types
        elif isinstance(value, (np.integer, np.floating)):
            json_safe_results[key] = value.item()
        else:
            json_safe_results[key] = value
    
    return json_safe_results

def _convert_list_items(items):
    """Helper to convert numpy types within lists"""
    result = []
    for item in items:
        if isinstance(item, (np.integer, np.floating)):
            result.append(item.item())
        elif isinstance(item, (np.ndarray, pd.Series)):
            result.append(item.tolist())
        elif isinstance(item, pd.Timestamp):
            result.append(item.strftime('%Y-%m-%d %H:%M:%S'))
        elif isinstance(item, dict):
            result.append(prepare_results_for_json(item))
        elif isinstance(item, list):
            result.append(_convert_list_items(item))
        else:
            result.append(item)
    return result

def select_features(df):
    """Select features for the model."""
    # Technical features
    tech_features = [col for col in df.columns if any(x in col for x in ['atr', 'rsi', 'ma_diff', 'price_range'])]
    
    # Regime features
    regime_features = [col for col in df.columns if any(x in col for x in ['hmm_', 'transformer_', 'feature_'])]
    
    # lagged features
    lag_features = [col for col in df.columns if 'lag' in col]
    
    # Include position and confidence if available
    meta_features = ['meta_position', 'meta_confidence'] if 'meta_position' in df.columns else []
    
    # Combine all features
    all_features = tech_features + regime_features + lag_features + meta_features
    
    # Filter out features that don't exist
    features = [f for f in all_features if f in df.columns]
    
    # Ensure all features are numeric
    for feature in features:
        if not np.issubdtype(df[feature].dtype, np.number):
            logger.warning(f"Non-numeric feature {feature} will be dropped")
    
    features = [f for f in features if np.issubdtype(df[f].dtype, np.number)]
    
    logger.info(f"Selected {len(features)} features: {features}")
    return features

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
    
    # Ensure 'position' is mapped to 'meta_position' if it exists
    if 'position' in df.columns and 'meta_position' not in df.columns:
        df['meta_position'] = df['position']
    
    # Select features
    feature_cols = select_features(df)
    
    # Split data into train and test
    np.random.seed(seed)
    train_size = int(len(df) * (1 - test_size))
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    
    logger.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Calculate minimum trades target based on metalabeled strategy
    meta_trades = 0
    if 'meta_position' in df.columns:
        meta_trades = len(df[df['meta_position'].diff() != 0])
        logger.info(f"Original metalabeled strategy made {meta_trades} trades")
    
    return train_data, test_data, feature_cols, meta_trades

def evaluate_sac(agent, data, feature_cols, args, min_trades=None):
    """Evaluate the SAC agent on the given data."""
    # Import locally to avoid circular imports
    from src.RL.SAC.trading_env import TradingEnvironment
    
    # Check if feature count matches agent's state_dim
    if hasattr(agent, 'state_dim') and len(feature_cols) != agent.state_dim:
        logger.warning(f"Feature count ({len(feature_cols)}) does not match agent's state_dim ({agent.state_dim})")
        # If feature count doesn't match, we'll let the environment handle it
        # The environment might need to pad or truncate features
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        feature_columns=feature_cols,
        transaction_cost=args.transaction_cost,
        trade_penalty=0,  # No penalty during evaluation
        holding_penalty=0,  # No penalty during evaluation
        correct_direction_reward=0,  # No reward during evaluation
        min_trades_per_episode=0,  # No minimum trades during evaluation
        min_trades_penalty=0  # No penalty during evaluation
    )
    
    # Run evaluation
    state = env.reset()
    done = False
    episode_steps = 0
    
    # Evaluate with progress bar
    max_steps = len(data)
    from tqdm import tqdm
    pbar = tqdm(total=max_steps, desc="Evaluating")
    
    while not done and episode_steps < max_steps:
        # Select action (deterministic for evaluation)
        action = agent.select_action(state, evaluate=True)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state
        episode_steps += 1
        
        # Update progress bar
        pbar.update(1)
    
    pbar.close()
    
    # Get results
    results = env.get_results()
    
    # Calculate additional metrics
    returns = np.array(results['returns'])
    
    # Calculate cumulative returns
    cum_returns = np.cumsum(returns)
    
    # Calculate Sharpe ratio
    if len(returns) > 0 and np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    else:
        sharpe = 0
    
    # Calculate drawdown
    portfolio_values = np.array(results['portfolio_values'])
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate win rate
    if results['trade_count'] > 0:
        trade_returns = []
        for trade in results['trades']:
            # Find the return from this trade
            if 'return' in trade:
                trade_returns.append(trade['return'])
        
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        else:
            win_rate = 0
    else:
        win_rate = 0
    
    # Add metrics to results
    metrics = {
        'total_return': np.sum(returns),
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trade_count': results['trade_count'],
        'transaction_costs': results['transaction_costs'],
        'final_balance': results['balance'],
        'cum_returns': cum_returns.tolist() if isinstance(cum_returns, np.ndarray) else cum_returns,
        'positions': results['positions']
    }
    
    # Merge with results
    results.update(metrics)
    
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

def compare_with_metalabeled(sac_results, train_data, test_data, args):
    """Compare SAC results with metalabeled strategy."""
    logger.info("Comparing SAC with metalabeled strategy")
    
    # Calculate metalabeled strategy returns
    train_meta_returns = calculate_metalabeled_returns(train_data)
    test_meta_returns = calculate_metalabeled_returns(test_data)
    
    # Calculate metalabeled strategy metrics for test data
    meta_returns = test_data['meta_position'].shift(1) * test_data['returns'] if 'meta_position' in test_data.columns else pd.Series(0, index=test_data.index)
    meta_returns = meta_returns.fillna(0)
    
    # Calculate transaction costs for metalabeled strategy
    if 'meta_position' in test_data.columns:
        meta_position_changes = (test_data['meta_position'].diff() != 0)
        meta_trades = meta_position_changes.sum()
        meta_costs = meta_trades * args.transaction_cost
    else:
        meta_trades = 0
        meta_costs = 0
    
    # Calculate final return for metalabeled strategy
    meta_final_return = meta_returns.sum()
    
    # Calculate Sharpe ratio for metalabeled strategy
    if len(meta_returns) > 0 and meta_returns.std() != 0:
        meta_sharpe = meta_returns.mean() / meta_returns.std() * np.sqrt(252)  # Annualized
    else:
        meta_sharpe = 0
    
    # Calculate drawdown for metalabeled strategy
    meta_cum_returns = meta_returns.cumsum()
    meta_peak = meta_cum_returns.cummax()
    meta_drawdown = (meta_peak - meta_cum_returns) / meta_peak.replace(0, 1)
    meta_max_drawdown = meta_drawdown.max() if len(meta_drawdown) > 0 else 0
    
    # Get SAC metrics
    sac_final_return = sac_results['total_return']
    sac_trades = sac_results['trade_count']
    sac_costs = sac_results['transaction_costs']
    sac_sharpe = sac_results['sharpe_ratio']
    sac_max_drawdown = sac_results['max_drawdown']
    
    # Compare the strategies
    logger.info(f"SAC: Return={sac_final_return:.4f}, Trades={sac_trades}, Costs={sac_costs:.4f}, "
               f"Sharpe={sac_sharpe:.2f}, Max Drawdown={sac_max_drawdown:.4f}")
    logger.info(f"Metalabeled: Return={meta_final_return:.4f}, Trades={meta_trades}, Costs={meta_costs:.4f}, "
               f"Sharpe={meta_sharpe:.2f}, Max Drawdown={meta_max_drawdown:.4f}")
    
    # Calculate percentage improvements
    if meta_final_return != 0:
        return_improvement = (sac_final_return / meta_final_return - 1) * 100
    else:
        return_improvement = float('inf') if sac_final_return > 0 else float('-inf')
    
    if meta_trades != 0:
        trade_reduction = (1 - sac_trades / meta_trades) * 100
    else:
        trade_reduction = float('inf') if sac_trades == 0 else float('-inf')
    
    if meta_costs != 0:
        cost_reduction = (1 - sac_costs / meta_costs) * 100
    else:
        cost_reduction = float('inf') if sac_costs == 0 else float('-inf')
    
    logger.info(f"Return improvement: {return_improvement:.2f}%")
    logger.info(f"Trade reduction: {trade_reduction:.2f}%")
    logger.info(f"Cost reduction: {cost_reduction:.2f}%")
    
    # Save comparison metrics to CSV
    comparison = pd.DataFrame({
        'Metric': ['Final Return', 'Number of Trades', 'Transaction Costs', 
                 'Sharpe Ratio', 'Max Drawdown',
                 'Cost per Trade', 'Return per Trade'],
        'SAC': [sac_final_return, sac_trades, sac_costs, 
               sac_sharpe, sac_max_drawdown,
               0 if sac_trades == 0 else sac_costs/sac_trades,
               0 if sac_trades == 0 else sac_final_return/sac_trades],
        'Metalabeled': [meta_final_return, meta_trades, meta_costs, 
                      meta_sharpe, meta_max_drawdown,
                      meta_costs/meta_trades if meta_trades > 0 else 0, 
                      meta_final_return/meta_trades if meta_trades > 0 else 0]
    })
    comparison.to_csv(os.path.join(args.output_dir, 'strategy_comparison.csv'), index=False)
    
    # Plot comparison
    plot_model_comparison(
        sac_results, train_meta_returns, test_meta_returns, 
        train_data, test_data, args
    )

def plot_results(results, train_data, test_data, save_path):
    """Plot evaluation results."""
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Get data for plotting
    cum_returns = np.array(results['cum_returns'])
    positions = np.array(results['positions'])
    
    # Get test data index for x-axis
    if hasattr(test_data, 'index'):
        x_values = test_data.index
    else:
        x_values = np.arange(len(test_data))
    
    # Plot cumulative returns
    axs[0].plot(x_values[:len(cum_returns)], cum_returns, label='SAC Returns')
    
    # Plot metalabeled strategy returns if available
    if 'meta_position' in test_data.columns and 'returns' in test_data.columns:
        meta_returns = test_data['meta_position'].shift(1) * test_data['returns']
        meta_cum_returns = meta_returns.fillna(0).cumsum()
        axs[0].plot(x_values[:len(meta_cum_returns)], meta_cum_returns, 
                   label='Metalabeled Returns', linestyle='--')
    
    axs[0].set_title('Cumulative Returns')
    axs[0].set_ylabel('Return')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot positions
    axs[1].plot(x_values[:len(positions)], positions)
    axs[1].set_title('SAC Positions')
    axs[1].set_ylabel('Position Size')
    axs[1].grid(True)
    
    # Plot meta positions if available
    if 'meta_position' in test_data.columns:
        axs[2].plot(x_values[:len(test_data)], test_data['meta_position'].values)
        axs[2].set_title('Metalabeled Positions')
        axs[2].set_ylabel('Position Size')
        axs[2].grid(True)
    
    # Add price to right y-axis if close is available
    if 'close' in test_data.columns:
        ax_price = axs[0].twinx()
        ax_price.plot(x_values[:len(test_data)], test_data['close'].values, 
                     color='gray', alpha=0.5, label='Price')
        ax_price.set_ylabel('Price')
    
    # Format x-axis if using datetime
    if pd.api.types.is_datetime64_any_dtype(x_values):
        fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_metrics(metrics, save_path):
    """Plot training metrics."""
    # Create figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 15))
    
    # Plot episode rewards
    axs[0].plot(metrics['rewards'])
    axs[0].set_title('Episode Rewards')
    axs[0].set_ylabel('Reward')
    axs[0].grid(True)
    
    # Plot trade count and transaction costs
    ax_trades = axs[1]
    ax_trades.plot(metrics['trades'], label='Trades')
    ax_trades.set_title('Trade Count and Transaction Costs')
    ax_trades.set_ylabel('Number of Trades')
    ax_trades.grid(True)
    
    ax_costs = ax_trades.twinx()
    ax_costs.plot(metrics['costs'], color='orange', label='Costs')
    ax_costs.set_ylabel('Transaction Costs')
    
    # Add legend
    lines, labels = ax_trades.get_legend_handles_labels()
    lines2, labels2 = ax_costs.get_legend_handles_labels()
    ax_trades.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # Plot returns and Sharpe ratios
    ax_returns = axs[2]
    ax_returns.plot(metrics['returns'], label='Returns')
    ax_returns.set_title('Returns and Sharpe Ratio')
    ax_returns.set_ylabel('Return')
    ax_returns.grid(True)
    
    ax_sharpe = ax_returns.twinx()
    ax_sharpe.plot(metrics['sharpes'], color='green', label='Sharpe')
    ax_sharpe.set_ylabel('Sharpe Ratio')
    
    # Add legend
    lines, labels = ax_returns.get_legend_handles_labels()
    lines2, labels2 = ax_sharpe.get_legend_handles_labels()
    ax_returns.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # Plot losses
    axs[3].plot(metrics['actor_losses'], label='Actor Loss')
    axs[3].plot(metrics['critic_losses'], label='Critic Loss')
    axs[3].plot(metrics['alpha_losses'], label='Alpha Loss')
    axs[3].set_title('Training Losses')
    axs[3].set_ylabel('Loss')
    axs[3].set_xlabel('Episode')
    axs[3].grid(True)
    axs[3].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(sac_results, train_meta_returns, test_meta_returns, 
                         train_data, test_data, args):
    """Plot comparison between SAC and metalabeled strategy."""
    # Create figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot cumulative returns for both training and testing
    # Training data
    if hasattr(train_data, 'index'):
        train_x = train_data.index
    else:
        train_x = np.arange(len(train_data))
    
    # Testing data
    if hasattr(test_data, 'index'):
        test_x = test_data.index
    else:
        test_x = np.arange(len(test_data)) + len(train_data)
    
    # Plot training returns
    train_meta_returns_values = train_meta_returns.values if hasattr(train_meta_returns, 'values') else train_meta_returns
    
    # Plot test returns
    test_meta_returns_values = test_meta_returns.values if hasattr(test_meta_returns, 'values') else test_meta_returns
    sac_cum_returns = np.array(sac_results['cum_returns'])
    
    # Create combined plot
    axs[0].plot(train_x, train_meta_returns_values, color='blue', linestyle='--', alpha=0.5)
    axs[0].plot(test_x[:len(test_meta_returns_values)], test_meta_returns_values,
               color='blue', label='Metalabeled Strategy')
    axs[0].plot(test_x[:len(sac_cum_returns)], sac_cum_returns,
               color='green', label='SAC Strategy')
    
    # Add vertical line to separate train and test
    axs[0].axvline(x=test_x[0], color='red', linestyle='--', 
                  label='Train/Test Split')
    
    axs[0].set_title('Cumulative Returns Comparison')
    axs[0].set_ylabel('Cumulative Return')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot positions for both strategies in test set
    sac_positions = np.array(sac_results['positions'])
    
    if 'meta_position' in test_data.columns:
        meta_positions = test_data['meta_position'].values
        
        # Align lengths
        min_len = min(len(sac_positions), len(meta_positions))
        
        # Create comparison plot
        axs[1].plot(test_x[:min_len], meta_positions[:min_len], 
                   color='blue', label='Metalabeled Position')
        axs[1].plot(test_x[:min_len], sac_positions[:min_len], 
                   color='green', label='SAC Position')
        
        axs[1].set_title('Position Comparison')
        axs[1].set_ylabel('Position')
        axs[1].legend()
        axs[1].grid(True)
    
    # Format x-axis if using datetime
    if pd.api.types.is_datetime64_any_dtype(train_x) or pd.api.types.is_datetime64_any_dtype(test_x):
        fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'))
    plt.close()
    
    # Also create a plot showing trade frequency difference
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate trade points
    if 'meta_position' in test_data.columns:
        meta_trades = (test_data['meta_position'].diff() != 0)
        meta_trade_points = test_x[meta_trades.values]
        
        # Calculate SAC trade points from positions
        sac_positions = np.array(sac_results['positions'])
        sac_trades = np.diff(sac_positions, prepend=0)
        sac_trades = np.abs(sac_trades) > 0.1  # Use threshold
        sac_trade_points = test_x[:len(sac_trades)][sac_trades]
        
        # Plot price if available
        if 'close' in test_data.columns:
            ax.plot(test_x, test_data['close'].values, color='gray', alpha=0.5, label='Price')
        
        # Plot trade points
        ax.scatter(meta_trade_points, [1] * len(meta_trade_points), color='blue', 
                  marker='|', s=100, label='Metalabeled Trades')
        ax.scatter(sac_trade_points, [0.95] * len(sac_trade_points), color='green', 
                  marker='|', s=100, label='SAC Trades')
        
        ax.set_title('Trade Frequency Comparison')
        ax.set_ylabel('Trade Events')
        ax.legend()
        ax.grid(True)
        
        # Format x-axis if using datetime
        if pd.api.types.is_datetime64_any_dtype(test_x):
            fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'trade_frequency_comparison.png'))
        plt.close() 