#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from regime_performance_analysis import RegimePerformanceAnalyzer

def plot_regime_returns_comparison(regimes_csv_path, output_dir=None):
    """Plot a comparison of returns distribution for all regime detection methods."""
    analyzer = RegimePerformanceAnalyzer(regimes_csv_path)
    data = analyzer.regimes_data.copy()
    
    # Calculate cumulative returns
    data['cumulative_returns'] = data['returns'].cumsum()
    
    # Define colors for regimes
    regime_colors = {
        'uptrend': 'green',
        'downtrend': 'red',
        'mean_reversion': 'blue',
        'volatile': 'orange',
        'neutral': 'gray'
    }
    
    # Create figure with 3 subplots (one for each model)
    fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    
    # Set titles for each subplot
    titles = [
        'Feature-based Model Regimes vs. Cumulative Returns',
        'HMM Model Regimes vs. Cumulative Returns',
        'Transformer Model Regimes vs. Cumulative Returns'
    ]
    
    # Plot each model's regimes
    regime_columns = ['feature_regime', 'hmm_regime', 'transformer_regime']
    
    for i, (ax, regime_col, title) in enumerate(zip(axes, regime_columns, titles)):
        # Plot cumulative returns line
        ax.plot(data.index, data['cumulative_returns'], 'k-', alpha=0.7, linewidth=1.2, label='Cumulative Returns')
        
        # Get unique regimes for this model
        unique_regimes = data[regime_col].unique()
        
        # Add colored background for each regime
        current_regime = None
        start_idx = 0
        
        for j, (idx, row) in enumerate(data.iterrows()):
            regime = row[regime_col]
            
            if regime != current_regime:
                if current_regime is not None:
                    end_idx = j - 1
                    ax.axvspan(data.index[start_idx], data.index[end_idx], 
                              alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
                current_regime = regime
                start_idx = j
        
        # Add the last regime segment
        if current_regime is not None:
            ax.axvspan(data.index[start_idx], data.index[-1], 
                      alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
        
        # Create legend handles for regimes
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=regime_colors.get(regime, 'gray'), 
                                        alpha=0.5, label=regime) 
                          for regime in unique_regimes if regime in regime_colors]
        
        # Add returns line to legend
        legend_elements.append(plt.Line2D([0], [0], color='k', label='Cumulative Returns'))
        
        # Add title and legend
        ax.set_title(title, fontsize=14)
        ax.legend(handles=legend_elements, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines at major drawdowns or regime changes
        max_drawdowns = {}
        for regime in unique_regimes:
            regime_data = data[data[regime_col] == regime]
            cum_returns = regime_data['returns'].cumsum().values
            
            if len(cum_returns) > 0:
                peak = np.maximum.accumulate(cum_returns)
                drawdown = peak - cum_returns
                if len(drawdown) > 0:
                    max_idx = np.argmax(drawdown)
                    if max_idx > 0 and max_idx < len(regime_data):
                        max_drawdowns[regime] = (regime_data.index[max_idx], drawdown[max_idx])
        
        # Mark max drawdowns
        for regime, (date, drawdown) in max_drawdowns.items():
            if drawdown > 0.05:  # Only mark significant drawdowns
                ax.axvline(x=date, color=regime_colors.get(regime, 'gray'), 
                          linestyle='--', alpha=0.8, linewidth=0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(output_dir, 'data')
    
    base_name = os.path.basename(regimes_csv_path).split('.')[0]
    fig_path = os.path.join(output_dir, f"{base_name}_regime_comparison.png")
    plt.savefig(fig_path)
    print(f"Regime comparison plot saved to {fig_path}")
    
    return fig_path

def plot_regime_boxplots(regimes_csv_path, output_dir=None):
    """Plot boxplots of returns by regime for all models."""
    analyzer = RegimePerformanceAnalyzer(regimes_csv_path)
    data = analyzer.regimes_data.copy()
    
    # Create a large figure with 3 subplots (one for each model)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # Set titles for each subplot
    titles = [
        'Feature-based Model Returns',
        'HMM Model Returns',
        'Transformer Model Returns'
    ]
    
    # Plot each model's regime boxplots
    regime_columns = ['feature_regime', 'hmm_regime', 'transformer_regime']
    
    for i, (ax, regime_col, title) in enumerate(zip(axes, regime_columns, titles)):
        # Convert returns to percentage for better visualization
        data['returns_pct'] = data['returns'] * 100
        
        # Create boxplot
        sns.boxplot(x=regime_col, y='returns_pct', data=data, ax=ax, palette='viridis')
        
        # Add swarmplot for distribution visualization
        sns.stripplot(x=regime_col, y='returns_pct', data=data, 
                     ax=ax, size=2, color='black', alpha=0.3)
        
        # Add title and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Regime')
        ax.set_ylabel('Returns (%)')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(output_dir, 'data')
    
    base_name = os.path.basename(regimes_csv_path).split('.')[0]
    fig_path = os.path.join(output_dir, f"{base_name}_regime_boxplots.png")
    plt.savefig(fig_path)
    print(f"Regime boxplots saved to {fig_path}")
    
    return fig_path

def plot_regime_metrics_comparison(regimes_csv_path, output_dir=None):
    """Plot a comparison of key metrics across regimes and models."""
    analyzer = RegimePerformanceAnalyzer(regimes_csv_path)
    all_metrics = analyzer.analyze_all_regimes()
    
    # Create metrics we want to compare
    metrics_to_plot = ['Cumulative Return (%)', 'Average Return (%)', 
                       'Sharpe Ratio', 'Win Rate (%)', 'Volatility (%)']
    
    # Create figure
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(15, 20))
    
    # For each metric, plot bar chart comparing across regimes and models
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Prepare data for plotting
        plot_data = []
        model_names = []
        regime_names = []
        values = []
        
        for model_name, metrics_df in all_metrics.items():
            for regime, row in metrics_df.iterrows():
                model_names.append(model_name.replace('_regime', ''))
                regime_names.append(regime)
                values.append(row[metric])
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Model': model_names,
            'Regime': regime_names,
            'Value': values
        })
        
        # Plot as grouped bar chart
        sns.barplot(x='Regime', y='Value', hue='Model', data=plot_df, ax=ax, palette='viridis')
        
        # Add title and labels
        ax.set_title(f'{metric} by Regime and Model', fontsize=14)
        ax.set_xlabel('Regime')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha='center', va='center', fontsize=8, color='black', 
                       xytext=(0, 5), textcoords='offset points')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(output_dir, 'data')
    
    base_name = os.path.basename(regimes_csv_path).split('.')[0]
    fig_path = os.path.join(output_dir, f"{base_name}_regime_metrics_comparison.png")
    plt.savefig(fig_path)
    print(f"Regime metrics comparison saved to {fig_path}")
    
    return fig_path

def main():
    parser = argparse.ArgumentParser(description='Compare regime performances across different models')
    parser.add_argument('--regimes_csv', type=str, default='../data/cmma_regimes.csv',
                        help='Path to CSV file with regime data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output plots')
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.regimes_csv):
        print(f"Error: Regimes CSV file {args.regimes_csv} does not exist.")
        return
    
    # Generate comparison plots
    plot_regime_returns_comparison(args.regimes_csv, args.output_dir)
    plot_regime_boxplots(args.regimes_csv, args.output_dir)
    plot_regime_metrics_comparison(args.regimes_csv, args.output_dir)
    
    print("All comparison plots generated successfully.")

if __name__ == "__main__":
    main() 