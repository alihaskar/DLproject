#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from tabulate import tabulate

class RegimePerformanceAnalyzer:
    def __init__(self, regimes_csv_path):
        """Initialize the analyzer with regimes data."""
        self.regimes_data = pd.read_csv(regimes_csv_path)
        
        # Convert index to datetime if available
        if 'DateTime' in self.regimes_data.columns:
            self.regimes_data['DateTime'] = pd.to_datetime(self.regimes_data['DateTime'])
            self.regimes_data.set_index('DateTime', inplace=True)
        
        # Make sure we have returns data
        if 'returns' not in self.regimes_data.columns:
            raise ValueError("Regimes data must contain a 'returns' column.")
            
        # Make sure we have regime columns
        required_columns = ['feature_regime', 'hmm_regime', 'transformer_regime']
        missing_columns = [col for col in required_columns if col not in self.regimes_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def calculate_regime_metrics(self, regime_column):
        """Calculate performance metrics for each regime type."""
        data = self.regimes_data.copy()
        
        # Group by regime
        grouped_data = data.groupby(regime_column)
        
        # Calculate metrics for each regime
        metrics = {}
        
        for regime, group in grouped_data:
            # Skip if no data for this regime
            if len(group) == 0:
                continue
                
            # Calculate metrics
            returns = group['returns'].values
            
            # Number of days/periods in this regime
            num_periods = len(group)
            
            # Total periods in the regime as a percentage
            pct_periods = num_periods / len(data) * 100
            
            # Average daily/period return
            avg_return = np.mean(returns) * 100  # Convert to percentage
            
            # Cumulative return
            cum_return = np.sum(returns) * 100  # Convert to percentage
            
            # Standard deviation of returns (volatility)
            volatility = np.std(returns) * 100  # Convert to percentage
            
            # Sharpe ratio (assuming 0 risk-free rate for simplicity)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Win rate (percentage of positive returns)
            win_rate = np.sum(returns > 0) / len(returns) * 100  # Percentage
            
            # Max drawdown
            cumulative = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative)
            max_drawdown = np.max(drawdown) * 100  # Convert to percentage
            
            # Store metrics
            metrics[regime] = {
                'Number of Periods': num_periods,
                'Percentage of Periods (%)': pct_periods,
                'Average Return (%)': avg_return,
                'Cumulative Return (%)': cum_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe,
                'Win Rate (%)': win_rate,
                'Max Drawdown (%)': max_drawdown
            }
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        
        return metrics_df
    
    def analyze_all_regimes(self):
        """Analyze performance across all regime types."""
        # Analyze each regime model
        feature_metrics = self.calculate_regime_metrics('feature_regime')
        hmm_metrics = self.calculate_regime_metrics('hmm_regime')
        transformer_metrics = self.calculate_regime_metrics('transformer_regime')
        
        return {
            'feature_regime': feature_metrics,
            'hmm_regime': hmm_metrics,
            'transformer_regime': transformer_metrics
        }
    
    def plot_regime_performance(self, regime_type='hmm_regime', figsize=(15, 10)):
        """Plot strategy performance by regime."""
        data = self.regimes_data.copy()
        
        # Calculate cumulative returns
        data['cumulative_returns'] = data['returns'].cumsum()
        
        # Get unique regimes and their colors
        unique_regimes = data[regime_type].unique()
        
        # Define colors for regimes
        regime_colors = {
            'uptrend': 'green',
            'downtrend': 'red',
            'mean_reversion': 'blue',
            'volatile': 'orange',
            'neutral': 'gray'
        }
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot cumulative returns for each regime
        for regime in unique_regimes:
            regime_data = data[data[regime_type] == regime]
            if len(regime_data) > 0:
                plt.scatter(regime_data.index, regime_data['cumulative_returns'], 
                          c=regime_colors.get(regime, 'gray'), 
                          label=regime, alpha=0.7, s=20)
        
        # Plot overall cumulative returns line
        plt.plot(data.index, data['cumulative_returns'], 'k-', alpha=0.5, label='Overall')
        
        # Add labels and title
        plt.title(f'Strategy Performance by {regime_type.replace("_", " ").title()}')
        plt.ylabel('Cumulative Returns')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Save figure
        base_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        fig_path = os.path.join(output_dir, f"{base_name}_{regime_type}_performance.png")
        plt.savefig(fig_path)
        print(f"Performance plot saved to {fig_path}")
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, regime_type='hmm_regime', figsize=(15, 10)):
        """Plot distribution of returns for each regime."""
        data = self.regimes_data.copy()
        
        # Get unique regimes
        unique_regimes = data[regime_type].unique()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot distribution for each regime
        for i, regime in enumerate(unique_regimes):
            regime_data = data[data[regime_type] == regime]['returns']
            if len(regime_data) > 0:
                sns.kdeplot(regime_data, label=regime, fill=True)
        
        # Add labels and title
        plt.title(f'Distribution of Returns by {regime_type.replace("_", " ").title()}')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Save figure
        base_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        fig_path = os.path.join(output_dir, f"{base_name}_{regime_type}_returns_dist.png")
        plt.savefig(fig_path)
        print(f"Returns distribution plot saved to {fig_path}")
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self, plot=True):
        """Run the full analysis and return results."""
        # Analyze all regimes
        all_metrics = self.analyze_all_regimes()
        
        # Display results
        for regime_type, metrics in all_metrics.items():
            print(f"\n--- {regime_type.replace('_', ' ').title()} Performance Metrics ---\n")
            print(tabulate(metrics, headers='keys', tablefmt='pretty', floatfmt='.3f'))
        
        # Plot if requested
        if plot:
            for regime_type in all_metrics.keys():
                self.plot_regime_performance(regime_type)
                self.plot_returns_distribution(regime_type)
        
        return all_metrics

def main():
    parser = argparse.ArgumentParser(description='Analyze strategy performance across market regimes')
    parser.add_argument('--regimes_csv', type=str, default='../data/cmma_regimes.csv',
                        help='Path to CSV file with regime data')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting')
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.regimes_csv):
        print(f"Error: Regimes CSV file {args.regimes_csv} does not exist.")
        return
    
    analyzer = RegimePerformanceAnalyzer(args.regimes_csv)
    analyzer.run_analysis(plot=not args.no_plot)

if __name__ == "__main__":
    main() 