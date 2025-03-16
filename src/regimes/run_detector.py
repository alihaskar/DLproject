#!/usr/bin/env python
import os
import argparse
from market_regime_detector import MarketRegimeDetector

def main():
    parser = argparse.ArgumentParser(description='Run Market Regime Detection')
    parser.add_argument('--csv_path', type=str, default='../data/cmma.csv',
                        help='Path to CSV file with OHLC and strategy data')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save regimes CSV')
    parser.add_argument('--plot', action='store_true',
                        help='Plot market price with regime highlights')
    parser.add_argument('--regime_type', type=str, default='both',
                        choices=['feature', 'hmm', 'transformer', 'all', 'both'],
                        help='Regime detection method to use for plotting. "both" plots hmm and transformer, "all" plots all methods')
    
    args = parser.parse_args()
    
    # Make sure the input file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: Input file {args.csv_path} does not exist.")
        return
    
    print(f"Running market regime detection on {args.csv_path}...")
    detector = MarketRegimeDetector(args.csv_path)
    
    # Detect all regimes
    regimes_df = detector.detect_all_regimes()
    print("Regime detection complete.")
    
    # Save regimes to CSV
    output_path = detector.save_regimes_to_csv(args.output_path)
    print(f"Regimes saved to {output_path}")
    
    # Plot regimes if requested
    if args.plot:
        if args.regime_type == 'all':
            print("Plotting all regime types...")
            detector.plot_regimes(regime_column='feature_regime')
            detector.plot_regimes(regime_column='hmm_regime')
            detector.plot_regimes(regime_column='transformer_regime')
        elif args.regime_type == 'both':
            print("Plotting HMM and Transformer regimes...")
            detector.plot_regimes(regime_column='hmm_regime')
            detector.plot_regimes(regime_column='transformer_regime')
        else:
            regime_column = f"{args.regime_type}_regime"
            print(f"Plotting {regime_column}...")
            detector.plot_regimes(regime_column=regime_column)

if __name__ == "__main__":
    main() 