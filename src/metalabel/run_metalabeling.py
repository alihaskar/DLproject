import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.metalabel.triple_barrier import TripleBarrierLabeler, MetaLabeler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load regimes data
    logger.info("Loading regimes data")
    regimes_data = pd.read_csv("data/cmma_regimes.csv")
    regimes_data['DateTime'] = pd.to_datetime(regimes_data['DateTime'])
    regimes_data.set_index('DateTime', inplace=True)
    
    # Add technical features
    logger.info("Adding technical features")
    labeler = TripleBarrierLabeler(
        price_col='close',
        position_col='position',
        returns_col='returns',
        upper_multiplier=2.0,  # 2x ATR for profit target
        lower_multiplier=1.0,  # 1x ATR for stop loss
        max_periods=5,         # 5-day maximum holding period
        atr_window=14          # 14-day ATR
    )
    
    # Number of lags to create
    max_lags = 5
    logger.info(f"Creating features with {max_lags} lags")
    
    # Add technical features including multiple lags
    enhanced_data = labeler.add_features(regimes_data, max_lags=max_lags)
    
    # Encode regime features
    logger.info("Encoding regime features")
    
    # Encode HMM regimes
    hmm_regimes = pd.get_dummies(enhanced_data['hmm_regime'], prefix='hmm')
    enhanced_data = pd.concat([enhanced_data, hmm_regimes], axis=1)
    
    # Encode transformer regimes
    transformer_regimes = pd.get_dummies(enhanced_data['transformer_regime'], prefix='transformer')
    enhanced_data = pd.concat([enhanced_data, transformer_regimes], axis=1)
    
    # Create comprehensive feature list including new price-based features
    feature_cols = [
        # Technical indicators (will be lagged in the model)
        'atr', 'rsi', 'close_ma_diff', 
        # New price-based features
        'price_range', 'high_low_ratio', 'price_ma_ratio', 
        # Returns
        'returns',
        # HMM regimes 
        'hmm_uptrend', 'hmm_downtrend', 'hmm_volatile', 'hmm_neutral',
        # Transformer regimes
        'transformer_uptrend', 'transformer_downtrend', 'transformer_volatile', 'transformer_neutral'
    ]
    
    # Filter out features that don't exist in the data
    feature_cols = [col for col in feature_cols if col in enhanced_data.columns]
    
    logger.info(f"Using base features (will be expanded with {max_lags} lags): {feature_cols}")
    
    # Create meta-labeler
    meta_labeler = MetaLabeler(
        labeler=labeler,
        feature_cols=feature_cols
    )
    
    # Create features and labels with multiple lags
    labeled_data = meta_labeler.create_features_and_labels(enhanced_data, max_lags=max_lags)
    
    # Train meta-labeler (will use lagged features internally)
    metrics = meta_labeler.fit(labeled_data, test_size=0.3)
    
    # Apply meta-labeling WITH regime filtering
    logger.info("Applying meta-labeling with regime filtering")
    predicted_data = meta_labeler.predict(
        labeled_data,
        filter_regimes=True,
        hmm_uptrend_only=True,  # Only trade in HMM uptrends
        exclude_transformer_downtrend=True  # Avoid transformer-identified downtrends
    )
    
    # Backtest with regime filtering
    backtest_data = meta_labeler.backtest(predicted_data)
    
    # Save results to CSV with a name that indicates regime filtering
    logger.info("Saving results to CSV")
    output_path = f"data/cmma_metalabeled_atr_lag{max_lags}_regime_filtered.csv"
    backtest_data.to_csv(output_path)
    logger.info(f"Results saved to {output_path}")
    
    # Plot backtest results
    logger.info("Plotting backtest results")
    fig = meta_labeler.plot_backtest(backtest_data)
    
    # Save plot
    plot_path = f"reports/metalabel_backtest_atr_lag{max_lags}_regime_filtered.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")
    
    # Display key metrics
    original_returns = backtest_data['cum_returns'].iloc[-1]
    strategy_returns = backtest_data['cum_strategy_returns'].iloc[-1]
    
    logger.info(f"Original cumulative returns: {original_returns:.2%}")
    logger.info(f"Strategy cumulative returns: {strategy_returns:.2%}")
    
    if 'meta_position' in backtest_data.columns:
        meta_returns = meta_labeler.backtest(backtest_data, apply_meta=True)['cum_strategy_returns'].iloc[-1]
        logger.info(f"Meta-strategy cumulative returns: {meta_returns:.2%}")
    
    return backtest_data

if __name__ == "__main__":
    main() 