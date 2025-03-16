import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging
import ta

logger = logging.getLogger(__name__)

class TripleBarrierLabeler:
    """
    Triple barrier labeling method for meta-labeling.
    
    This method uses upper and lower price barriers along with a time barrier
    to create labels.
    """
    
    def __init__(self, 
                 price_col: str = 'close',
                 position_col: str = 'position',
                 returns_col: str = 'returns',
                 upper_multiplier: float = 3.0,
                 lower_multiplier: float = 2.0,
                 max_periods: int = 10,
                 atr_window: int = 14):
        """
        Initialize triple barrier labeler.
        
        Args:
            price_col: Name of the price column
            position_col: Name of the position column
            returns_col: Name of the returns column
            upper_multiplier: Upper barrier ATR multiplier
            lower_multiplier: Lower barrier ATR multiplier
            max_periods: Maximum number of periods to look ahead
            atr_window: Window length for ATR calculation
        """
        self.price_col = price_col
        self.position_col = position_col
        self.returns_col = returns_col
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.max_periods = max_periods
        self.atr_window = atr_window
        
    def add_features(self, data: pd.DataFrame, max_lags=3) -> pd.DataFrame:
        """
        Add technical indicators as features.
        
        Args:
            data: DataFrame containing OHLC data
            max_lags: Maximum number of lags to create
            
        Returns:
            DataFrame with added features
        """
        df = data.copy()
        
        # Add ATR if not already present
        if 'atr' not in df.columns:
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=self.atr_window
            )
        
        # Add RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Add moving average and difference
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['close_ma_diff'] = df['close'] - df['ma20']
        
        # Add returns lags
        if self.returns_col in df.columns:
            for lag in range(1, max_lags+1):
                df[f'{self.returns_col}_lag{lag}'] = df[self.returns_col].shift(lag)
        
        # Add OHLC and price volatility features
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Add more complex features
        df['price_ma_ratio'] = df['close'] / df['ma20']
        
        # Create lagged features to avoid lookahead bias
        tech_features = ['atr', 'rsi', 'close_ma_diff', 'price_range', 'high_low_ratio', 'price_ma_ratio']
        for feature in tech_features:
            if feature in df.columns:
                for lag in range(1, max_lags+1):
                    df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        
        # Create lagged regime features
        regime_features = ['hmm_regime', 'transformer_regime', 'feature_regime']
        for feature in regime_features:
            if feature in df.columns:
                for lag in range(1, max_lags+1):
                    df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        
        # Fill NaN values in all columns
        df = df.fillna(method='bfill')
        
        return df
    
    def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create labels using triple barrier method.
        
        Args:
            data: DataFrame containing price, position, and returns columns
            
        Returns:
            DataFrame with labels added
        """
        # Create a copy of the dataframe
        df = data.copy()
        
        # Make sure returns column exists
        if self.returns_col not in df.columns:
            logger.warning(f"Returns column '{self.returns_col}' not found. Calculating returns.")
            df[self.returns_col] = df[self.price_col].pct_change()
            
        # Make sure ATR column exists
        if 'atr' not in df.columns:
            logger.warning(f"ATR column not found. Calculating ATR.")
            df = self.add_features(df)
        
        # Initialize columns
        df['meta_label'] = 0
        df['barrier_touched'] = None
        
        # Get indices where we have a position
        position_mask = df[self.position_col] != 0
        position_indices = df.index[position_mask]
        
        logger.info(f"Found {len(position_indices)} positions in data")
        
        # Process each position
        for i, idx in enumerate(position_indices):
            if i >= len(position_indices) - self.max_periods:
                # Skip if we don't have enough future data
                continue
            
            # Get the position direction
            position = df.loc[idx, self.position_col]
            
            if position == 0:
                continue
            
            # Get entry price and ATR at entry
            entry_price = df.loc[idx, self.price_col]
            entry_atr = df.loc[idx, 'atr']
            
            # Calculate barriers using ATR
            if position > 0:  # Long position
                upper_barrier = entry_price + (entry_atr * self.upper_multiplier)
                lower_barrier = entry_price - (entry_atr * self.lower_multiplier)
            else:  # Short position
                upper_barrier = entry_price - (entry_atr * self.upper_multiplier)
                lower_barrier = entry_price + (entry_atr * self.lower_multiplier)
            
            # Look ahead for max_periods or until we hit a barrier
            exit_idx = None
            barrier_touched = None
            
            for j in range(1, self.max_periods + 1):
                look_ahead_idx = df.index[df.index.get_loc(idx) + j]
                future_price = df.loc[look_ahead_idx, self.price_col]
                
                # Check if we hit a barrier
                if position > 0:  # Long position
                    if future_price >= upper_barrier:
                        exit_idx = look_ahead_idx
                        barrier_touched = 'upper'
                        break
                    elif future_price <= lower_barrier:
                        exit_idx = look_ahead_idx
                        barrier_touched = 'lower'
                        break
                else:  # Short position
                    if future_price <= upper_barrier:
                        exit_idx = look_ahead_idx
                        barrier_touched = 'upper'
                        break
                    elif future_price >= lower_barrier:
                        exit_idx = look_ahead_idx
                        barrier_touched = 'lower'
                        break
            
            # If we hit the max periods without touching a barrier
            if exit_idx is None:
                exit_idx = df.index[df.index.get_loc(idx) + self.max_periods]
                barrier_touched = 'time'
            
            # Calculate returns from entry to exit
            exit_price = df.loc[exit_idx, self.price_col]
            returns = (exit_price / entry_price - 1) * position
            
            # Set meta label based on barriers touched and returns
            if barrier_touched == 'upper' or (barrier_touched == 'time' and returns > 0):
                df.loc[idx, 'meta_label'] = 1
            else:
                df.loc[idx, 'meta_label'] = 0
                
            df.loc[idx, 'barrier_touched'] = barrier_touched
        
        # Count labels
        num_labels = len(df[~df['barrier_touched'].isna()])
        num_positive = df['meta_label'].sum()
        
        if num_labels > 0:
            logger.info(f"Created {num_labels} labels with {num_positive} positive ({num_positive/num_labels:.2%})")
        else:
            logger.warning("No labels were created")
        
        return df

class MetaLabeler:
    """
    Meta-labeling for trading strategies.
    """
    
    def __init__(self, 
                 labeler=None,
                 model=None,
                 feature_cols=None):
        """
        Initialize meta-labeler.
        
        Args:
            labeler: Labeling method instance
            model: Machine learning model for meta-labeling
            feature_cols: List of feature columns to use
        """
        self.labeler = labeler or TripleBarrierLabeler()
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_cols = feature_cols or []
        self._is_trained = False
        self.metrics = {}
        
    def create_features_and_labels(self, data: pd.DataFrame, max_lags=3) -> pd.DataFrame:
        """
        Create features and labels for meta-labeling.
        
        Args:
            data: DataFrame containing OHLC and position data
            max_lags: Maximum number of lags to create
            
        Returns:
            DataFrame with features and labels
        """
        # Add features
        df = self.labeler.add_features(data, max_lags=max_lags)
        
        # Create labels
        df = self.labeler.create_labels(df)
        
        return df
        
    def fit(self, data: pd.DataFrame, test_size=0.2, random_state=42):
        """
        Train the meta-labeling model.
        
        Args:
            data: DataFrame with features and labels
            test_size: Size of test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with performance metrics
        """
        # Get all lag feature columns to avoid lookahead bias
        lag_pattern = '_lag'
        lagged_feature_cols = []
        
        for col in data.columns:
            # Include columns that are explicitly lagged
            if lag_pattern in col:
                lagged_feature_cols.append(col)
            # Include regime indicators (these don't need lagging as they're already known)
            elif col.startswith('hmm_') or col.startswith('transformer_'):
                lagged_feature_cols.append(col)
        
        # Filter to include only the requested feature columns or their lagged versions
        filtered_cols = []
        for feature in self.feature_cols:
            # For technical indicators, use their lag versions
            if feature in ['atr', 'rsi', 'close_ma_diff', 'price_range', 'high_low_ratio', 'price_ma_ratio', 'returns']:
                for col in lagged_feature_cols:
                    if col.startswith(f"{feature}_lag"):
                        filtered_cols.append(col)
            # For regime indicators, use as is
            elif feature in lagged_feature_cols:
                filtered_cols.append(feature)
        
        # Make sure we have at least some features
        if not filtered_cols:
            logger.warning("No lagged feature columns found. Using all available lagged columns.")
            filtered_cols = lagged_feature_cols
        
        logger.info(f"Using lagged features for training: {filtered_cols}")
        
        # Extract features with positions only
        position_mask = data[self.labeler.position_col] != 0
        X = data.loc[position_mask, filtered_cols].values
        y = data.loc[position_mask, 'meta_label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Display metrics prominently
        logger.info("\n" + "="*50)
        logger.info(" "*15 + "MODEL PERFORMANCE METRICS")
        logger.info("="*50)
        logger.info(f"ACCURACY:  {self.metrics['accuracy']:.4f}")
        logger.info(f"PRECISION: {self.metrics['precision']:.4f}")
        logger.info(f"RECALL:    {self.metrics['recall']:.4f}")
        logger.info(f"F1 SCORE:  {self.metrics['f1']:.4f}")
        logger.info("="*50 + "\n")
        
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Store the lagged feature columns for prediction
        self.lagged_feature_cols = filtered_cols
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = sorted(zip(filtered_cols, importances), key=lambda x: x[1], reverse=True)
            logger.info("Feature importances:")
            for feature, importance in feature_importance:
                logger.info(f"  {feature}: {importance:.4f}")
        
        self._is_trained = True
        return self.metrics
    
    def predict(self, data: pd.DataFrame, 
                filter_regimes=False,
                hmm_uptrend_only=False, 
                transformer_uptrend_only=False,
                exclude_hmm_downtrend=False,
                exclude_transformer_downtrend=False) -> pd.DataFrame:
        """
        Apply meta-labeling to filter positions.
        
        Args:
            data: DataFrame with features and positions
            filter_regimes: Whether to filter trades based on regime conditions
            hmm_uptrend_only: Only take positions during HMM uptrend
            transformer_uptrend_only: Only take positions during Transformer uptrend
            exclude_hmm_downtrend: Exclude positions during HMM downtrend
            exclude_transformer_downtrend: Exclude positions during Transformer downtrend
            
        Returns:
            DataFrame with meta-labeled positions
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Create a copy
        df = data.copy()
        
        # Extract features for positions
        position_mask = df[self.labeler.position_col] != 0
        pos_indices = df.index[position_mask]
        
        if len(pos_indices) == 0:
            logger.warning("No positions found for prediction")
            df['meta_position'] = 0
            return df
        
        # Get features using lagged columns to avoid lookahead bias
        X = df.loc[position_mask, self.lagged_feature_cols].values
        
        # Predict
        y_pred = self.model.predict(X)
        
        # Create meta position column
        df['meta_position'] = 0
        
        # Apply meta labels to positions
        regime_filtered_count = 0
        ml_accepted_count = 0
        final_positions_count = 0
        
        for i, idx in enumerate(pos_indices):
            # Check regime conditions if filtering is enabled
            skip_due_to_regime = False
            
            if filter_regimes:
                if hmm_uptrend_only and 'hmm_uptrend' in df.columns and df.loc[idx, 'hmm_uptrend'] != 1:
                    skip_due_to_regime = True
                
                if transformer_uptrend_only and 'transformer_uptrend' in df.columns and df.loc[idx, 'transformer_uptrend'] != 1:
                    skip_due_to_regime = True
                
                if exclude_hmm_downtrend and 'hmm_downtrend' in df.columns and df.loc[idx, 'hmm_downtrend'] == 1:
                    skip_due_to_regime = True
                
                if exclude_transformer_downtrend and 'transformer_downtrend' in df.columns and df.loc[idx, 'transformer_downtrend'] == 1:
                    skip_due_to_regime = True
            
            if skip_due_to_regime:
                regime_filtered_count += 1
                continue
            
            # Only take the position if meta-label is positive
            if y_pred[i] == 1:
                ml_accepted_count += 1
                df.loc[idx, 'meta_position'] = df.loc[idx, self.labeler.position_col]
                final_positions_count += 1
        
        # Log filtering stats
        if filter_regimes:
            logger.info(f"Regime filtering: {regime_filtered_count} positions filtered out ({regime_filtered_count/len(pos_indices):.2%})")
        
        logger.info(f"ML model accepted: {ml_accepted_count} positions ({ml_accepted_count/(len(pos_indices)-regime_filtered_count):.2%})")
        logger.info(f"Final positions: {final_positions_count} out of {len(pos_indices)} original positions ({final_positions_count/len(pos_indices):.2%})")
        
        return df
    
    def backtest(self, data: pd.DataFrame, apply_meta=True):
        """
        Backtest trading strategy with meta-labels.
        
        Args:
            data: DataFrame with features, positions and meta-positions
            apply_meta: Whether to use meta-positions or original positions
            
        Returns:
            DataFrame with backtest results
        """
        df = data.copy()
        
        # Calculate returns
        if 'returns' not in df.columns:
            df['returns'] = df[self.labeler.price_col].pct_change()
        
        # Use either meta-positions or original positions
        pos_col = 'meta_position' if apply_meta and 'meta_position' in df.columns else self.labeler.position_col
        
        # Calculate strategy returns WITHOUT shifting positions 
        # since we're using close prices and taking positions at the close
        df['strategy_returns'] = df[pos_col] * df['returns']
        
        # Calculate cumulative returns
        df['cum_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Use cumsum instead of cumprod to avoid flat line at -1 when large losses occur
        df['cum_strategy_returns'] = df['strategy_returns'].cumsum()
        
        # Add backtest diagnostics
        logger.info("\n" + "="*50)
        logger.info(" "*15 + "BACKTEST DIAGNOSTICS")
        logger.info("="*50)
        
        # Position analysis
        total_periods = len(df)
        positions = df[pos_col].abs().sum()
        position_pct = positions / total_periods * 100
        
        # Direction analysis
        longs = (df[pos_col] > 0).sum()
        shorts = (df[pos_col] < 0).sum()
        zeros = (df[pos_col] == 0).sum()
        
        # Returns analysis
        avg_return = df['returns'].mean() * 100
        avg_strategy_return = df['strategy_returns'].mean() * 100
        
        # Risk metrics
        strategy_std = df['strategy_returns'].std() * 100
        market_std = df['returns'].std() * 100
        
        # Maximum drawdown (market)
        market_cum_returns = df['cum_returns']
        market_running_max = market_cum_returns.cummax()
        market_drawdown = market_running_max - market_cum_returns
        market_max_drawdown = market_drawdown.max() * 100
        
        # Maximum drawdown (strategy)
        strategy_cum_returns = df['cum_strategy_returns']
        strategy_running_max = strategy_cum_returns.cummax()
        strategy_drawdown = strategy_running_max - strategy_cum_returns
        strategy_max_drawdown = strategy_drawdown.max() * 100
        
        # Winning days
        winning_days = (df['strategy_returns'] > 0).sum()
        losing_days = (df['strategy_returns'] < 0).sum()
        win_rate = winning_days / (winning_days + losing_days) * 100
        
        # Display diagnostics
        logger.info(f"Total periods: {total_periods}")
        logger.info(f"Periods with positions: {positions} ({position_pct:.2f}%)")
        logger.info(f"Long positions: {longs}, Short positions: {shorts}, No position: {zeros}")
        logger.info(f"Average daily market return: {avg_return:.4f}%")
        logger.info(f"Average daily strategy return: {avg_strategy_return:.4f}%")
        logger.info(f"Market volatility (std): {market_std:.4f}%")
        logger.info(f"Strategy volatility (std): {strategy_std:.4f}%")
        logger.info(f"Market max drawdown: {market_max_drawdown:.2f}%")
        logger.info(f"Strategy max drawdown: {strategy_max_drawdown:.2f}%")
        logger.info(f"Winning days: {winning_days} ({win_rate:.2f}%)")
        logger.info(f"Losing days: {losing_days} ({100-win_rate:.2f}%)")
        
        # Final returns
        final_market_return = df['cum_returns'].iloc[-1] * 100
        final_strategy_return = df['cum_strategy_returns'].iloc[-1] * 100
        
        logger.info(f"Final market return: {final_market_return:.2f}%")
        logger.info(f"Final strategy return: {final_strategy_return:.2f}%")
        logger.info("="*50)
        
        return df
    
    def plot_backtest(self, backtest_df: pd.DataFrame, figsize=(12, 6)):
        """
        Plot backtest results.
        
        Args:
            backtest_df: DataFrame with backtest results
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        plt.plot(backtest_df['cum_returns'], label='Market Returns')
        plt.plot(backtest_df['cum_strategy_returns'], label='Strategy Returns')
        
        if 'meta_position' in backtest_df.columns:
            meta_backtest_df = self.backtest(backtest_df, apply_meta=True)
            plt.plot(meta_backtest_df['cum_strategy_returns'], label='Meta Strategy Returns')
        
        # Add model metrics to the title if available
        if hasattr(self, 'metrics') and self.metrics:
            metrics_str = f"Accuracy: {self.metrics.get('accuracy', 0):.3f}, Precision: {self.metrics.get('precision', 0):.3f}, Recall: {self.metrics.get('recall', 0):.3f}"
            plt.title(f'Backtest Results\n{metrics_str}')
        else:
            plt.title('Backtest Results')
            
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf() 