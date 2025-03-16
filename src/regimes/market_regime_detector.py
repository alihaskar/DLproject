import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os

warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    def __init__(self, csv_path):
        """Initialize the regime detector with data from CSV."""
        self.csv_path = csv_path
        # Read the data
        self.data = pd.read_csv(csv_path)
        # Convert DateTime to datetime type if it exists
        if 'DateTime' in self.data.columns:
            self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
            self.data.set_index('DateTime', inplace=True)
        
        # Create copy of original data
        self.original_data = self.data.copy()
        # Initialize regimes dataframe
        self.regimes_df = None
        
    def preprocess_data(self):
        """Calculate features for regime detection."""
        self.data = self.original_data.copy()
        
        # Rename columns to standard OHLC if needed
        if 'gbopen' in self.data.columns:
            self.data = self.data.rename(columns={
                'gbopen': 'open',
                'gbhigh': 'high',
                'gblow': 'low',
                'gbclose': 'close'
            })
        
        # Calculate moving averages
        self.data['ma20'] = self.data['close'].rolling(window=20).mean()
        self.data['ma50'] = self.data['close'].rolling(window=50).mean()
        
        # Calculate features for regime detection
        self.data['std_20'] = self.data['close'].rolling(window=20).std()
        self.data['close_minus_ma20'] = self.data['close'] - self.data['ma20']
        self.data['ma50_diff'] = self.data['ma50'].diff()
        
        # Drop NaN values
        self.data.dropna(inplace=True)
        
        # Create features dataframe
        self.features = self.data[['std_20', 'close_minus_ma20', 'ma50_diff']].copy()
        
        return self.data
    
    def detect_feature_based_regime(self):
        """Detect market regimes based on predefined features and rules."""
        # Make sure data is preprocessed
        if 'std_20' not in self.data.columns:
            self.preprocess_data()
        
        data = self.data.copy()
        
        # Standardize features
        scaler = StandardScaler()
        features = data[['std_20', 'close_minus_ma20', 'ma50_diff']].values
        scaled_features = scaler.fit_transform(features)
        
        # Determine regimes based on rules
        regimes = np.zeros(len(data), dtype=int)
        
        for i in range(len(data)):
            std = scaled_features[i, 0]
            close_ma20 = scaled_features[i, 1]
            ma50_diff = scaled_features[i, 2]
            
            # Trend detection
            if close_ma20 > 0.5 and ma50_diff > 0.2:
                regimes[i] = 0  # Uptrend
            elif close_ma20 < -0.5 and ma50_diff < -0.2:
                regimes[i] = 1  # Downtrend
            # Mean reversion detection
            elif abs(close_ma20) > 1.0 and np.sign(close_ma20) != np.sign(ma50_diff):
                regimes[i] = 2  # Mean reversion
            # Volatility detection
            elif std > 1.0:
                regimes[i] = 3  # Volatile
            else:
                regimes[i] = 4  # Neutral
        
        # Map numeric regimes to names
        regime_mapping = {
            0: 'uptrend', 
            1: 'downtrend', 
            2: 'mean_reversion', 
            3: 'volatile', 
            4: 'neutral'
        }
        
        data['feature_regime'] = [regime_mapping[r] for r in regimes]
        return data['feature_regime']
    
    def detect_hmm_regime(self, n_states=4):
        """Detect market regimes using Hidden Markov Model."""
        # Make sure data is preprocessed
        if 'std_20' not in self.data.columns:
            self.preprocess_data()
        
        data = self.data.copy()
        
        # Prepare data for HMM
        features = data[['std_20', 'close_minus_ma20', 'ma50_diff']].values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train HMM
        model = hmm.GaussianHMM(n_components=n_states, 
                               covariance_type="full", 
                               n_iter=100, 
                               random_state=42)
        model.fit(scaled_features)
        
        # Predict hidden states
        hidden_states = model.predict(scaled_features)
        
        # Calculate state characteristics
        state_means = model.means_
        
        # Analyze state means to determine regime characteristics
        regime_mapping = {}
        
        for i in range(n_states):
            mean_std = state_means[i, 0]
            mean_close_ma20 = state_means[i, 1]
            mean_ma50_diff = state_means[i, 2]
            
            if mean_close_ma20 > 0 and mean_ma50_diff > 0:
                regime_mapping[i] = 'uptrend'
            elif mean_close_ma20 < 0 and mean_ma50_diff < 0:
                regime_mapping[i] = 'downtrend'
            elif abs(mean_close_ma20) > abs(mean_ma50_diff) and np.sign(mean_close_ma20) != np.sign(mean_ma50_diff):
                regime_mapping[i] = 'mean_reversion'
            elif mean_std > np.mean([state_means[j, 0] for j in range(n_states)]):
                regime_mapping[i] = 'volatile'
            else:
                regime_mapping[i] = 'neutral'
        
        data['hmm_regime'] = [regime_mapping[state] for state in hidden_states]
        
        return data['hmm_regime']
    
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=2, num_layers=2, num_classes=5):
            """Simple Transformer model for regime classification."""
            super().__init__()
            
            # Embedding layer
            self.embedding = nn.Linear(input_dim, d_model)
            
            # Positional encoding is handled implicitly since we're using a small sequence
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output layer
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )
            
        def forward(self, x):
            # Input shape: [batch_size, seq_len, input_dim]
            x = self.embedding(x)
            x = self.transformer_encoder(x)
            # Get the representation of the last element in the sequence
            x = x[:, -1, :]
            # Classify
            x = self.classifier(x)
            return x
    
    def prepare_sequences(self, features, sequence_length=20):
        """Prepare sequences for transformer model."""
        sequences = []
        for i in range(len(features) - sequence_length + 1):
            sequences.append(features[i:i+sequence_length])
        return np.array(sequences)
    
    def detect_transformer_regime(self, sequence_length=20, epochs=100, batch_size=64):
        """Detect market regimes using a Transformer model."""
        # Make sure data is preprocessed
        if 'std_20' not in self.data.columns:
            self.preprocess_data()
        
        data = self.data.copy()
        
        # Prepare features
        features = data[['std_20', 'close_minus_ma20', 'ma50_diff']].values
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Create sequences
        X = self.prepare_sequences(scaled_features, sequence_length)
        
        # Use HMM to create labels for supervised learning
        hmm_regime = self.detect_hmm_regime()
        hmm_regime_numeric = pd.Series(hmm_regime).map({
            'uptrend': 0, 
            'downtrend': 1, 
            'mean_reversion': 2, 
            'volatile': 3, 
            'neutral': 4
        }).values
        
        # Adjust labels to match sequences
        y = hmm_regime_numeric[sequence_length-1:]
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[2]  # Number of features
        model = self.TransformerModel(input_dim=input_dim)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
        
        # Predict regimes for the entire dataset
        model.eval()
        with torch.no_grad():
            all_X_tensor = torch.FloatTensor(X)
            predictions = model(all_X_tensor)
            _, predicted = torch.max(predictions, 1)
            predicted = predicted.numpy()
        
        # Map numeric predictions to regime names
        regime_mapping = {
            0: 'uptrend', 
            1: 'downtrend', 
            2: 'mean_reversion', 
            3: 'volatile', 
            4: 'neutral'
        }
        
        # Create full array of predictions (account for sequence_length offset)
        full_predictions = np.zeros(len(data), dtype=int)
        full_predictions.fill(4)  # Default to neutral
        full_predictions[sequence_length-1:] = predicted
        
        data['transformer_regime'] = [regime_mapping[p] for p in full_predictions]
        
        return data['transformer_regime']
    
    def detect_all_regimes(self):
        """Detect regimes using all methods and return combined dataframe."""
        # Preprocess data
        self.preprocess_data()
        
        # Apply all detection methods
        feature_regime = self.detect_feature_based_regime()
        hmm_regime = self.detect_hmm_regime()
        transformer_regime = self.detect_transformer_regime()
        
        # Create regimes dataframe
        regimes_df = pd.DataFrame({
            'feature_regime': feature_regime,
            'hmm_regime': hmm_regime,
            'transformer_regime': transformer_regime
        })
        
        # Add OHLC price data and position
        regimes_df['open'] = self.data['open']
        regimes_df['high'] = self.data['high']
        regimes_df['low'] = self.data['low']
        regimes_df['close'] = self.data['close']
        regimes_df['position'] = self.data['position'] if 'position' in self.data.columns else None
        regimes_df['returns'] = self.data['stg'] if 'stg' in self.data.columns else None
        
        self.regimes_df = regimes_df
        return regimes_df
    
    def save_regimes_to_csv(self, output_path=None):
        """Save the detected regimes to a CSV file."""
        if self.regimes_df is None:
            self.detect_all_regimes()
        
        if output_path is None:
            # Create a filename based on the input filename
            base_dir = os.path.dirname(self.csv_path)
            base_name = os.path.basename(self.csv_path).split('.')[0]
            output_path = os.path.join(base_dir, f"{base_name}_regimes.csv")
        
        self.regimes_df.to_csv(output_path)
        print(f"Regimes saved to {output_path}")
        return output_path
    
    def plot_regimes(self, regime_column='hmm_regime', figsize=(15, 10)):
        """Plot market prices with detected regimes highlighted."""
        if self.regimes_df is None:
            self.detect_all_regimes()
        
        df = self.regimes_df.copy()
        
        # Define colors for regimes
        regime_colors = {
            'uptrend': 'green',
            'downtrend': 'red',
            'mean_reversion': 'blue',
            'volatile': 'orange',
            'neutral': 'gray'
        }
        
        # Get unique regimes and create legend handles
        unique_regimes = df[regime_column].unique()
        legend_elements = [Patch(facecolor=regime_colors[regime], label=regime) 
                          for regime in unique_regimes if regime in regime_colors]
        
        # Base path for saving figures
        base_dir = os.path.dirname(self.csv_path)
        base_name = os.path.basename(self.csv_path).split('.')[0]
        
        # PLOT 1: Price with regimes
        fig_price = plt.figure(figsize=figsize)
        
        # Plot price
        plt.plot(df.index, df['close'], 'k-', alpha=0.7, label='Price')
        
        # Add colored background for each regime period
        current_regime = None
        start_idx = 0
        
        for i, (idx, row) in enumerate(df.iterrows()):
            regime = row[regime_column]
            
            if regime != current_regime:
                if current_regime is not None:
                    end_idx = i - 1
                    plt.axvspan(df.index[start_idx], df.index[end_idx], 
                                alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
                current_regime = regime
                start_idx = i
        
        # Add the last regime segment
        if current_regime is not None:
            plt.axvspan(df.index[start_idx], df.index[-1], 
                        alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
        
        # Add labels and title
        plt.title(f'Market Price with {regime_column.replace("_", " ").title()} Regimes')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend(handles=legend_elements, loc='best')
        
        # Save price figure
        price_fig_path = os.path.join(base_dir, f"{base_name}_{regime_column}_price.png")
        plt.savefig(price_fig_path)
        print(f"Price plot saved to {price_fig_path}")
        plt.tight_layout()
        plt.show()
        
        # PLOT 2: Strategy returns with regimes (if available)
        if 'returns' in df.columns and df['returns'] is not None:
            fig_returns = plt.figure(figsize=figsize)
            plt.plot(df.index, df['returns'].cumsum(), 'b-', label='Cumulative Strategy Returns')
            
            # Add colored background for each regime
            current_regime = None
            start_idx = 0
            
            for i, (idx, row) in enumerate(df.iterrows()):
                regime = row[regime_column]
                
                if regime != current_regime:
                    if current_regime is not None:
                        end_idx = i - 1
                        plt.axvspan(df.index[start_idx], df.index[end_idx], 
                                    alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
                    current_regime = regime
                    start_idx = i
            
            # Add the last regime segment
            if current_regime is not None:
                plt.axvspan(df.index[start_idx], df.index[-1], 
                            alpha=0.2, color=regime_colors.get(current_regime, 'gray'))
            
            plt.title(f'Strategy Returns with {regime_column.replace("_", " ").title()} Regimes')
            plt.ylabel('Cumulative Returns')
            plt.grid(True, alpha=0.3)
            plt.legend(handles=legend_elements, loc='best')
            
            # Save returns figure
            returns_fig_path = os.path.join(base_dir, f"{base_name}_{regime_column}_returns.png")
            plt.savefig(returns_fig_path)
            print(f"Returns plot saved to {returns_fig_path}")
            plt.tight_layout()
            plt.show()

def main():
    """Main function to run the market regime detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect market regimes from OHLC data')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with OHLC data')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save regimes CSV')
    parser.add_argument('--plot', action='store_true', help='Plot regimes')
    parser.add_argument('--regime_type', type=str, default='hmm', 
                        choices=['feature', 'hmm', 'transformer'], 
                        help='Regime detection method to plot')
    
    args = parser.parse_args()
    
    detector = MarketRegimeDetector(args.csv_path)
    regimes_df = detector.detect_all_regimes()
    detector.save_regimes_to_csv(args.output_path)
    
    if args.plot:
        regime_column = f"{args.regime_type}_regime"
        detector.plot_regimes(regime_column=regime_column)

if __name__ == "__main__":
    main() 