import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import joblib
import os
import ta

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various technical indicators."""
        # Create copy to avoid modifying original data
        df = df.copy()
        
        # Trend Indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['Stoch_RSI'] = ta.momentum.stochrsi(df['Close'])
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volatility Indicators
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price Derivatives
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close']).diff()
        df['Price_Change'] = df['Close'] - df['Close'].shift(1)
        
        # Price Levels
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        
        # Volatility Features
        df['Return_Volatility'] = df['Daily_Return'].rolling(window=20).std()
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Gap Features
        df['Overnight_Gap'] = df['Open'] / df['Close'].shift(1) - 1
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and target for ML model."""
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Define feature columns
        self.feature_columns = [
            # Trend Indicators
            'SMA_20', 'SMA_50', 'EMA_20', 'ADX', 'MACD',
            
            # Momentum Indicators
            'RSI', 'Stoch_RSI', 'MFI',
            
            # Volatility Indicators
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'ATR',
            
            # Volume Indicators
            'OBV', 'Volume_Ratio',
            
            # Price Derivatives
            'Daily_Return', 'Log_Return', 'Price_Change',
            
            # Price Levels
            'Price_to_SMA20', 'Price_to_SMA50',
            
            # Volatility Features
            'Return_Volatility', 'High_Low_Range',
            
            # Gap Features
            'Overnight_Gap'
        ]
        
        # Create target variable (next day's return)
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Scale features
        X = df[self.feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        # Create feature DataFrame
        features_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        return features_df, df['Target']
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for making predictions."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before making predictions")
            
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Select features
        X = df[self.feature_columns].iloc[-1:]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
    def save_model(self, path: str) -> None:
        """Save the preprocessor state to disk."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the scaler and feature columns
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }, path)
        
    def load_model(self, path: str) -> None:
        """Load the preprocessor state from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No preprocessor found at {path}")
            
        # Load the saved state
        saved_state = joblib.load(path)
        
        # Restore the state
        self.scaler = saved_state['scaler']
        self.feature_columns = saved_state['feature_columns']
        self.is_fitted = saved_state['is_fitted']
    
    def get_feature_importance(self, model) -> List[Tuple[str, float]]:
        """Get feature importance from the model."""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return list(zip(self.feature_columns, importance))
        return [] 