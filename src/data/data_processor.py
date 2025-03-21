"""
Data processing module for Tesla stock price prediction.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler
from ..config.model_config import ModelConfig

class DataProcessor:
    def __init__(self, symbol: str = "TSLA", start_date: str = None, end_date: str = None):
        """
        Initialize DataProcessor.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None
        self.processed_data = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        print(f"\nFetching {self.symbol} historical data...")
        ticker = yf.Ticker(self.symbol)
        
        if self.start_date and self.end_date:
            self.raw_data = ticker.history(start=self.start_date, end=self.end_date)
        else:
            self.raw_data = ticker.history(period="max")
            
        print(f"Downloaded data from {self.raw_data.index.min().strftime('%Y-%m-%d')} "
              f"to {self.raw_data.index.max().strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(self.raw_data)}")
        
        return self.raw_data
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price returns and log returns."""
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        # Trend Indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stochastic_K'] = stoch.stoch()
        df['Stochastic_D'] = stoch.stoch_signal()
        df['ROC'] = ta.momentum.ROCIndicator(df['Close']).roc()
        
        # Volatility Indicators
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['Bollinger_High'] = bb.bollinger_hband()
        df['Bollinger_Mid'] = bb.bollinger_mavg()
        df['Bollinger_Low'] = bb.bollinger_lband()
        
        # Volume Indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the dataset."""
        # Price-based features
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Change'] = df['Close'] - df['Open']
        
        # Volume-based features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Bollinger Band features
        df['BB_Width'] = (df['Bollinger_High'] - df['Bollinger_Low']) / df['Bollinger_Mid']
        df['BB_Position'] = (df['Close'] - df['Bollinger_Low']) / (df['Bollinger_High'] - df['Bollinger_Low'])
        
        return df
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for training and trading simulation.
        
        Returns:
            pd.DataFrame: Processed data with technical indicators
        """
        # Fetch data
        print("\nFetching TSLA historical data...")
        df = self.fetch_data()
        print(f"Downloaded data from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(df)}\n")
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['ATR'] = self.calculate_atr(df)
        df['ROC'] = self.calculate_roc(df['Close'])
        df['MFI'] = self.calculate_mfi(df)
        df['OBV'] = self.calculate_obv(df)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Trend strength
        df['Trend_Strength'] = abs(df['SMA_5'] - df['SMA_20']) / df['SMA_20']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training the model.
        
        Args:
            data (pd.DataFrame): Processed data with all features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (sequences) and y (targets)
        """
        # Get feature columns from config
        feature_cols = ModelConfig.get_all_features()
        
        # Ensure all required features are present
        missing_features = [col for col in feature_cols if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Normalize the features
        scaled_data = self.scaler.fit_transform(data[feature_cols])
        
        X, y = [], []
        for i in range(len(scaled_data) - ModelConfig.SEQUENCE_LENGTH):
            X.append(scaled_data[i:(i + ModelConfig.SEQUENCE_LENGTH)])
            y.append(data['Returns'].iloc[i + ModelConfig.SEQUENCE_LENGTH])
        
        return np.array(X), np.array(y)
    
    def get_train_val_test_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split the data into training, validation, and test sets."""
        train_size = int(len(X) * ModelConfig.TRAIN_SPLIT)
        val_size = int(len(X) * ModelConfig.VAL_SPLIT)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def analyze_10k_assignment(self, data, lookback_days=5):
        """
        Analyze 10-K assignment probabilities for the last n days.
        
        Args:
            data (pd.DataFrame): Historical price data
            lookback_days (int): Number of days to analyze
            
        Returns:
            dict: Assignment probabilities and relevant metrics
        """
        recent_data = data.tail(lookback_days)
        
        analysis = {
            'dates': recent_data.index.strftime('%Y-%m-%d').tolist(),
            'prices': recent_data['Close'].tolist(),
            'returns': recent_data['Returns'].tolist(),
            'volumes': recent_data['Volume'].tolist(),
            'volatility': recent_data['Volatility'].tolist(),
            'rsi': recent_data['RSI'].tolist(),
            'assignment_prob': []
        }
        
        # Calculate assignment probability based on multiple factors
        for idx in range(len(recent_data)):
            day_data = recent_data.iloc[idx]
            
            # Factors contributing to assignment probability
            vol_factor = min(day_data['Volume'] / data['Volume'].mean(), 2.0)
            price_trend = day_data['Returns'] > 0
            rsi_extreme = abs(day_data['RSI'] - 50) > 20
            vol_spike = day_data['Volume'] > data['Volume'].rolling(20).mean().iloc[-1] * 1.5
            trend_strength = day_data['Trend_Strength'] > data['Trend_Strength'].mean()
            
            # Weighted probability calculation
            prob = 0.0
            prob += 0.3 * float(vol_factor > 1.2)  # Volume weight
            prob += 0.2 * float(price_trend)       # Price trend weight
            prob += 0.2 * float(rsi_extreme)       # RSI weight
            prob += 0.15 * float(vol_spike)        # Volume spike weight
            prob += 0.15 * float(trend_strength)   # Trend strength weight
            
            analysis['assignment_prob'].append(round(prob * 100, 2))
        
        return analysis

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def calculate_atr(self, data, period=14):
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_roc(self, prices, period=12):
        """Calculate Rate of Change."""
        return (prices - prices.shift(period)) / prices.shift(period) * 100

    def calculate_mfi(self, df, period=14):
        """
        Calculate Money Flow Index.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            period (int): Period for MFI calculation
            
        Returns:
            pd.Series: Money Flow Index values
        """
        # Calculate typical price
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * df['Volume']
        
        # Get the positive and negative money flow
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        # Calculate price difference
        price_diff = typical_price.diff()
        
        # Set positive and negative money flows
        positive_flow[price_diff > 0] = raw_money_flow[price_diff > 0]
        negative_flow[price_diff < 0] = raw_money_flow[price_diff < 0]
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return mfi

    def calculate_obv(self, data):
        """Calculate On-Balance Volume."""
        obv = pd.Series(0, index=data.index)
        obv.iloc[0] = data['Volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv 