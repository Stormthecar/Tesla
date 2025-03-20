import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from typing import Dict, Tuple, Optional
import logging

from src.data.preprocessor import DataPreprocessor
from src.models.predictor import StockPredictor
from src.data.market_data import MarketData
from config.trading_config import (
    INITIAL_CAPITAL,
    TRANSACTION_FEE,
    HISTORICAL_DATA_PATH,
    MODEL_SAVE_PATH,
    TRADING_START_TIME,
    TRADING_END_TIME
)

class TradingAgent:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.shares_held = 0
        self.transaction_fee = TRANSACTION_FEE
        self.trading_history = []
        self.model = StockPredictor()
        self.preprocessor = DataPreprocessor()
        self.market_data = MarketData("TSLA")
        self.est_tz = pytz.timezone('America/New_York')
        
    def load_historical_data(self, file_path: str = HISTORICAL_DATA_PATH) -> pd.DataFrame:
        """Load and preprocess historical data."""
        try:
            # Try to get real-time data first
            end_date = datetime.now(self.est_tz)
            start_date = end_date - pd.Timedelta(days=730)  # 2 years of data
            df = self.market_data.get_historical_data(start_date, end_date)
        except Exception as e:
            logging.warning(f"Failed to fetch real-time data: {e}. Falling back to CSV file.")
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    def get_current_market_data(self) -> Dict:
        """Get current market data."""
        return self.market_data.get_current_price()
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        now = datetime.now(self.est_tz)
        current_time = now.time()
        
        # Check if it's a weekday and within trading hours
        return (now.weekday() < 5 and
                TRADING_START_TIME <= current_time <= TRADING_END_TIME)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the ML model."""
        return self.preprocessor.prepare_prediction_data(df)
    
    def make_prediction(self, features: pd.DataFrame) -> Tuple[str, float]:
        """Make trading prediction based on features."""
        # Load the trained model if not loaded
        if not self.model.is_trained:
            self.model.load_model(MODEL_SAVE_PATH)
        
        # Get prediction and confidence
        expected_return, confidence = self.model.predict(features)
        
        # Make trading decision
        return self.model.make_trading_decision(expected_return, confidence)
    
    def execute_trade(self, action: str, amount: float, price: float) -> None:
        """Execute a trade and update account status."""
        if action == 'BUY':
            shares_to_buy = (amount * self.current_capital * (1 - self.transaction_fee)) / price
            self.shares_held += shares_to_buy
            self.current_capital -= amount * self.current_capital
        elif action == 'SELL':
            shares_to_sell = amount * self.shares_held
            proceeds = shares_to_sell * price * (1 - self.transaction_fee)
            self.current_capital += proceeds
            self.shares_held -= shares_to_sell
        
        self.trading_history.append({
            'timestamp': datetime.now(self.est_tz),
            'action': action,
            'amount': amount,
            'price': price,
            'capital': self.current_capital,
            'shares': self.shares_held,
            'portfolio_value': self.get_portfolio_value(price)
        })
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        return self.current_capital + (self.shares_held * current_price)
    
    def generate_daily_order(self) -> Dict:
        """Generate trading order based on current market data."""
        if not self.is_market_open():
            return {'action': 'HOLD', 'amount': 0, 'timestamp': datetime.now(self.est_tz)}
        
        # Get current market data
        current_data = self.get_current_market_data()
        
        # Get historical data and append current data
        historical_data = self.load_historical_data()
        current_df = pd.DataFrame([current_data])
        df = pd.concat([historical_data, current_df], ignore_index=True)
        
        # Prepare features and make prediction
        features = self.prepare_features(df)
        action, amount = self.make_prediction(features)
        
        return {
            'action': action,
            'amount': amount,
            'timestamp': current_data['timestamp']
        }

def main():
    # Initialize trading agent
    agent = TradingAgent()
    
    try:
        # Check if market is open
        if not agent.is_market_open():
            print("Market is currently closed.")
            return
        
        # Generate and execute order
        order = agent.generate_daily_order()
        
        # Get current price
        current_data = agent.get_current_market_data()
        current_price = current_data['close']
        
        # Execute trade if not HOLD
        if order['action'] != 'HOLD':
            agent.execute_trade(order['action'], order['amount'], current_price)
        
        # Print order and portfolio details
        print(f"\nOrder Details (as of {order['timestamp']}):")
        print(f"Action: {order['action']}")
        print(f"Amount: {order['amount']:.2%}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"\nPortfolio Status:")
        print(f"Cash: ${agent.current_capital:.2f}")
        print(f"Shares Held: {agent.shares_held:.4f}")
        print(f"Portfolio Value: ${agent.get_portfolio_value(current_price):.2f}")
        
    except Exception as e:
        logging.error(f"Error in trading execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 