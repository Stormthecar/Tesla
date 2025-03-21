"""
Trading simulation for Tesla stock using ML predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_processor import DataProcessor
from src.models.lstm_model import StockPredictor
from src.config.model_config import ModelConfig
import torch

class TradingSimulator:
    def __init__(self, initial_capital=10000, transaction_fee=0.01):
        """
        Initialize trading simulator.
        
        Args:
            initial_capital (float): Starting capital in USD
            transaction_fee (float): Transaction fee as a percentage (e.g., 0.01 for 1%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.shares = 0
        self.trades = []
        self.daily_portfolio_value = []
        
    def calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value."""
        return self.current_capital + (self.shares * current_price)
    
    def execute_trade(self, action, amount, price, timestamp):
        """
        Execute a trade based on the action and amount.
        
        Args:
            action (str): 'buy' or 'sell'
            amount (float): For buy: dollar amount, For sell: number of shares
            price (float): Current stock price
            timestamp (datetime): Time of trade
            
        Returns:
            bool: Whether the trade was executed successfully
        """
        if action == 'buy':
            # Calculate maximum shares we can buy with the amount
            fee = self.transaction_fee * amount
            available_amount = amount - fee
            
            if available_amount <= 0 or amount > self.current_capital:
                return False
                
            shares_to_buy = available_amount / price
            total_cost = amount  # Amount includes the fee
            
            self.shares += shares_to_buy
            self.current_capital -= total_cost
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'buy',
                'shares': shares_to_buy,
                'price': price,
                'fee': fee,
                'total_value': total_cost
            })
            return True
            
        elif action == 'sell':
            if amount <= 0 or amount > self.shares:
                return False
                
            sale_value = amount * price
            fee = self.transaction_fee * sale_value
            net_proceeds = sale_value - fee
            
            self.shares -= amount
            self.current_capital += net_proceeds
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'sell',
                'shares': amount,
                'price': price,
                'fee': fee,
                'total_value': net_proceeds
            })
            return True
            
        return False
    
    def get_trading_metrics(self):
        """Calculate trading performance metrics."""
        if not self.daily_portfolio_value:
            return {}
        
        initial_value = self.initial_capital
        final_value = self.daily_portfolio_value[-1]
        returns = (final_value - initial_value) / initial_value
        
        daily_returns = pd.Series([
            (v2 - v1) / v1 
            for v1, v2 in zip(self.daily_portfolio_value[:-1], self.daily_portfolio_value[1:])
        ])
        
        metrics = {
            'total_return': returns * 100,
            'total_trades': len(self.trades),
            'total_fees_paid': sum(trade['fee'] for trade in self.trades),
            'final_portfolio_value': final_value,
            'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(),
        }
        
        return metrics
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from portfolio value history."""
        if not self.daily_portfolio_value:
            return 0
        
        peak = self.daily_portfolio_value[0]
        max_drawdown = 0
        
        for value in self.daily_portfolio_value:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100

def run_simulation(days=5):
    """
    Run trading simulation for the specified number of days.
    
    Args:
        days (int): Number of last trading days to simulate
    """
    # Initialize components
    processor = DataProcessor()
    data = processor.prepare_data()
    simulator = TradingSimulator()
    
    # Get feature dimension from config
    input_size = len(ModelConfig.get_all_features())
    
    # Load the trained model
    predictor = StockPredictor(input_size=input_size)
    predictor.load_model('models/lstm_model.pth')
    predictor.model.eval()
    
    # Get last n days of data
    simulation_data = data.tail(days)
    
    print("\n=== Trading Simulation ===")
    print(f"Period: {simulation_data.index[0].strftime('%Y-%m-%d')} to {simulation_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${simulator.initial_capital:,.2f}")
    print(f"Transaction Fee: {simulator.transaction_fee*100}%")
    print("\nDaily Trading Activity:")
    print("-" * 100)
    
    # Create sequences for prediction
    feature_data = processor.create_sequences(data)[0]
    
    # Track moving average of predictions for trend
    prediction_history = []
    price_history = []
    last_trade_day = None
    cooldown_days = 2  # Wait at least 2 days between trades
    
    # Simulate trading for each day
    for i, (timestamp, row) in enumerate(simulation_data.iterrows()):
        current_price = row['Close']
        price_history.append(current_price)
        
        # Get model prediction
        with torch.no_grad():
            features = torch.FloatTensor(feature_data[-(days-i)-1:-(days-i)]).to(predictor.device) if i < days-1 else torch.FloatTensor(feature_data[-1:]).to(predictor.device)
            prediction, _ = predictor.model(features)
            prediction = prediction.cpu().squeeze().item()
        
        # Track prediction history
        prediction_history.append(prediction)
        
        # Calculate prediction trend (5-day moving average)
        pred_trend = np.mean(prediction_history[-5:]) if len(prediction_history) >= 5 else prediction
        
        # Calculate price volatility
        price_volatility = np.std(price_history[-5:]) / np.mean(price_history[-5:]) if len(price_history) >= 5 else 0
        
        # Check if we're in cooldown period
        in_cooldown = False
        if last_trade_day is not None:
            days_since_trade = (timestamp - last_trade_day).days
            in_cooldown = days_since_trade < cooldown_days
        
        # Generate trading signal based on prediction, trend, and conditions
        signal = generate_trading_signal(
            prediction, 
            pred_trend, 
            price_volatility,
            threshold=0.008,  # Increased threshold to 0.8%
            volatility_threshold=0.02  # Don't trade if 5-day volatility > 2%
        )
        
        # Don't trade if in cooldown
        if in_cooldown:
            signal = 'hold'
        
        # Execute trade based on signal
        portfolio_value = simulator.calculate_portfolio_value(current_price)
        trade_executed = False
        
        if signal == 'buy' and simulator.current_capital > 0:
            # Only buy if we don't already have a significant position
            current_exposure = (simulator.shares * current_price) / portfolio_value
            if current_exposure < 0.3:  # Reduced maximum exposure to 30%
                # Buy with 15% of available capital if prediction is positive
                buy_amount = min(simulator.current_capital * 0.15, simulator.current_capital)
                if buy_amount >= 100:  # Minimum trade size of $100
                    trade_executed = simulator.execute_trade('buy', buy_amount, current_price, timestamp)
        
        elif signal == 'sell' and simulator.shares > 0:
            # Sell 90% of shares if prediction is negative
            shares_to_sell = simulator.shares * 0.9
            if shares_to_sell * current_price >= 100:  # Minimum trade size of $100
                trade_executed = simulator.execute_trade('sell', shares_to_sell, current_price, timestamp)
        
        # Update last trade day if a trade was executed
        if trade_executed:
            last_trade_day = timestamp
        
        # Record daily portfolio value
        portfolio_value = simulator.calculate_portfolio_value(current_price)
        simulator.daily_portfolio_value.append(portfolio_value)
        
        # Print daily summary
        print(f"Date: {timestamp.strftime('%Y-%m-%d')}")
        print(f"Price: ${current_price:.2f}")
        print(f"Predicted Return: {prediction:.4f}")
        print(f"Prediction Trend: {pred_trend:.4f}")
        print(f"Price Volatility: {price_volatility:.4f}")
        print(f"Action: {signal.upper()}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Cash: ${simulator.current_capital:,.2f}")
        print(f"Shares: {simulator.shares:.4f}")
        if in_cooldown:
            print(f"In trade cooldown ({cooldown_days - days_since_trade} days remaining)")
        print("-" * 100)
    
    # Sell all remaining shares at the end
    if simulator.shares > 0:
        simulator.execute_trade('sell', simulator.shares, current_price, timestamp)
        final_portfolio_value = simulator.calculate_portfolio_value(current_price)
        simulator.daily_portfolio_value[-1] = final_portfolio_value
    
    # Calculate and display final metrics
    metrics = simulator.get_trading_metrics()
    
    print("\nFinal Trading Metrics:")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Number of Trades: {metrics['total_trades']}")
    print(f"Total Fees Paid: ${metrics['total_fees_paid']:.2f}")
    print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")

def generate_trading_signal(predicted_return, pred_trend, price_volatility, threshold=0.008, volatility_threshold=0.02):
    """
    Generate trading signal based on model prediction, trend, and market conditions.
    
    Args:
        predicted_return (float): Predicted return from the model
        pred_trend (float): Average of recent predictions
        price_volatility (float): Current price volatility
        threshold (float): Minimum return threshold for trading
        volatility_threshold (float): Maximum allowed volatility for trading
        
    Returns:
        str: 'buy', 'sell', or 'hold'
    """
    # Don't trade if volatility is too high
    if price_volatility > volatility_threshold:
        return 'hold'
        
    # Strong buy signal: both current prediction and trend are significantly positive
    if predicted_return > threshold and pred_trend > threshold * 0.75:
        return 'buy'
    # Strong sell signal: either current prediction or trend is significantly negative
    elif predicted_return < -threshold * 0.75 or pred_trend < -threshold:
        return 'sell'
    else:
        return 'hold'

if __name__ == "__main__":
    run_simulation()  # Default is last 5 trading days 