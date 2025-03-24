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
        self.cash = initial_capital
        self.transaction_fee = transaction_fee
        self.shares = 0
        self.trades = []
        self.trades_executed = 0
        self.last_trade_day = None
        
        # Initialize data processor and model
        self.data_processor = DataProcessor()
        self.data = self.data_processor.prepare_data()
        self.dates = self.data.index
        self.prices = self.data['Close']
        
        # Initialize model
        input_size = len(ModelConfig.get_all_features())
        self.predictor = StockPredictor(input_size=input_size)
        self.predictor.load_model('models/lstm_model.pth')
        self.predictor.model.eval()
        
    def calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value."""
        return self.cash + (self.shares * current_price)
    
    def calculate_portfolio_exposure(self):
        """Calculate current portfolio exposure to stock."""
        if self.shares <= 0:
            return 0
        current_price = self.prices[self.dates[-1]]
        portfolio_value = self.calculate_portfolio_value(current_price)
        return (self.shares * current_price) / portfolio_value if portfolio_value > 0 else 0
    
    def calculate_buy_amount(self, current_price, position_size):
        """Calculate number of shares to buy based on position size."""
        available_cash = self.cash * position_size
        if available_cash < 100:  # Minimum trade size
            return 0
        fee = self.transaction_fee * available_cash
        net_cash = available_cash - fee
        return int(net_cash / current_price)
    
    def calculate_trend(self, date, window=5):
        """Calculate price trend."""
        idx = self.dates.get_loc(date)
        if idx < window:
            return 0
        prices = self.prices[idx-window:idx]
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    
    def calculate_volatility(self, date, window=5):
        """Calculate price volatility."""
        idx = self.dates.get_loc(date)
        if idx < window:
            return 0
        prices = self.prices[idx-window:idx]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def predict_return(self, date):
        """Get model prediction for a specific date."""
        idx = self.dates.get_loc(date)
        
        # Get the sequence length from model config
        sequence_length = 10  # Default sequence length
        
        # Ensure we have enough data for the sequence
        if idx < sequence_length:
            return 0
            
        # Create sequence for prediction
        data_window = self.data.iloc[idx-sequence_length:idx]
        features = self.data_processor.create_sequences(data_window)[0]
        
        if len(features) == 0:
            return 0
            
        # Make prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.predictor.device)
            prediction, _ = self.predictor.model(features_tensor)
            return prediction.cpu().squeeze().item()
    
    def print_trading_metrics(self, simulation_data):
        """Print final trading metrics."""
        if not simulation_data:
            return
            
        initial_value = self.initial_capital
        final_value = simulation_data[-1]['portfolio_value']
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print("\n=== Trading Simulation Results ===")
        print(f"Period: {simulation_data[0]['date'].strftime('%Y-%m-%d')} to {simulation_data[-1]['date'].strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${initial_value:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {self.trades_executed}")
        print(f"Transaction Fees Paid: ${sum(t['fee'] for t in self.trades):,.2f}")
        
    def execute_trade(self, shares, price, action):
        """Execute a trade."""
        if shares == 0:
            return False
            
        if action == "BUY":
            cost = shares * price
            fee = cost * self.transaction_fee
            total_cost = cost + fee
            
            if total_cost > self.cash:
                return False
                
            self.cash -= total_cost
            self.shares += shares
            self.trades.append({'shares': shares, 'price': price, 'fee': fee})
            return True
            
        elif action == "SELL":
            if abs(shares) > self.shares:
                return False
                
            proceeds = abs(shares) * price
            fee = proceeds * self.transaction_fee
            net_proceeds = proceeds - fee
            
            self.cash += net_proceeds
            self.shares += shares  # shares is negative for sells
            self.trades.append({'shares': shares, 'price': price, 'fee': fee})
            return True
            
        return False
    
    def generate_trading_advice(self, date, i, total_days):
        """Generate trading advice for 9:00 AM submission."""
        predicted_return = self.predict_return(date)
        trend = self.calculate_trend(date, window=5)
        volatility = self.calculate_volatility(date, window=5)
        
        # More aggressive position sizing
        max_buy_amount = self.cash * 0.4  # Increased from 0.3 to 0.4
        
        # Lower thresholds for trading
        if predicted_return > 0.005 and trend > 0 and self.calculate_portfolio_exposure() < 0.4:  # Lowered from 0.008
            if self.cash >= 100:  # Minimum trade size check
                return f"Buy: ${min(max_buy_amount, self.cash):.2f}"
        elif predicted_return < -0.005 or trend < 0:  # Lowered from -0.008
            if self.shares > 0:
                shares_to_sell = int(self.shares * 0.9)
                if shares_to_sell > 0:
                    return f"Sell: {shares_to_sell} shares"
        
        # Force a buy on second-to-last day if no trades have occurred
        if self.trades_executed == 0 and i == total_days - 2 and self.cash >= 100:
            forced_amount = self.cash * 0.5  # More aggressive forced trade
            return f"Buy: ${forced_amount:.2f} (Forced)"
        
        return "Hold: No transaction"

    def execute_order(self, date, order_type, amount, execution_price):
        """Execute order at 10:00 AM."""
        if order_type == "Buy":
            # Convert dollar amount to shares
            shares = int(float(amount.replace("$", "")) / execution_price)
            if shares > 0:
                if self.execute_trade(shares, execution_price, "BUY"):
                    return f"Executed: Buy {shares} shares at ${execution_price:.2f}"
        elif order_type == "Sell":
            shares = int(amount)
            if shares > 0:
                if self.execute_trade(-shares, execution_price, "SELL"):
                    return f"Executed: Sell {shares} shares at ${execution_price:.2f}"
        return "No execution"

    def run_simulation(self, days=5):
        """Run the trading simulation with specific timing requirements."""
        self.trades_executed = 0
        simulation_data = []
        
        # Ensure we have enough data
        available_days = len(self.dates)
        if available_days < days:
            print(f"Warning: Only {available_days} days available, using all available data")
            days = available_days
        
        simulation_dates = self.dates[-days:]
        print(f"\nSimulating trades from {simulation_dates[0].strftime('%Y-%m-%d')} to {simulation_dates[-1].strftime('%Y-%m-%d')}")
        
        for i, date in enumerate(simulation_dates):
            print(f"\n=== Day {i+1}: {date.strftime('%Y-%m-%d')} ===")
            
            # 9:00 AM - Generate and submit trading advice
            print("\n9:00 AM - Trading Advice Submission:")
            advice = self.generate_trading_advice(date, i, days)
            print(advice)
            
            # Parse the advice
            order_type = advice.split(":")[0]
            amount = advice.split(":")[1].strip().split(" ")[0] if ":" in advice else "0"
            
            # 10:00 AM - Execute orders
            print("\n10:00 AM - Order Execution:")
            execution_price = self.prices[date]
            execution_result = self.execute_order(date, order_type, amount, execution_price)
            print(execution_result)
            
            # Record daily data
            daily_data = {
                'date': date,
                'price': execution_price,
                'advice': advice,
                'execution': execution_result,
                'portfolio_value': self.calculate_portfolio_value(execution_price),
                'cash': self.cash,
                'shares': self.shares
            }
            simulation_data.append(daily_data)
            
            print(f"\nEnd of Day Summary:")
            print(f"Portfolio Value: ${daily_data['portfolio_value']:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Shares: {self.shares}")
            
            # Handle final day liquidation
            if i == days - 1 and self.shares > 0:
                print("\nFinal Day - Liquidating Position:")
                if self.execute_trade(-self.shares, execution_price, "SELL"):
                    print(f"Sold {self.shares} shares at ${execution_price:.2f}")
                    daily_data['execution'] += f"\nLiquidated remaining {self.shares} shares"
                    daily_data['portfolio_value'] = self.calculate_portfolio_value(execution_price)
                    daily_data['cash'] = self.cash
                    daily_data['shares'] = self.shares
        
        self.print_trading_metrics(simulation_data)
        return simulation_data

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
    simulator = TradingSimulator()
    simulator.run_simulation() 