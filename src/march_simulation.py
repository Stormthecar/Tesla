"""
Trading simulation for Tesla stock specifically for March 24-28, 2025.
Saves state between runs to continue simulation day by day.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import yfinance as yf
import pytz
from src.data.data_processor import DataProcessor
from src.models.lstm_model import StockPredictor
from src.config.model_config import ModelConfig
import torch

class MarchSimulator:
    def __init__(self, initial_capital=10000, transaction_fee=0.01, simulation_date=None):
        """Initialize trading simulator."""
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.simulation_date = simulation_date or datetime.now().strftime('%Y-%m-%d')
        
        # Define simulation period and prices (for testing)
        self.simulation_dates = pd.date_range(start='2025-03-24', end='2025-03-28', freq='B')
        self.simulation_prices = {
            '2025-03-24': 249.50,  # Monday
            '2025-03-25': 252.75,  # Tuesday
            '2025-03-26': 247.80,  # Wednesday
            '2025-03-27': 251.20,  # Thursday
            '2025-03-28': 253.90   # Friday
        }
        
        # Initialize data processor and model
        self.data_processor = DataProcessor()
        self.data = self.data_processor.prepare_data()
        
        # Initialize model
        input_size = len(ModelConfig.get_all_features())
        self.predictor = StockPredictor(input_size=input_size)
        self.predictor.load_model('models/lstm_model.pth')
        self.predictor.model.eval()
        
        # Load previous state or initialize new state
        self.state_file = 'simulation_state.json'
        self.load_state()
        
        print("\nSimulation Prices (for testing):")
        for date in self.simulation_dates:
            date_str = date.strftime('%Y-%m-%d')
            print(f"- {date_str}: ${self.simulation_prices[date_str]:.2f}")

    def is_market_open(self):
        """Check if the US stock market is currently open."""
        now = datetime.now(pytz.timezone('America/New_York'))
        
        # Check if it's a weekday
        if now.weekday() >= 5:
            return False
        
        # Market hours are 9:30 AM - 4:00 PM EST
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close

    def get_real_time_price(self):
        """Get real-time Tesla stock price."""
        try:
            tesla = yf.Ticker("TSLA")
            current_price = tesla.info['regularMarketPrice']
            print(f"Retrieved real-time TSLA price: ${current_price:.2f}")
            return current_price
        except Exception as e:
            print(f"Error fetching real-time price: {e}")
            return None

    def get_price_for_date(self, date):
        """Get price for a specific date."""
        date_str = date.strftime('%Y-%m-%d')
        
        # If we're running in real-time during market hours
        if date_str == datetime.now().strftime('%Y-%m-%d') and self.is_market_open():
            real_price = self.get_real_time_price()
            if real_price is not None:
                return real_price
        
        # Otherwise use simulation price
        return self.simulation_prices[date_str]

    def load_state(self):
        """Load previous simulation state or initialize new state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.cash = state['cash']
                self.shares = state['shares']
                self.trades = state['trades']
                self.trades_executed = state['trades_executed']
                self.last_simulated_date = state['last_simulated_date']
                print(f"\nLoaded previous state from {self.last_simulated_date}")
                print(f"Cash: ${self.cash:.2f}")
                print(f"Shares: {self.shares}")
                print(f"Trades executed: {self.trades_executed}")
        else:
            self.cash = self.initial_capital
            self.shares = 0
            self.trades = []
            self.trades_executed = 0
            self.last_simulated_date = None
            print("\nStarting new simulation")

    def save_state(self, current_date):
        """Save current simulation state."""
        state = {
            'cash': self.cash,
            'shares': self.shares,
            'trades': self.trades,
            'trades_executed': self.trades_executed,
            'last_simulated_date': current_date.strftime('%Y-%m-%d')
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=4, default=str)
        print(f"\nSaved simulation state for {current_date.strftime('%Y-%m-%d')}")

    def save_daily_action(self, action_data):
        """Save action for the current day."""
        actions_file = 'trading_actions.json'
        if os.path.exists(actions_file):
            with open(actions_file, 'r') as f:
                actions = json.load(f)
        else:
            actions = []
        
        actions.append(action_data)
        
        with open(actions_file, 'w') as f:
            json.dump(actions, f, indent=4, default=str)
        print(f"\nSaved action to {actions_file}")

    def calculate_portfolio_value(self, date):
        """Calculate total portfolio value."""
        current_price = self.get_price_for_date(date)
        return self.cash + (self.shares * current_price)
    
    def calculate_portfolio_exposure(self, date):
        """Calculate current portfolio exposure to stock."""
        if self.shares <= 0:
            return 0
        current_price = self.get_price_for_date(date)
        portfolio_value = self.calculate_portfolio_value(date)
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
        try:
            idx = self.data.index.get_loc(date)
        except KeyError:
            # Find the closest previous trading day
            valid_dates = self.data.index[self.data.index <= date]
            if len(valid_dates) > 0:
                date = valid_dates[-1]
                idx = self.data.index.get_loc(date)
            else:
                return 0
                
        if idx < window:
            return 0
        prices = self.data['Close'][idx-window:idx]
        return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    
    def calculate_volatility(self, date, window=5):
        """Calculate price volatility."""
        try:
            idx = self.data.index.get_loc(date)
        except KeyError:
            # Find the closest previous trading day
            valid_dates = self.data.index[self.data.index <= date]
            if len(valid_dates) > 0:
                date = valid_dates[-1]
                idx = self.data.index.get_loc(date)
            else:
                return 0
                
        if idx < window:
            return 0
        prices = self.data['Close'][idx-window:idx]
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def predict_return(self, date):
        """Get model prediction for a specific date."""
        try:
            idx = self.data.index.get_loc(date)
        except KeyError:
            print(f"Warning: Date {date.strftime('%Y-%m-%d')} not found in data")
            # Find the closest previous trading day
            valid_dates = self.data.index[self.data.index <= date]
            if len(valid_dates) > 0:
                date = valid_dates[-1]
                idx = self.data.index.get_loc(date)
                print(f"Using closest available date: {date.strftime('%Y-%m-%d')}")
            else:
                return 0
        
        sequence_length = 10
        if idx < sequence_length:
            return 0
            
        data_window = self.data.iloc[idx-sequence_length:idx]
        features = self.data_processor.create_sequences(data_window)[0]
        
        if len(features) == 0:
            return 0
            
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.predictor.device)
            prediction, _ = self.predictor.model(features_tensor)
            return prediction.cpu().squeeze().item()
    
    def generate_trading_advice(self, date, i, total_days):
        """Generate trading advice for 9:00 AM submission."""
        current_price = self.get_price_for_date(date)
        
        # Force a trade on the first day to ensure at least one trade occurs
        if i == 0 and self.cash >= 100:
            forced_amount = self.cash * 0.3  # Use 30% of cash
            return f"Buy: ${forced_amount:.2f} (Initial)"
        
        # If we haven't traded yet by the second-to-last day, force a trade
        if self.trades_executed == 0 and i == total_days - 2 and self.cash >= 100:
            forced_amount = self.cash * 0.5  # More aggressive on second-to-last day
            return f"Buy: ${forced_amount:.2f} (Forced)"
        
        # Normal trading logic
        max_buy_amount = self.cash * 0.4
        if self.cash >= 100 and self.calculate_portfolio_exposure(date) < 0.4:
            return f"Buy: ${min(max_buy_amount, self.cash):.2f}"
        elif self.shares > 0:
            shares_to_sell = int(self.shares * 0.9)
            if shares_to_sell > 0:
                return f"Sell: {shares_to_sell} shares"
        
        return "Hold: No transaction"

    def execute_order(self, date, order_type, amount):
        """Execute order at 10:00 AM."""
        current_price = self.get_price_for_date(date)
        if order_type == "Buy":
            shares = int(float(amount.replace("$", "")) / current_price)
            if shares > 0:
                if self.execute_trade(shares, current_price, "BUY"):
                    return f"Executed: Buy {shares} shares at ${current_price:.2f}"
        elif order_type == "Sell":
            shares = int(amount)
            if shares > 0:
                if self.execute_trade(-shares, current_price, "SELL"):
                    return f"Executed: Sell {shares} shares at ${current_price:.2f}"
        return "No execution"

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
            self.trades_executed += 1
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
            self.trades_executed += 1
            return True
            
        return False

    def run_march_simulation(self):
        """Run the trading simulation for the current day only."""
        current_date = pd.Timestamp(self.simulation_date)
        
        # For demo purposes, allow simulating any date in our range
        valid_dates = [pd.Timestamp(d) for d in self.simulation_prices.keys()]
        if current_date not in valid_dates:
            print(f"Current date {current_date.strftime('%Y-%m-%d')} is not in simulation period")
            print("For demo purposes, please specify a date between Mar 24-28, 2025")
            return
            
        # Check if we already simulated this date
        if self.last_simulated_date:
            last_date = pd.Timestamp(self.last_simulated_date)
            if current_date <= last_date:
                print(f"Already simulated trading for {current_date.strftime('%Y-%m-%d')}")
                return
            if current_date > last_date + timedelta(days=1):
                print(f"Cannot skip days. Please simulate {(last_date + timedelta(days=1)).strftime('%Y-%m-%d')} first")
                return
        
        print("\n=== Tesla Stock Trading Simulation ===")
        print(f"Date: {current_date.strftime('%Y-%m-%d')}")
        print(f"Portfolio Value: ${self.calculate_portfolio_value(current_date):.2f}")
        
        current_price = self.get_price_for_date(current_date)
        print(f"Current Price: ${current_price:.2f}")
        
        # 9:00 AM - Generate and submit trading advice
        print("\n9:00 AM - Trading Advice Submission:")
        day_index = valid_dates.index(current_date)
        advice = self.generate_trading_advice(current_date, day_index, len(valid_dates))
        print(advice)
        
        # Parse the advice
        order_type = advice.split(":")[0]
        amount = advice.split(":")[1].strip().split(" ")[0] if ":" in advice else "0"
        
        # 10:00 AM - Execute orders
        print("\n10:00 AM - Order Execution:")
        execution_result = self.execute_order(current_date, order_type, amount)
        print(execution_result)
        
        # Record daily action
        daily_action = {
            'date': current_date.strftime('%Y-%m-%d'),
            'price': current_price,
            'advice': advice,
            'execution_result': execution_result,
            'portfolio_value': self.calculate_portfolio_value(current_date),
            'cash': self.cash,
            'shares': self.shares
        }
        
        # Save daily action
        self.save_daily_action(daily_action)
        
        print(f"\nEnd of Day Summary:")
        print(f"Portfolio Value: ${daily_action['portfolio_value']:.2f}")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Shares: {self.shares}")
        
        # Handle final day
        if current_date == valid_dates[-1] and self.shares > 0:
            print("\nFinal Day - Liquidating Position:")
            if self.execute_trade(-self.shares, current_price, "SELL"):
                print(f"Sold {self.shares} shares at ${current_price:.2f}")
                daily_action['execution_result'] += f"\nLiquidated remaining {self.shares} shares"
                daily_action['portfolio_value'] = self.calculate_portfolio_value(current_date)
                daily_action['cash'] = self.cash
                daily_action['shares'] = self.shares
                self.save_daily_action(daily_action)
        
        # Save state
        self.save_state(current_date)
        
        # Print daily metrics
        total_fees = sum(trade['fee'] for trade in self.trades)
        print("\n=== Daily Trading Results ===")
        print(f"Portfolio Value: ${daily_action['portfolio_value']:.2f}")
        print(f"Day's Return: {((daily_action['portfolio_value'] - self.initial_capital) / self.initial_capital) * 100:.2f}%")
        print(f"Total Trades: {self.trades_executed}")
        print(f"Total Fees Paid: ${total_fees:.2f}")

if __name__ == "__main__":
    import sys
    simulation_date = sys.argv[1] if len(sys.argv) > 1 else None
    simulator = MarchSimulator(simulation_date=simulation_date)
    simulator.run_march_simulation() 