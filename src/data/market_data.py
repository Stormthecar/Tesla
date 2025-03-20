import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
from typing import Optional, Dict
import logging

class MarketData:
    def __init__(self, symbol: str = "TSLA"):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.est_tz = pytz.timezone('America/New_York')
        
    def get_current_price(self) -> Dict[str, float]:
        """Get the current market price and trading volume."""
        try:
            # Get real-time data (1m interval for the last 1 day)
            data = self.ticker.history(period="1d", interval="1m")
            if data.empty:
                raise ValueError("No data received from Yahoo Finance")
            
            latest = data.iloc[-1]
            return {
                'timestamp': datetime.now(self.est_tz),
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'close': latest['Close'],
                'volume': latest['Volume']
            }
        except Exception as e:
            logging.error(f"Error fetching current price: {str(e)}")
            raise
    
    def get_historical_data(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          interval: str = "1d") -> pd.DataFrame:
        """
        Get historical market data.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        """
        try:
            # If dates not provided, get last 2 years of data
            if start_date is None:
                start_date = datetime.now(self.est_tz) - timedelta(days=730)
            if end_date is None:
                end_date = datetime.now(self.est_tz)
            
            # Download historical data
            df = self.ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                raise ValueError("No historical data received from Yahoo Finance")
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Ensure all required columns are present
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Available columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def get_company_info(self) -> Dict:
        """Get company information and key statistics."""
        try:
            info = self.ticker.info
            return {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
            }
        except Exception as e:
            logging.error(f"Error fetching company info: {str(e)}")
            raise 