from datetime import datetime, time
import pytz

# Trading Parameters
INITIAL_CAPITAL = 10000.0
TRANSACTION_FEE = 0.01  # 1%

# Trading Schedule
TRADING_START_TIME = time(9, 30)  # 9:30 AM EST
TRADING_END_TIME = time(16, 0)    # 4:00 PM EST
ORDER_SUBMISSION_DEADLINE = time(9, 0)  # 9:00 AM EST
ORDER_EXECUTION_TIME = time(10, 0)      # 10:00 AM EST

# Trading Period
TRADING_START_DATE = datetime(2025, 3, 24, tzinfo=pytz.timezone('America/New_York'))
TRADING_END_DATE = datetime(2025, 3, 28, tzinfo=pytz.timezone('America/New_York'))

# Model Parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Trading Strategy Parameters
BUY_THRESHOLD = 0.01    # 1% expected return
SELL_THRESHOLD = -0.01  # -1% expected return
CONFIDENCE_THRESHOLD = 0.7

# Technical Indicators Parameters
TECHNICAL_PARAMS = {
    'SMA_20': 20,
    'SMA_50': 50,
    'EMA_20': 20,
    'RSI': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BB_PERIOD': 20,
    'BB_STD': 2,
    'VOLUME_SMA': 20
}

# Feature Columns
FEATURE_COLUMNS = [
    'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'Signal_Line',
    'BB_middle', 'BB_upper', 'BB_lower', 'Volume_Ratio',
    'Daily_Return', 'Price_Change'
]

# Model Paths
MODEL_SAVE_PATH = 'models/trained_model.joblib'
PREPROCESSOR_SAVE_PATH = 'models/preprocessor.joblib'

# Data Paths
HISTORICAL_DATA_PATH = 'TSLA.csv'
DAILY_ORDERS_PATH = 'data/daily_orders.csv'
PERFORMANCE_METRICS_PATH = 'data/performance_metrics.csv' 