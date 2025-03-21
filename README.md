# Tesla Stock Trading Simulation

This project implements a machine learning-based trading agent for Tesla (TSLA) stocks, focusing on the simulation period of March 24-28, 2025. The agent uses LSTM predictions to make trading decisions while adhering to specific trading rules and constraints.

## Project Overview

- **Simulation Period**: March 24-28, 2025 (5 trading days)
- **Initial Capital**: $10,000
- **Transaction Fee**: 1% per trade
- **Trading Rules**: Daily order submission with buy/sell/hold decisions
- **Minimum Trade Size**: $100

## Project Structure

```
├── src/
│   ├── config/
│   │   └── model_config.py     # Model configuration and feature definitions
│   ├── data/
│   │   └── data_processor.py   # Data preparation and technical indicators
│   ├── models/
│   │   └── lstm_model.py       # LSTM model architecture
│   ├── utils/
│   │   └── metrics.py          # Trading performance metrics
│   ├── train.py                # Model training script
│   └── trading_simulation.py   # Trading simulation implementation
├── models/                     # Saved model checkpoints
├── requirements.txt            # Project dependencies
└── README.md                  # Project documentation
```

## Trading Strategy

The trading agent employs a conservative strategy with the following features:
- LSTM-based return predictions
- 5-day moving average trend analysis
- Volatility-based trading restrictions
- Position size management
- Trading cooldown periods
- Automatic final position liquidation

### Trading Rules
1. **Buy Conditions**:
   - Predicted return > 0.8%
   - Positive trend confirmation
   - Maximum 30% portfolio exposure
   - 15% position sizing
   
2. **Sell Conditions**:
   - Predicted return < -0.8%
   - Negative trend
   - 90% position liquidation
   
3. **Risk Management**:
   - 2-day trading cooldown between trades
   - No trading when volatility > 2%
   - Minimum trade size of $100
   - Maximum drawdown monitoring

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the LSTM model:
```bash
python -m src.train
```

2. Run the trading simulation:
```bash
python -m src.trading_simulation
```

## Performance Metrics

The simulation tracks various performance metrics:
- Total Return
- Number of Trades
- Transaction Fees
- Portfolio Value
- Sharpe Ratio
- Maximum Drawdown

## Model Features

The LSTM model uses various technical indicators including:
- Price returns
- Moving averages
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators
- Trend strength indicators

## Contributing

This is an academic project for demonstration purposes. Feel free to use the code as a reference for similar trading simulations. 