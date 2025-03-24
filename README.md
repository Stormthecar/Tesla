# Tesla Stock Trading Simulation

This project implements a machine learning-based trading agent for Tesla (TSLA) stocks, with dual functionality for real-time trading during market hours and simulation testing for March 24-28, 2025. The agent uses LSTM predictions to make trading decisions while adhering to specific trading rules and constraints.

## Project Overview

### Real-Time Trading
- **Trading Hours**: 9:30 AM - 4:00 PM EST (Market Hours)
- **Price Data**: Real-time TSLA stock prices from Yahoo Finance
- **State Management**: Maintains portfolio state between trading sessions
- **Action Logging**: Records all trading decisions and executions

### Simulation Testing
- **Test Period**: March 24-28, 2025 (5 trading days)
- **Test Prices**: Predefined daily prices for testing
- **Initial Capital**: $10,000
- **Transaction Fee**: 1% per trade
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
│   └── march_simulation.py     # Trading simulation with real-time capabilities
├── models/                     # Saved model checkpoints
├── simulation_state.json       # Current portfolio state
├── trading_actions.json        # Trading history and actions
├── requirements.txt            # Project dependencies
└── README.md                  # Project documentation
```

## Trading Strategy

The trading agent employs a conservative strategy with the following features:
- LSTM-based return predictions
- 5-day moving average trend analysis
- Volatility-based trading restrictions
- Position size management (30-50% of portfolio)
- Guaranteed execution of at least one trade
- Automatic final position liquidation

### Trading Rules
1. **Buy Conditions**:
   - Initial trade: 30% of portfolio
   - Normal trades: Up to 40% of portfolio
   - Forced trade if no trades by second-to-last day
   - Minimum trade size of $100
   
2. **Sell Conditions**:
   - 90% position liquidation when selling
   - Complete liquidation on final day
   
3. **Risk Management**:
   - Portfolio exposure tracking
   - Transaction fee consideration
   - State persistence between sessions

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

### Real-Time Trading
Run the simulation during market hours (9:30 AM - 4:00 PM EST):
```bash
python -m src.march_simulation
```
The simulation will use real-time TSLA prices from Yahoo Finance.

### Testing Mode
Test specific dates in the March 24-28, 2025 period:
```bash
python -m src.march_simulation 2025-03-24  # Test March 24
python -m src.march_simulation 2025-03-25  # Test March 25
# etc...
```

### State Management
- `simulation_state.json`: Maintains portfolio state (cash, shares, trades)
- `trading_actions.json`: Records all trading actions and their results

## Performance Metrics

The simulation tracks various performance metrics:
- Daily Portfolio Value
- Cash Position
- Number of Shares
- Total Trades Executed
- Transaction Fees Paid
- Day's Return Percentage

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- torch: LSTM model implementation
- yfinance: Real-time stock data
- pytz: Timezone handling for market hours

## Contributing

This is an academic project for demonstration purposes. Feel free to use the code as a reference for similar trading simulations. 