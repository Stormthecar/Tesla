# Tesla Stock Trading Agent

A machine learning-based trading agent for Tesla (TSLA) stocks that predicts price movements and makes trading decisions.

## Project Overview

This project implements a machine learning trading agent that:
- Predicts Tesla stock price movements using historical data
- Makes trading decisions (Buy, Sell, or Hold) based on ML predictions
- Simulates trading with a starting capital of $10,000
- Operates during the specified trading period (March 24-28, 2025)

## Features

- Historical data analysis and preprocessing
- Technical indicator calculation
- Machine learning model for price prediction
- Trading strategy implementation
- Performance tracking and evaluation

## Project Structure

```
├── data/               # Data storage directory
├── src/               # Source code
│   ├── data/         # Data processing modules
│   ├── models/       # ML model implementations
│   └── trading/      # Trading strategy modules
├── config/           # Configuration files
├── notebooks/        # Jupyter notebooks for analysis
├── tests/           # Unit tests
├── venv/            # Virtual environment
└── setup.py         # Setup script
```

## Setup

### Option 1: Using the Setup Script (Recommended)

1. Clone the repository
2. Run the setup script:
   ```bash
   python setup.py
   ```
3. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source ./venv/bin/activate
     ```

### Option 2: Manual Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source ./venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Trading Agent

1. Make sure your virtual environment is activated
2. Train the model:
   ```bash
   python src/train_model.py
   ```
3. Run the trading agent:
   ```bash
   python src/main.py
   ```

## Trading Rules

- Starting capital: $10,000 USD
- Transaction fee: 1% per trade
- Trading period: March 24-28, 2025
- Order submission deadline: 9:00 AM EST
- Order execution: 10:00 AM EST

## Performance Metrics

The agent's performance is evaluated based on:
- Final account balance
- Transaction costs
- Trading accuracy
- Risk-adjusted returns

## License

MIT License 