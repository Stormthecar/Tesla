"""
Utility functions for model evaluation and trading strategy analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import confusion_matrix

def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate trading-specific metrics.
    
    Args:
        y_true: Array of actual returns
        y_pred: Array of predicted returns
        
    Returns:
        Dictionary of trading metrics
    """
    # Direction accuracy
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    direction_accuracy = np.mean(correct_direction)
    
    # Trading strategy returns (simple long/short based on predictions)
    strategy_returns = np.sign(y_pred) * y_true
    cumulative_returns = np.exp(np.sum(strategy_returns)) - 1
    
    # Sharpe ratio (assuming daily data)
    excess_returns = strategy_returns - 0.0001  # Assuming 0.01% daily risk-free rate
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    # Maximum drawdown
    cumulative = np.exp(np.cumsum(strategy_returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_drawdown = np.min(drawdowns)
    
    # Win rate
    wins = np.sum(strategy_returns > 0)
    total_trades = len(strategy_returns)
    win_rate = wins / total_trades
    
    # Profit factor
    gross_profits = np.sum(strategy_returns[strategy_returns > 0])
    gross_losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    return {
        'direction_accuracy': direction_accuracy,
        'cumulative_returns': cumulative_returns,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

def calculate_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate confusion matrix based metrics for directional prediction.
    
    Args:
        y_true: Array of actual returns
        y_pred: Array of predicted returns
        
    Returns:
        Dictionary of confusion metrics
    """
    # Convert to directional predictions
    y_true_dir = np.sign(y_true)
    y_pred_dir = np.sign(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_dir, y_pred_dir)
    
    # Extract values (handling the case where some classes might be missing)
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_risk_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate risk-adjusted performance metrics.
    
    Args:
        returns: Array of strategy returns
        
    Returns:
        Dictionary of risk metrics
    """
    # Annualized return (assuming daily data)
    annual_return = np.mean(returns) * 252
    
    # Annualized volatility
    annual_vol = np.std(returns) * np.sqrt(252)
    
    # Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(252)
    sortino_ratio = annual_return / downside_deviation if downside_deviation != 0 else 0
    
    # Maximum drawdown
    cumulative = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_drawdown = np.min(drawdowns)
    
    # Calmar ratio
    calmar_ratio = -annual_return / max_drawdown if max_drawdown != 0 else 0
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown
    }

def evaluate_strategy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive strategy evaluation combining all metrics.
    
    Args:
        y_true: Array of actual returns
        y_pred: Array of predicted returns
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    # Get trading signals
    signals = np.sign(y_pred)
    strategy_returns = signals * y_true
    
    # Calculate all metrics
    trading_metrics = calculate_trading_metrics(y_true, y_pred)
    confusion_metrics = calculate_confusion_metrics(y_true, y_pred)
    risk_metrics = calculate_risk_metrics(strategy_returns)
    
    # Combine all metrics
    all_metrics = {
        **trading_metrics,
        **confusion_metrics,
        **risk_metrics
    }
    
    return all_metrics 