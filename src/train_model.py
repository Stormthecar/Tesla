import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from src.data.preprocessor import DataPreprocessor
from src.models.predictor import StockPredictor
from config.trading_config import (
    MODEL_PARAMS, TECHNICAL_PARAMS, FEATURE_COLUMNS,
    MODEL_SAVE_PATH, PREPROCESSOR_SAVE_PATH,
    HISTORICAL_DATA_PATH, PERFORMANCE_METRICS_PATH
)

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for training."""
    # Load historical data
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare features and target
    X, y = preprocessor.prepare_features(df)
    
    # Save preprocessor
    preprocessor.save_model(PREPROCESSOR_SAVE_PATH)
    
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[StockPredictor, Dict[str, float]]:
    """Train the model and return performance metrics."""
    # Initialize and train model
    model = StockPredictor()
    metrics = model.train(X, y)
    
    # Save model
    model.save_model(MODEL_SAVE_PATH)
    
    return model, metrics

def plot_feature_importance(model: StockPredictor) -> None:
    """Plot feature importance scores."""
    importance = model.get_feature_importance()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(importance.values()), y=list(importance.keys()))
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()

def plot_prediction_distribution(model: StockPredictor, X: pd.DataFrame, y: pd.Series) -> None:
    """Plot distribution of actual vs predicted returns."""
    y_pred = model.model.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=50, alpha=0.5, label='Actual Returns')
    plt.hist(y_pred, bins=50, alpha=0.5, label='Predicted Returns')
    plt.title('Distribution of Actual vs Predicted Returns')
    plt.xlabel('Return')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()

def save_metrics(metrics: Dict[str, float]) -> None:
    """Save model performance metrics."""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(PERFORMANCE_METRICS_PATH, index=False)

def main():
    # Create necessary directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Loading and preparing data...")
    X, y = load_and_prepare_data()
    
    print("Training model...")
    model, metrics = train_model(X, y)
    
    print("Generating visualizations...")
    plot_feature_importance(model)
    plot_prediction_distribution(model, X, y)
    
    print("Saving metrics...")
    save_metrics(metrics)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 