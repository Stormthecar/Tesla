import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Tuple, Dict, Optional
import os

class StockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,          # Increased from 100
            max_depth=15,              # Increased from 10
            min_samples_split=10,      # Increased from 5
            min_samples_leaf=4,        # Increased from 2
            max_features='sqrt',       # Added feature selection
            n_jobs=-1,                 # Use all CPU cores
            random_state=42
        )
        self.is_trained = False
        self.cv_scores = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return performance metrics."""
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_mse_scores = []
        cv_r2_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_val)
            
            # Calculate metrics
            cv_mse_scores.append(mean_squared_error(y_val, y_pred))
            cv_r2_scores.append(r2_score(y_val, y_pred))
        
        # Store cross-validation scores
        self.cv_scores = {
            'cv_mse_mean': np.mean(cv_mse_scores),
            'cv_mse_std': np.std(cv_mse_scores),
            'cv_r2_mean': np.mean(cv_r2_scores),
            'cv_r2_std': np.std(cv_r2_scores)
        }
        
        # Final training on all data
        self.model.fit(X, y)
        self.is_trained = True
        
        # Make predictions on training set
        y_pred = self.model.predict(X)
        
        # Calculate final metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            **self.cv_scores
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[float, float]:
        """
        Make predictions and return expected return and confidence.
        Returns: (expected_return, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from all trees
        predictions = []
        for estimator in self.model.estimators_:
            predictions.append(estimator.predict(X))
        
        # Calculate mean prediction and standard deviation
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence (inverse of standard deviation)
        confidence = 1 / (1 + std_pred)
        
        return mean_pred[0], confidence[0]
    
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model and additional info
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'cv_scores': self.cv_scores
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        # Load the saved state
        saved_state = joblib.load(path)
        
        # Restore the state
        self.model = saved_state['model']
        self.is_trained = saved_state['is_trained']
        self.cv_scores = saved_state.get('cv_scores', {})
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        return dict(zip(self.model.feature_names_in_, importance))
    
    def make_trading_decision(self, expected_return: float, confidence: float) -> Tuple[str, float]:
        """
        Make trading decision based on prediction and confidence.
        Returns: (action, amount)
        """
        # Define thresholds
        BUY_THRESHOLD = 0.01    # 1% expected return
        SELL_THRESHOLD = -0.01  # -1% expected return
        CONFIDENCE_THRESHOLD = 0.7
        
        if confidence < CONFIDENCE_THRESHOLD:
            return 'HOLD', 0
        
        if expected_return > BUY_THRESHOLD:
            return 'BUY', 0.95  # Use 95% of capital to account for fees
        elif expected_return < SELL_THRESHOLD:
            return 'SELL', 1.0  # Sell all shares
        else:
            return 'HOLD', 0 