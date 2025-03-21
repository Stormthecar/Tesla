"""
Training script for Tesla stock price prediction model.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple

from src.data.data_processor import DataProcessor
from src.models.lstm_model import StockPredictor
from src.config.model_config import ModelConfig

def create_dataloaders(data_dict: Dict[str, np.ndarray]) -> Dict[str, torch.utils.data.DataLoader]:
    """Create PyTorch DataLoaders for training."""
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        X = torch.FloatTensor(data_dict[f'X_{split}'])
        y = torch.FloatTensor(data_dict[f'y_{split}'])
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=ModelConfig.BATCH_SIZE,
            shuffle=(split == 'train')
        )
    
    return dataloaders

def train_epoch(model: StockPredictor, dataloader: torch.utils.data.DataLoader) -> float:
    """Train for one epoch and return average loss."""
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for X_batch, y_batch in progress_bar:
        X_batch = X_batch.to(model.device)
        y_batch = y_batch.to(model.device)
        
        loss = model.train_step(X_batch, y_batch)
        total_loss += loss
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss:.6f}'})
    
    return total_loss / num_batches

def validate(model: StockPredictor, dataloader: torch.utils.data.DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate the model and return loss, predictions, and targets."""
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    all_attention_weights = []
    
    progress_bar = tqdm(dataloader, desc='Validating')
    for X_batch, y_batch in progress_bar:
        X_batch = X_batch.to(model.device)
        y_batch = y_batch.to(model.device)
        
        loss, predictions, attention_weights = model.validate_step(X_batch, y_batch)
        total_loss += loss
        num_batches += 1
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())
        all_attention_weights.extend(attention_weights.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss:.6f}'})
    
    return (
        total_loss / num_batches,
        np.array(all_predictions),
        np.array(all_targets),
        np.array(all_attention_weights)
    )

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics and directional accuracy."""
    # Basic regression metrics
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Directional accuracy (sign prediction)
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    correct_direction = np.sum((np.sign(y_true) == np.sign(y_pred)).astype(int))
    total_samples = len(y_true)
    metrics['directional_accuracy'] = (correct_direction * 100.0) / total_samples  # Convert to percentage
    
    return metrics

def plot_training_history(history: Dict[str, list], save_path: str):
    """Plot training and validation loss history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([-0.2, 0.2], [-0.2, 0.2], 'r--')
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Actual vs Predicted Returns')
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(y_true, label='Actual Returns')
    sns.kdeplot(y_pred, label='Predicted Returns')
    plt.title('Distribution of Returns')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_attention(attention_weights, feature_names, save_path):
    """Plot attention weights heatmap.
    
    Args:
        attention_weights (np.ndarray): Attention weights of shape (batch, seq_len, seq_len, num_heads)
        feature_names (list): List of feature names
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Average across batches and heads to get feature importance
    # Shape: (seq_len, seq_len)
    avg_attention = attention_weights.mean(axis=(0, -1))
    
    # Create heatmap
    sns.heatmap(
        avg_attention,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='YlOrRd'
    )
    
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Query Features')
    plt.ylabel('Key Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(epochs=50):
    """Train the LSTM model."""
    print("\n=== Training Tesla Stock Prediction Model ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize data processor and prepare data
    processor = DataProcessor()
    data = processor.prepare_data()
    
    # Create sequences
    X, y = processor.create_sequences(data)
    
    # Split data
    splits = processor.get_train_val_test_split(X, y)
    X_train, y_train = splits['X_train'], splits['y_train']
    X_val, y_val = splits['X_val'], splits['y_val']
    X_test, y_test = splits['X_test'], splits['y_test']
    
    # Initialize model and get device
    input_size = len(ModelConfig.get_all_features())
    predictor = StockPredictor(input_size=input_size)
    device = predictor.device
    print(f"\nTraining on device: {device}")
    
    # Convert to tensors and move to device
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    print("\nTraining Progress:")
    print("-" * 100)
    
    best_val_loss = float('inf')
    patience = ModelConfig.EARLY_STOPPING_PATIENCE
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_losses = []
        predictor.model.train()
        for i in range(0, len(X_train), ModelConfig.BATCH_SIZE):
            batch_X = X_train[i:i + ModelConfig.BATCH_SIZE]
            batch_y = y_train[i:i + ModelConfig.BATCH_SIZE]
            loss = predictor.train_step(batch_X, batch_y)
            train_losses.append(loss)
        
        # Validation
        val_losses = []
        predictor.model.eval()
        with torch.no_grad():
            for i in range(0, len(X_val), ModelConfig.BATCH_SIZE):
                batch_X = X_val[i:i + ModelConfig.BATCH_SIZE]
                batch_y = y_val[i:i + ModelConfig.BATCH_SIZE]
                val_loss, _, _ = predictor.validate_step(batch_X, batch_y)
                val_losses.append(val_loss.item())  # Get the scalar value directly
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate scheduler
        predictor.scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            predictor.save_model('models/lstm_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f} - "
              f"Val Loss: {avg_val_loss:.6f}")
    
    # Test evaluation
    predictor.model.eval()
    test_losses = []
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_test), ModelConfig.BATCH_SIZE):
            batch_X = X_test[i:i + ModelConfig.BATCH_SIZE]
            batch_y = y_test[i:i + ModelConfig.BATCH_SIZE]
            test_loss, pred, _ = predictor.validate_step(batch_X, batch_y)
            test_losses.append(test_loss.item())
            # Handle both single predictions and batches
            if isinstance(pred, torch.Tensor):
                pred_list = pred.cpu().squeeze().tolist()
                predictions.extend([pred_list] if isinstance(pred_list, float) else pred_list)
            else:
                predictions.append(float(pred))
    
    avg_test_loss = np.mean(test_losses)
    print(f"\nTest Loss: {avg_test_loss:.6f}")
    
    # Calculate metrics
    predictions = np.array(predictions)
    y_test = y_test.cpu().numpy()
    
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    # Calculate directional accuracy
    correct_direction = np.sum(np.sign(predictions) == np.sign(y_test))
    directional_accuracy = (correct_direction / len(y_test)) * 100
    
    print("\nTest Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    return predictor

if __name__ == "__main__":
    train_model() 