"""
Configuration for the LSTM model.
"""

class ModelConfig:
    # Model parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BIDIRECTIONAL = True
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    SEQUENCE_LENGTH = 60
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 5
    GRADIENT_CLIP = 1.0
    
    @staticmethod
    def get_all_features():
        """Get list of all features used in the model."""
        return [
            'Close',
            'Returns',
            'Volume',
            'SMA_5',
            'SMA_20',
            'RSI',
            'MACD',
            'ATR',
            'ROC',
            'MFI',
            'OBV',
            'Volume_MA',
            'Volume_MA_Ratio',
            'Trend_Strength'
        ] 