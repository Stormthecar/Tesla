"""
Enhanced LSTM model with multi-head attention and squeeze-excitation for stock price prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.model_config import ModelConfig

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        
        # Linear projections and reshape for multi-head
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.proj(attn_output)
        
        return output, attn_weights

class EnhancedResidualLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=ModelConfig.BIDIRECTIONAL
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * 2 if ModelConfig.BIDIRECTIONAL else hidden_size)
        self.se = SqueezeExcitation(hidden_size * 2 if ModelConfig.BIDIRECTIONAL else hidden_size)
        
    def forward(self, x):
        residual = x
        out, _ = self.lstm(x)
        out = self.dropout(out)
        
        # Apply squeeze-excitation
        out = out.transpose(1, 2)  # Change to (batch, channels, seq_len)
        out = self.se(out)
        out = out.transpose(1, 2)  # Change back to (batch, seq_len, channels)
        
        # Residual connection if dimensions match
        if residual.size(-1) == out.size(-1):
            out = out + residual
            
        out = self.layer_norm(out)
        return out

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Create pyramid levels
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                
        # Create top-down pathway
        self.top_down = nn.ModuleList()
        for i in range(len(hidden_sizes)-1, 0, -1):
            self.top_down.append(nn.Linear(hidden_sizes[i], hidden_sizes[i-1]))
            
    def forward(self, x):
        # Bottom-up pathway
        features = []
        for layer in self.layers:
            x = F.relu(layer(x))
            features.append(x)
            
        # Top-down pathway and lateral connections
        for i, td_layer in enumerate(self.top_down):
            top_down_feature = td_layer(features[-1-i])
            features[-2-i] = features[-2-i] + top_down_feature
            
        return torch.cat(features, dim=-1)

class EnhancedStockLSTM(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = ModelConfig.HIDDEN_SIZE
        self.num_layers = ModelConfig.NUM_LAYERS
        self.dropout = ModelConfig.DROPOUT
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Feature pyramid for initial feature extraction
        fpn_sizes = [self.hidden_size, self.hidden_size//2, self.hidden_size//4]
        self.fpn = FeaturePyramidNetwork(input_size, fpn_sizes)
        fpn_output_size = sum(fpn_sizes)  # Size after concatenation
        
        # Dimension reduction after FPN
        self.dim_reduction = nn.Sequential(
            nn.Linear(fpn_output_size, input_size),
            nn.ReLU(),
            nn.LayerNorm(input_size)
        )
        
        # Enhanced LSTM layers
        hidden_factor = 2 if ModelConfig.BIDIRECTIONAL else 1
        self.lstm_layers = nn.ModuleList([
            EnhancedResidualLSTMBlock(
                input_size if i == 0 else self.hidden_size * hidden_factor,
                self.hidden_size,
                self.dropout
            ) for i in range(self.num_layers)
        ])
        
        # Multi-head attention
        self.attention = MultiHeadAttention(self.hidden_size * hidden_factor)
        
        # Output layers
        total_hidden = self.hidden_size * hidden_factor
        self.fc1 = nn.Linear(total_hidden, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc3 = nn.Linear(self.hidden_size // 2, 1)
        
        # Skip connection projection
        self.skip_proj = nn.Linear(self.hidden_size, self.hidden_size // 2)
        
        # Normalization and dropout
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size // 2)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Feature pyramid and dimension reduction
        x = self.fpn(x)
        x = self.dim_reduction(x)
        
        # Process through LSTM layers
        lstm_out = x
        for lstm_layer in self.lstm_layers:
            lstm_out = lstm_layer(lstm_out)
        
        # Apply multi-head attention
        attended_out, attention_weights = self.attention(lstm_out)
        
        # Global average pooling
        out = torch.mean(attended_out, dim=1)
        
        # Final dense layers with skip connections
        out1 = self.fc1(out)
        out1 = self.bn1(out1)  # BatchNorm1d expects [N, C] format
        out1 = F.relu(out1)
        out1 = self.dropout1(out1)
        
        # Project skip connection to match dimensions
        skip = self.skip_proj(out1)
        
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)  # BatchNorm1d expects [N, C] format
        out2 = F.relu(out2)
        out2 = self.dropout2(out2)
        
        # Add skip connection
        out2 = out2 + skip
        
        out3 = self.fc3(out2)
        
        return out3, attention_weights

class StockPredictor:
    def __init__(self, input_size: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedStockLSTM(input_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=ModelConfig.LEARNING_RATE,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=ModelConfig.REDUCE_LR_FACTOR,
            patience=ModelConfig.REDUCE_LR_PATIENCE,
            verbose=True
        )
        
    def train_step(self, X_batch, y_batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions, _ = self.model(X_batch)
        loss = self.criterion(predictions.squeeze(), y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            ModelConfig.GRADIENT_CLIP
        )
        
        self.optimizer.step()
        return loss.item()
    
    def validate_step(self, X_batch, y_batch):
        self.model.eval()
        with torch.no_grad():
            predictions, attention_weights = self.model(X_batch)
            # Ensure predictions and y_batch have the same shape
            predictions = predictions.squeeze()
            if len(predictions.shape) == 0:  # If scalar tensor
                predictions = predictions.unsqueeze(0)
            loss = self.criterion(predictions, y_batch)
        return loss, predictions, attention_weights
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions, attention_weights = self.model(X)
        return predictions.cpu().numpy(), attention_weights.cpu().numpy()
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)
        
    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 