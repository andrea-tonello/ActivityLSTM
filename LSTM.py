import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivityLSTM(nn.Module):
    def __init__(self, input_shape, num_classes, dropout=0.3):
        super(ActivityLSTM, self).__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_shape, 
                            hidden_size=256, 
                            batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(input_size=256, 
                            hidden_size=128, 
                            batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(input_size=128, 
                            hidden_size=64, 
                            batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Get last time step

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)