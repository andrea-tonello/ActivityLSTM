import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.H1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.X1 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b1 = nn.Parameter(torch.Tensor(hidden_size))
        
        self.H2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.X2 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b2 = nn.Parameter(torch.Tensor(hidden_size))
        
        self.H3 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.X3 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b3 = nn.Parameter(torch.Tensor(hidden_size))
        
        self.H4 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.X4 = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.b4 = nn.Parameter(torch.Tensor(hidden_size))
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.init_weights()
        
    def init_weights(self):
        """
        Initializes the weights of the model parameters uniformly within a range determined by the hidden size.
        The weights are initialized from a uniform distribution in the range [-1/sqrt(hidden_size), 1/sqrt(hidden_size)].
        Returns:
            None
        """

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the LSTM layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor containing the hidden states for each time step,
                            of shape (batch_size, sequence_length, hidden_size).
        """
        device = x.device

        all_hidden = []
        hidden = torch.zeros((x.size(0), self.hidden_size), device=device)
        channel = torch.zeros((x.size(0), self.hidden_size), device=device)
        
        for t in range(x.size(1)):
            x_t = x[:,t,:]
            sigma1 = self.sigmoid(x_t @ self.X1 + hidden @ self.H1 + self.b1)
            channel = channel*sigma1
            sigma2 = self.sigmoid(x_t @ self.X2 + hidden @ self.H2 + self.b2)
            tanh1 = self.tanh(x_t @ self.X3 + hidden @ self.H3 + self.b3)
            channel = channel + sigma2*tanh1
            sigma3 = self.sigmoid(x_t @ self.X4 + hidden @ self.H4 + self.b4)
            hidden = self.tanh(channel) * sigma3
            all_hidden.append(hidden)
            
        return torch.stack(all_hidden, dim=1)



class ActivityLSTM(nn.Module):
    def __init__(self, input_shape, num_classes, dropout=0.3):
        super().__init__()
        
        # LSTM layers
        self.lstm1 = LSTM(input_size=input_shape, hidden_size=256)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = LSTM(input_size=256, hidden_size=128)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = LSTM(input_size=128, hidden_size=64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # LSTM layers
        x = self.lstm1(x)
        x = self.dropout1(x)
        
        x = self.lstm2(x)
        x = self.dropout2(x)
        
        x = self.lstm3(x)
        x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)