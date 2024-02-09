import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self, dims, dropout_prob=0.2):
        super(LinearProbe, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dims, 1),
            # nn.Dropout(dropout_prob),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    

class MLPProbe(nn.Module):
    def __init__(self, dims, hidden_dims=256, dropout_prob=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dims, 16)  # First hidden layer
        self.fc2 = nn.Linear(16, 16)          # Second hidden layer
        self.fc3 = nn.Linear(16, 1)            # Output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Applying ReLU activation function
        x = F.relu(self.fc2(x))  # Applying ReLU activation function
        x = self.fc3(x)          # No activation, since it's a regression problem
        return x