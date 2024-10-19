import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor_v1(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Actor_v1, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(n_observations, 128))
        self.dropouts.append(nn.Dropout(0.0))
        
        # Hidden layers
        for _ in range(2):
            self.layers.append(nn.Linear(128, 128))
            self.dropouts.append(nn.Dropout(0.0))
        
        # Output layer
        self.layers.append(nn.Linear(128, n_actions))

    def forward(self, x, fine_tuning=False):
        if fine_tuning:
            with torch.no_grad():
                for i in range(len(self.layers) - 1):  # Apply dropout to all but the last layer
                    x = F.relu(self.layers[i](x))
                    x = self.dropouts[i](x)
        else:
            for i in range(len(self.layers) - 1):  # Apply dropout to all but the last layer
                x = F.relu(self.layers[i](x))
                x = self.dropouts[i](x)
        # Last layer without dropout
        x = self.layers[-1](x)
        
        return F.sigmoid(x)