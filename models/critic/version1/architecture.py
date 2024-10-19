import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic_v1(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Critic_v1, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(n_observations+n_actions, 128))
        self.dropouts.append(nn.Dropout(0.0))
        
        # Hidden layers
        for _ in range(4):
            self.layers.append(nn.Linear(128, 128))
            self.dropouts.append(nn.Dropout(0.0))
        
        # Output layer
        self.layers.append(nn.Linear(128, 1))

    def forward(self, action, observation):
        x = torch.cat((action, observation), dim=1)
        for i in range(len(self.layers) - 1): 
            x = F.relu(self.layers[i](x))
            x = self.dropouts[i](x)

        x = self.layers[-1](x)
        
        return x