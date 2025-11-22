import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth = 3):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim)) # Layer that maps the input to the hidden dimension
        layers.append(nn.ReLU())

        for _ in range(depth - 2): # Hidden layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, out_dim)) # Layer that maps the hidden dimension to the output dimension

        self.net = nn.Sequential(*layers) # Create the neural network

    def forward(self, x):
        return self.net(x) # Forward pass of the neural network



