import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, layers, activation=F.sigmoid):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = activation
    def forward(self, x):
        activation = self.activation
        for layer in self.layers[:-1]:
            x = activation(layer(x))
        return self.layers[-1](x)