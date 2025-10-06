"""
Neural network architecture components for AlphaZero.
"""

import torch
from torch import nn


class ResNet(nn.Module):
    """Complete residual neural network model."""
    
    def __init__(self, game, config):
        super().__init__()

        # Board dimensions
        self.board_size = (game.rows, game.cols)
        n_actions = game.cols  # Number of columns represent possible actions
        n_filters = config.n_filters
        
        self.base = ConvBase(config)  # Base layers

        # Policy head for choosing actions
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//4 * self.board_size[0] * self.board_size[1], n_actions)
        )

        # Value head for evaluating board states
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//32 * self.board_size[0] * self.board_size[1], 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Avoid using view with symbolic shapes for ONNX dynamo, prefer reshape
        x = self.base(x)
        x_value = self.value_head(x)
        x_policy = self.policy_head(x)
        return x_value, x_policy


class ConvBase(nn.Module):
    """Convolutional base for the network."""
    
    def __init__(self, config):
        super().__init__()
        
        n_filters = config.n_filters
        n_res_blocks = config.n_res_blocks

        # Initial convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )

        # List of residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(n_filters) for _ in range(n_res_blocks)]
        )

    def forward(self, x):
        x = self.conv(x)
        for block in self.res_blocks:
            x = block(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block, the backbone of a ResNet."""
    
    def __init__(self, n_filters):
        super().__init__()

        # Two convolutional layers, both with batch normalization
        self.conv_1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass x through layers and add skip connection
        output = self.relu(self.batch_norm_1(self.conv_1(x)))
        output = self.batch_norm_2(self.conv_2(output))
        return self.relu(output + x)
