"""
Configuration management for AlphaZero training.
"""

import torch


config_dict = {
    'device': torch.device('cuda') if torch.cuda.is_available() else 'cpu',
    'n_filters': 128,              # Number of convolutional filters used in residual blocks
    'n_res_blocks': 8,             # Number of residual blocks used in network
    'exploration_constant': 2,     # Exploration constant used in PUCT calculation
    'temperature': 1.25,           # Selection temperature. A greater temperature is a more uniform distribution
    'dirichlet_alpha': 1.,         # Alpha parameter for Dirichlet noise. Larger values mean more uniform noise
    'dirichlet_eps': 0.25,         # Weight of dirichlet noise
    'learning_rate': 0.001,        # Adam learning rate
    'training_epochs': 50,         # How many full training epochs
    'games_per_epoch': 100,        # How many self-played games per epoch
    'minibatch_size': 128,         # Size of each minibatch used in learning update 
    'n_minibatches': 4,            # How many minibatches to accumulate per learning step
    'mcts_start_search_iter': 30,  # Number of Monte Carlo tree search iterations initially
    'mcts_max_search_iter': 150,   # Maximum number of MCTS iterations
    'mcts_search_increment': 1,    # After each epoch, how much should search iterations be increased by
}


class Config:
    """Configuration class that converts a dictionary to an object with attributes."""
    
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
