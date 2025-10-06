"""
Configuration management for AlphaZero training.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


config_dict = {
    'device': torch.device('cuda') if torch.cuda.is_available() else 'cpu',
    'n_filters': 128,              # Number of convolutional filters used in residual blocks
    'n_res_blocks': 8,             # Number of residual blocks used in network
    'exploration_constant': 2,     # Exploration constant used in PUCT calculation
    'temperature': 1.25,           # Selection temperature. A greater temperature is a more uniform distribution
    'dirichlet_alpha': 1.0,        # Alpha parameter for Dirichlet noise. Larger values mean more uniform noise
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


@dataclass
class AlphaZeroConfig:
    device: torch.device
    n_filters: int = 128
    n_res_blocks: int = 8
    exploration_constant: float = 2.0
    temperature: float = 1.25
    dirichlet_alpha: float = 1.0
    dirichlet_eps: float = 0.25
    learning_rate: float = 1e-3
    training_epochs: int = 50
    games_per_epoch: int = 100
    minibatch_size: int = 128
    n_minibatches: int = 4
    mcts_start_search_iter: int = 30
    mcts_max_search_iter: int = 150
    mcts_search_increment: int = 1
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]) -> "AlphaZeroConfig":
        # Provide sane defaults if keys are missing
        values = {**config_dict, **dictionary}
        # Ensure device is a torch.device
        device = values.get('device')
        if not isinstance(device, torch.device):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        values['device'] = device
        return cls(**values)
