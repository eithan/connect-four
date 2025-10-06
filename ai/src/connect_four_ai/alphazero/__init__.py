"""
AlphaZero implementation for Connect Four.
All credit for code goes to Kaggle Master auxeno at https://www.kaggle.com/code/auxeno/alphazero-connect-4-rl/notebook.

This module provides his complete Connect Four AlphaZero implementation broken apart into classes, including:
- Game engine (Connect4)
- Neural network architecture (ResNet)
- Monte Carlo Tree Search (MCTS)
- Training utilities (AlphaZeroTrainer)
- Playing agent (AlphaZeroAgent)
- Evaluation utilities (Evaluator)
"""

from .config import AlphaZeroConfig, config_dict
from .game import Connect4, ConnectFour
from .network import ResNet, ConvBase, ResidualBlock
from .mcts import MCTS, TreeNode
from .trainer import AlphaZeroTrainer
from .agent import AlphaZeroAgent
from .evaluator import Evaluator
from .onnxalphazeronetwork import ONNXAlphaZeroNetwork

__all__ = [
    'AlphaZeroConfig',
    'config_dict', 
    'Connect4',
    'ConnectFour',
    'ResNet',
    'ConvBase', 
    'ResidualBlock',
    'MCTS',
    'TreeNode',
    'AlphaZeroTrainer',
    'AlphaZeroAgent',
    'Evaluator',
    'ONNXAlphaZeroNetwork'
]
