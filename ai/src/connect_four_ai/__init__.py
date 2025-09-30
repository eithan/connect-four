"""
Connect Four AI Library

A Python library for implementing AI strategies for the Connect Four game.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .game import ConnectFourGame
from .ai import ConnectFourAI, RandomAI, MinimaxAI

__all__ = ["ConnectFourGame", "ConnectFourAI", "RandomAI", "MinimaxAI"]
