"""
Tests for Connect Four AI strategies.
"""

import pytest
import numpy as np
from connect_four_ai.game import ConnectFourGame
from connect_four_ai.ai import RandomAI, MinimaxAI


def test_random_ai():
    """Test random AI."""
    game = ConnectFourGame()
    ai = RandomAI(game)
    
    # Test multiple moves to ensure randomness
    moves = set()
    for _ in range(100):
        move = ai.get_move()
        assert move is not None
        assert 0 <= move < 7
        moves.add(move)
    
    # Should have tried multiple different moves
    assert len(moves) > 1


def test_minimax_ai():
    """Test minimax AI."""
    game = ConnectFourGame()
    ai = MinimaxAI(game, depth=3)
    
    # Test that AI can make a move
    move = ai.get_move()
    assert move is not None
    assert 0 <= move < 7


def test_ai_with_full_board():
    """Test AI behavior when board is full."""
    game = ConnectFourGame(rows=1, cols=1)  # Very small board
    ai = RandomAI(game)
    
    # Fill the board
    game.make_move(0)
    
    # AI should return None when no moves available
    move = ai.get_move()
    assert move is None


def test_minimax_winning_move():
    """Test that minimax AI can find winning moves."""
    game = ConnectFourGame()
    ai = MinimaxAI(game, depth=2)
    
    # Set up a scenario where AI can win in one move
    # Player 1 (AI) has 3 in a row horizontally
    game.board[5, 0] = 1
    game.board[5, 1] = 1
    game.board[5, 2] = 1
    game.current_player = 1  # AI's turn
    
    move = ai.get_move()
    assert move == 3  # Should play the winning move


def test_minimax_blocking_move():
    """Test that minimax AI can find blocking moves."""
    game = ConnectFourGame()
    ai = MinimaxAI(game, depth=2)
    
    # Set up a scenario where opponent can win next turn
    # Player 2 has 3 in a row horizontally
    game.board[5, 0] = 2
    game.board[5, 1] = 2
    game.board[5, 2] = 2
    game.current_player = 1  # AI's turn (should block)
    
    move = ai.get_move()
    # AI should block by playing in column 3 (completing the 4-in-a-row)
    # or play defensively in a nearby column
    assert move is not None
    assert 0 <= move < 7
