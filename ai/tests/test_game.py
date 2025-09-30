"""
Tests for Connect Four game logic.
"""

import pytest
import numpy as np
from connect_four_ai.game import ConnectFourGame


def test_game_initialization():
    """Test game initialization."""
    game = ConnectFourGame(rows=6, cols=7)
    assert game.rows == 6
    assert game.cols == 7
    assert game.current_player == 1
    assert not game.game_over
    assert game.winner is None
    assert np.all(game.board == 0)


def test_valid_move():
    """Test making a valid move."""
    game = ConnectFourGame()
    assert game.make_move(0)
    assert game.board[5, 0] == 1  # First piece goes to bottom
    assert game.current_player == 2


def test_invalid_move():
    """Test making an invalid move."""
    game = ConnectFourGame()
    assert not game.make_move(-1)  # Invalid column
    assert not game.make_move(7)   # Invalid column
    assert game.make_move(0)       # Valid move
    # Fill column 0 completely (6 moves)
    for _ in range(5):
        game.make_move(0)  # Player 2
        game.make_move(1)  # Player 1
    assert not game.make_move(0)   # Column 0 should now be full


def test_win_condition():
    """Test win condition detection."""
    game = ConnectFourGame()
    # Create a horizontal win for player 1
    for i in range(4):
        game.make_move(i)  # Player 1
        if i < 3:  # Don't make the last move for player 2
            game.make_move(i)  # Player 2
    
    assert game.game_over
    assert game.winner == 1


def test_draw_condition():
    """Test draw condition detection."""
    game = ConnectFourGame(rows=2, cols=2)  # Small board for testing
    # Fill the board without winning
    game.make_move(0)  # Player 1
    game.make_move(1)  # Player 2
    game.make_move(1)  # Player 1
    game.make_move(0)  # Player 2
    
    assert game.game_over
    assert game.winner == 0  # Draw


def test_get_valid_moves():
    """Test getting valid moves."""
    game = ConnectFourGame()
    valid_moves = game.get_valid_moves()
    assert valid_moves == list(range(7))
    
    # Fill column 0 completely (6 moves)
    for _ in range(6):
        game.make_move(0)
    
    valid_moves = game.get_valid_moves()
    assert 0 not in valid_moves  # Column 0 is full
    assert 1 in valid_moves      # Column 1 is empty
    assert 2 in valid_moves      # Column 2 is empty


def test_reset():
    """Test game reset."""
    game = ConnectFourGame()
    game.make_move(0)
    game.reset()
    
    assert game.current_player == 1
    assert not game.game_over
    assert game.winner is None
    assert np.all(game.board == 0)
