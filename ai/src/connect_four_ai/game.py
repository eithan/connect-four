"""
Connect Four game logic implementation.
"""

from typing import List, Optional, Tuple
import numpy as np


class ConnectFourGame:
    """Connect Four game implementation."""
    
    def __init__(self, rows: int = 6, cols: int = 7):
        """Initialize the game board.
        
        Args:
            rows: Number of rows in the board
            cols: Number of columns in the board
        """
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        
    def make_move(self, col: int) -> bool:
        """Make a move in the specified column.
        
        Args:
            col: Column index (0-based)
            
        Returns:
            True if move was successful, False otherwise
        """
        if self.game_over or col < 0 or col >= self.cols:
            return False
            
        # Find the first empty row in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                
                # Check for win
                if self._check_win(row, col):
                    self.game_over = True
                    self.winner = self.current_player
                elif self._is_board_full():
                    self.game_over = True
                    self.winner = 0  # Draw
                else:
                    self.current_player = 3 - self.current_player  # Switch player (1->2, 2->1)
                
                return True
                
        return False
    
    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win.
        
        Args:
            row: Row of the last move
            col: Column of the last move
            
        Returns:
            True if the move resulted in a win
        """
        player = self.board[row, col]
        
        # Check horizontal
        if self._count_consecutive(row, col, 0, 1, player) >= 4:
            return True
            
        # Check vertical
        if self._count_consecutive(row, col, 1, 0, player) >= 4:
            return True
            
        # Check diagonal (top-left to bottom-right)
        if self._count_consecutive(row, col, 1, 1, player) >= 4:
            return True
            
        # Check diagonal (top-right to bottom-left)
        if self._count_consecutive(row, col, 1, -1, player) >= 4:
            return True
            
        return False
    
    def _count_consecutive(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        """Count consecutive pieces in a direction.
        
        Args:
            row: Starting row
            col: Starting column
            dr: Row direction (-1, 0, 1)
            dc: Column direction (-1, 0, 1)
            player: Player to count for
            
        Returns:
            Number of consecutive pieces
        """
        count = 1  # Count the current piece
        
        # Count in positive direction
        r, c = row + dr, col + dc
        while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
            count += 1
            r, c = r + dr, c + dc
            
        # Count in negative direction
        r, c = row - dr, col - dc
        while 0 <= r < self.rows and 0 <= c < self.cols and self.board[r, c] == player:
            count += 1
            r, c = r - dr, c - dc
            
        return count
    
    def _is_board_full(self) -> bool:
        """Check if the board is full."""
        return np.all(self.board != 0)
    
    def get_valid_moves(self) -> List[int]:
        """Get list of valid column indices for the next move.
        
        Returns:
            List of valid column indices
        """
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
    
    def get_board_state(self) -> np.ndarray:
        """Get current board state.
        
        Returns:
            Copy of the current board state
        """
        return self.board.copy()
