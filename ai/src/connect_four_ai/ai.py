"""
AI strategies for Connect Four game.
"""

from typing import List, Optional
import numpy as np
from .game import ConnectFourGame


class ConnectFourAI:
    """Base class for Connect Four AI strategies."""
    
    def __init__(self, game: ConnectFourGame):
        """Initialize AI with a game instance.
        
        Args:
            game: ConnectFourGame instance
        """
        self.game = game
    
    def get_move(self) -> Optional[int]:
        """Get the AI's next move.
        
        Returns:
            Column index for the move, or None if no valid moves
        """
        raise NotImplementedError("Subclasses must implement get_move method")


class RandomAI(ConnectFourAI):
    """Random AI that makes random valid moves."""
    
    def get_move(self) -> Optional[int]:
        """Get a random valid move.
        
        Returns:
            Random valid column index, or None if no valid moves
        """
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None
        return np.random.choice(valid_moves)


class MinimaxAI(ConnectFourAI):
    """Minimax AI with alpha-beta pruning."""
    
    def __init__(self, game: ConnectFourGame, depth: int = 4):
        """Initialize Minimax AI.
        
        Args:
            game: ConnectFourGame instance
            depth: Search depth for minimax algorithm
        """
        super().__init__(game)
        self.depth = depth
    
    def get_move(self) -> Optional[int]:
        """Get the best move using minimax algorithm.
        
        Returns:
            Best column index, or None if no valid moves
        """
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None
        
        best_move = valid_moves[0]
        best_score = float('-inf')
        
        for move in valid_moves:
            # Make the move
            self.game.make_move(move)
            score = self._minimax(self.depth - 1, False, float('-inf'), float('inf'))
            # Undo the move
            self._undo_move(move)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(self, depth: int, maximizing: bool, alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning.
        
        Args:
            depth: Current search depth
            maximizing: True if maximizing player, False if minimizing
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Evaluation score
        """
        if depth == 0 or self.game.game_over:
            return self._evaluate_board()
        
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return 0  # Draw
        
        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                self.game.make_move(move)
                eval_score = self._minimax(depth - 1, False, alpha, beta)
                self._undo_move(move)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                self.game.make_move(move)
                eval_score = self._minimax(depth - 1, True, alpha, beta)
                self._undo_move(move)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def _evaluate_board(self) -> float:
        """Evaluate the current board state.
        
        Returns:
            Evaluation score (positive for AI advantage, negative for opponent)
        """
        if self.game.winner == 1:  # AI wins
            return 1000
        elif self.game.winner == 2:  # Opponent wins
            return -1000
        elif self.game.winner == 0:  # Draw
            return 0
        
        # Heuristic evaluation
        score = 0
        board = self.game.get_board_state()
        
        # Evaluate all possible 4-in-a-row positions
        for row in range(self.game.rows):
            for col in range(self.game.cols):
                if board[row, col] != 0:
                    continue
                    
                # Check horizontal
                score += self._evaluate_position(board, row, col, 0, 1)
                # Check vertical
                score += self._evaluate_position(board, row, col, 1, 0)
                # Check diagonal
                score += self._evaluate_position(board, row, col, 1, 1)
                score += self._evaluate_position(board, row, col, 1, -1)
        
        return score
    
    def _evaluate_position(self, board: np.ndarray, row: int, col: int, dr: int, dc: int) -> float:
        """Evaluate a specific position for potential 4-in-a-row.
        
        Args:
            board: Current board state
            row: Row position
            col: Column position
            dr: Row direction
            dc: Column direction
            
        Returns:
            Evaluation score for this position
        """
        ai_count = 0
        opponent_count = 0
        empty_count = 0
        
        for i in range(4):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < self.game.rows and 0 <= c < self.game.cols:
                if board[r, c] == 1:  # AI piece
                    ai_count += 1
                elif board[r, c] == 2:  # Opponent piece
                    opponent_count += 1
                else:  # Empty
                    empty_count += 1
            else:
                return 0  # Invalid position
        
        if ai_count > 0 and opponent_count > 0:
            return 0  # Blocked position
        
        if ai_count == 0 and opponent_count == 0:
            return 0  # Empty position
        
        # Score based on number of pieces in line
        if ai_count > 0:
            return ai_count ** 2
        else:
            return -(opponent_count ** 2)
    
    def _undo_move(self, col: int):
        """Undo the last move in the specified column.
        
        Args:
            col: Column to undo move in
        """
        for row in range(self.game.rows):
            if self.game.board[row, col] != 0:
                self.game.board[row, col] = 0
                break
        
        # Switch player back
        self.game.current_player = 3 - self.game.current_player
        self.game.game_over = False
        self.game.winner = None
