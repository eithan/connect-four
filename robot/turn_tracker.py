"""
Turn Tracker - Tracks Connect Four game state across board observations.

Compares consecutive board states to determine whose turn it is,
which column was played, and whether the game is over.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

ROWS = 6
COLS = 7


@dataclass
class GameState:
    board: np.ndarray
    current_player: int       # 1=red, 2=yellow (who plays NEXT)
    move_history: List[int]
    game_over: bool = False
    winner: Optional[int] = None
    winning_cells: Optional[List[Tuple[int, int]]] = None


class TurnTracker:
    def __init__(self, robot_player: int = 2):
        """robot_player: 1=red or 2=yellow (default yellow, goes second)."""
        self.robot_player = robot_player
        self.human_player = 3 - robot_player
        self.reset()

    def reset(self):
        self.state = GameState(board=np.zeros((ROWS, COLS), dtype=np.int8),
                               current_player=1, move_history=[])

    def update(self, detected_board):
        """Update state from a newly detected board. Returns result dict."""
        result = self.analyze_transition(detected_board)

        if self.state.game_over:
            return result

        if not result['changed']:
            return result

        row = result['move_row']
        col = result['move_col']
        new_value = result['move_player']

        self.state.board = detected_board.copy()
        self.state.move_history.append(col)

        win_cells = self._check_win(self.state.board, row, col, new_value)
        if win_cells:
            self.state.game_over = True
            self.state.winner = new_value
            self.state.winning_cells = win_cells
            result['game_over'] = True
            result['winner'] = new_value
            result['winning_cells'] = win_cells  # needed by game_loop win animation
        elif np.sum(self.state.board == 0) == 0:
            self.state.game_over = True
            self.state.winner = 0
            result['game_over'] = True
            result['winner'] = 0
        else:
            self.state.current_player = 3 - self.state.current_player

        result['is_robot_turn'] = (self.state.current_player == self.robot_player
                                   and not self.state.game_over)
        return result

    def analyze_transition(self, detected_board):
        """Validate a detected transition without mutating tracker state."""
        result = {'changed': False, 'move_col': None, 'move_player': None,
                  'move_row': None,
                  'is_robot_turn': False, 'game_over': self.state.game_over,
                  'winner': self.state.winner, 'error': None}

        if self.state.game_over:
            result['error'] = "Game is already over"
            return result

        diff = detected_board - self.state.board
        new_pieces = np.argwhere(diff != 0)

        if len(new_pieces) == 0:
            result['is_robot_turn'] = (self.state.current_player == self.robot_player)
            return result

        if len(new_pieces) > 1:
            result['error'] = f"Multiple changes detected ({len(new_pieces)} cells)"
            return result

        row, col = new_pieces[0]
        new_value = detected_board[row, col]

        if new_value != self.state.current_player:
            result['error'] = (f"Wrong color: expected player {self.state.current_player}, "
                              f"got {new_value}")
            return result

        expected_row = self._lowest_empty_row(self.state.board, col)
        if expected_row is None:
            result['error'] = f"Column {col} is full"
            return result
        if row != expected_row:
            result['error'] = f"Gravity violation at ({row},{col}), expected row {expected_row}"
            return result

        result['changed'] = True
        result['move_row'] = row
        result['move_col'] = col
        result['move_player'] = new_value
        return result

    def set_board(self, board):
        """Manually set board state (e.g. starting mid-game from an image)."""
        self.state.board = board.copy()
        red = int(np.sum(board == 1))
        yellow = int(np.sum(board == 2))
        self.state.current_player = 1 if red == yellow else 2

        for row in range(ROWS):
            for col in range(COLS):
                val = board[row, col]
                if val != 0:
                    win = self._check_win(board, row, col, val)
                    if win:
                        self.state.game_over = True
                        self.state.winner = val
                        self.state.winning_cells = win
                        return
        if np.sum(board == 0) == 0:
            self.state.game_over = True
            self.state.winner = 0

    @staticmethod
    def _lowest_empty_row(board, col):
        for row in range(ROWS - 1, -1, -1):
            if board[row, col] == 0:
                return row
        return None

    @staticmethod
    def _check_win(board, row, col, player):
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            cells = [(row, col)]
            for sign in [1, -1]:
                for i in range(1, 4):
                    r, c = row + dr * i * sign, col + dc * i * sign
                    if 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                        cells.append((r, c))
                    else:
                        break
            if len(cells) >= 4:
                return sorted(cells)
        return None

    @staticmethod
    def get_valid_moves(board):
        return [col for col in range(COLS) if board[0, col] == 0]
