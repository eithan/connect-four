"""
Connect Four game engine implementation.
"""

import numpy as np


class Connect4:
    """Connect 4 game engine, containing methods for game-related tasks."""
    
    def __init__(self):
        self.rows = 6
        self.cols = 7

    def get_next_state(self, state, action, to_play=1):
        """Play an action in a given state and return the resulting board."""
        # Pre-condition checks
        assert self.evaluate(state) == 0
        assert np.sum(abs(state)) != self.rows * self.cols
        assert action in self.get_valid_actions(state)
        
        # Identify next empty row in column
        row = np.where(state[:, action] == 0)[0][-1]
        
        # Apply action
        new_state = state.copy()
        new_state[row, action] = to_play
        return new_state

    def get_valid_actions(self, state):
        """Return a numpy array containing the indices of valid actions."""
        # If game over, no valid moves
        if self.evaluate(state) != 0:
            return np.array([])
        
        # Valid if the top cell of the column is empty
        return np.where(state[0] == 0)[0]

    def evaluate(self, state):
        """Evaluate the current position. Returns 1 for player 1 win, -1 for player 2 and 0 otherwise."""
        rows, cols = self.rows, self.cols

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                window_sum = int(state[r, c:c+4].sum())
                if window_sum == 4:
                    return 1
                if window_sum == -4:
                    return -1

        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                window_sum = int(state[r:r+4, c].sum())
                if window_sum == 4:
                    return 1
                if window_sum == -4:
                    return -1

        # Diagonal (down-right)
        for r in range(rows - 3):
            for c in range(cols - 3):
                window_sum = int(state[r, c] + state[r+1, c+1] + state[r+2, c+2] + state[r+3, c+3])
                if window_sum == 4:
                    return 1
                if window_sum == -4:
                    return -1

        # Anti-diagonal (down-left)
        for r in range(rows - 3):
            for c in range(3, cols):
                window_sum = int(state[r, c] + state[r+1, c-1] + state[r+2, c-2] + state[r+3, c-3])
                if window_sum == 4:
                    return 1
                if window_sum == -4:
                    return -1

        # No winner
        return 0

    def step(self, state, action, to_play=1):
        """Play an action in a given state. Return the next_state, reward and done flag."""
        # Get new state and reward
        next_state = self.get_next_state(state, action, to_play)
        reward = self.evaluate(next_state)
        
        # Check for game termination
        done = True if (reward != 0) or (np.sum(np.abs(next_state)) >= (self.rows * self.cols)) else False
        return next_state, reward, done

    def encode_state(self, state):
        """Convert state to tensor with 3 channels."""
        encoded_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        return encoded_state

    def reset(self):
        """Reset the board."""
        return np.zeros([self.rows, self.cols], dtype=np.int8)


class ConnectFour(Connect4):
    """Preferred name alias for `Connect4` for readability."""
    pass
