"""
AI Player - Interface to the AlphaZero Connect Four model.

Supports two modes:
1. ONNX inference: Loads your trained model and runs inference
2. Heuristic fallback: Simple evaluation for testing without the model

To use your trained model, place the ONNX files in:
    models/alphazero-network-model.onnx
    models/alphazero-network-model.onnx.data  (if external data)
"""

import numpy as np
import os
from typing import Optional, Tuple, List

ROWS = 6
COLS = 7


class AIPlayer:
    def __init__(self, model_path: Optional[str] = None, use_heuristic: bool = False):
        self.session = None
        self.use_heuristic = use_heuristic
        self.input_name = None

        if not use_heuristic:
            self._try_load_model(model_path)
        if self.session is None and not use_heuristic:
            print("[AIPlayer] No ONNX model found - using heuristic fallback.")
            print("           Place your model at: models/alphazero-network-model.onnx")
            self.use_heuristic = True

    def _try_load_model(self, model_path):
        base = os.path.dirname(os.path.abspath(__file__))
        paths = ([model_path] if model_path else []) + [
            os.path.join(base, "models", "alphazero-network-model.onnx"),
            os.path.join(base, "..", "ai", "src", "connect_four_ai", "models",
                        "alphazero-network-model.onnx"),
        ]
        for path in paths:
            if path and os.path.exists(path):
                try:
                    import onnxruntime as ort
                    self.session = ort.InferenceSession(path)
                    self.input_name = self.session.get_inputs()[0].name
                    outputs = self.session.get_outputs()
                    print(f"[AIPlayer] Loaded ONNX model: {path}")
                    print(f"           Input: {self.input_name} shape={self.session.get_inputs()[0].shape}")
                    for o in outputs:
                        print(f"           Output: {o.name} shape={o.shape}")
                    return
                except Exception as e:
                    print(f"[AIPlayer] Failed to load {path}: {e}")

    def get_move(self, board, player):
        """
        Get AI's chosen move.

        Args:
            board: 6x7 numpy array (0=empty, 1=red, 2=yellow)
            player: which player AI is (1 or 2)

        Returns:
            (column, info_dict)
        """
        valid_moves = [c for c in range(COLS) if board[0, c] == 0]
        if not valid_moves:
            return -1, {'error': 'No valid moves', 'policy': np.zeros(7),
                       'value': 0.0, 'method': 'none', 'valid_moves': []}

        if self.use_heuristic:
            return self._heuristic_move(board, player, valid_moves)
        return self._onnx_move(board, player, valid_moves)

    def _onnx_move(self, board, player, valid_moves):
        opponent = 3 - player
        inp = np.zeros((1, 3, ROWS, COLS), dtype=np.float32)
        inp[0, 0] = (board == player).astype(np.float32)    # channel 0: current player
        inp[0, 1] = (board == 0).astype(np.float32)         # channel 1: empty
        inp[0, 2] = (board == opponent).astype(np.float32)  # channel 2: opponent

        try:
            outputs = self.session.get_outputs()
            output_names = [o.name for o in outputs]
            results = self.session.run(output_names, {self.input_name: inp})

            # Find policy (shape [1,7]) and value (shape [1,1]) by size, not position
            logits = None
            value = 0.0
            for i, out in enumerate(outputs):
                flat = results[i].flatten()
                if flat.size == COLS:
                    logits = flat
                elif flat.size == 1:
                    value = float(flat[0])
            if logits is None:
                raise ValueError("Could not find policy output with 7 values")

            policy = np.exp(logits - np.max(logits))
            mask = np.zeros(COLS, dtype=np.float32)
            mask[valid_moves] = 1.0
            policy *= mask
            s = policy.sum()
            policy = policy / s if s > 0 else mask / mask.sum()
            col = valid_moves[np.argmax(policy[valid_moves])]
            return col, {'policy': policy, 'value': value, 'method': 'onnx',
                        'valid_moves': valid_moves}
        except Exception as e:
            print(f"[AIPlayer] ONNX inference failed: {e}, falling back")
            return self._heuristic_move(board, player, valid_moves)

    def _heuristic_move(self, board, player, valid_moves):
        """Simple heuristic: check wins, blocks, prefer center."""
        opponent = 3 - player
        scores = np.zeros(COLS, dtype=np.float32)

        for col in valid_moves:
            row = self._drop_row(board, col)
            if row is None:
                continue

            test = board.copy()
            test[row, col] = player
            if self._is_win(test, row, col, player):
                scores[col] = 1000.0
                continue

            test[row, col] = opponent
            if self._is_win(test, row, col, opponent):
                scores[col] = 500.0
                continue

            scores[col] = (COLS // 2 - abs(col - COLS // 2)) * 10
            test[row, col] = player
            scores[col] += self._count_alignments(test, row, col, player) * 5

        valid_scores = scores[valid_moves]
        policy = np.zeros(COLS, dtype=np.float32)
        if valid_scores.max() > 0:
            exp_s = np.exp(valid_scores - valid_scores.max())
            policy[valid_moves] = exp_s / exp_s.sum()
        else:
            policy[valid_moves] = 1.0 / len(valid_moves)

        col = valid_moves[np.argmax(scores[valid_moves])]
        return col, {'policy': policy, 'value': 0.0, 'method': 'heuristic',
                    'valid_moves': valid_moves}

    @staticmethod
    def _drop_row(board, col):
        for row in range(ROWS - 1, -1, -1):
            if board[row, col] == 0:
                return row
        return None

    @staticmethod
    def _is_win(board, row, col, player):
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            count = 1
            for sign in [1, -1]:
                for i in range(1, 4):
                    r, c = row + dr * i * sign, col + dc * i * sign
                    if 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                        count += 1
                    else:
                        break
            if count >= 4:
                return True
        return False

    @staticmethod
    def _count_alignments(board, row, col, player):
        count = 0
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            line = 1
            for sign in [1, -1]:
                for i in range(1, 4):
                    r, c = row + dr * i * sign, col + dc * i * sign
                    if 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                        line += 1
                    else:
                        break
            if line >= 2:
                count += line - 1
        return count
