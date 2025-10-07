"""
Evaluation utilities for AlphaZero training.
"""

import random
import numpy as np
import torch
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class Evaluator:
    """Class to evaluate the policy network's performance on simple moves."""
    
    def __init__(self, alphazero_trainer, num_examples=500, verbose=True):
        self.network = alphazero_trainer.network
        self.game = alphazero_trainer.game
        self.config = alphazero_trainer.config
        self.accuracies = []
        self.num_examples = num_examples
        self.verbose = verbose

        # Generate and prepare example states and actions for evaluation
        self.generate_examples()

    def select_action(self, state) -> int:
        """Select an action based on the given state, will choose a winning or blocking moves."""
        valid_actions = self.game.get_valid_actions(state)
        
        # Check for a winning move
        for action in valid_actions:
            next_state, reward, _ = self.game.step(state, action)
            if reward == 1:
                return action

        # Check for a blocking move
        flipped_state = -state
        for action in valid_actions:
            next_state, reward, _ = self.game.step(flipped_state, action)
            if reward == 1:
                return action

        # Default to random action if no winning or blocking move
        return int(random.choice(list(valid_actions)))

    def generate_examples(self) -> None:
        """Generate and prepare example states and actions for evaluation."""
        if self.verbose:
            logger.info(f"Generating {2 * self.num_examples} examples for evaluation...")
            
        winning_examples = self.generate_examples_for_condition('win')
        blocking_examples = self.generate_examples_for_condition('block')

        # Prepare states and actions for evaluation
        winning_example_states, winning_example_actions = zip(*winning_examples)
        blocking_example_states, blocking_example_actions = zip(*blocking_examples)

        target_states = np.concatenate([winning_example_states, blocking_example_states], axis=0)
        target_actions = np.concatenate([winning_example_actions, blocking_example_actions], axis=0)

        encoded_states = [self.game.encode_state(state) for state in target_states]
        self.X_target = torch.tensor(np.stack(encoded_states, axis=0), dtype=torch.float).to(self.config.device)
        self.y_target = torch.tensor(target_actions, dtype=torch.long).to(self.config.device)

    def generate_examples_for_condition(self, condition: str):
        """Generate examples based on either 'win' or 'block' conditions."""
        examples = []
        while len(examples) < self.num_examples:
            state = self.game.reset()
            while True:
                action = self.select_action(state)
                next_state, reward, done = self.game.step(state, action, to_play=1)
                
                if condition == 'win' and reward == 1:
                    examples.append((state, action))
                    break
                
                if done:
                    break
                
                state = next_state

                # Flipping the board for opponent's perspective
                action = self.select_action(-state)
                next_state, reward, done = self.game.step(state, action, to_play=-1)
                
                if condition == 'block' and reward == -1:
                    examples.append((-state, action))
                    break
                
                if done:
                    break
                
                state = next_state
        return examples

    def evaluate(self) -> None:
        """Evaluate the policy network's accuracy and append it to self.accuracies."""
        with torch.no_grad():
            self.network.eval()
            _, logits = self.network(self.X_target)
            pred_actions = logits.argmax(dim=1)
            accuracy = (pred_actions == self.y_target).float().mean().item()
        
        self.accuracies.append(accuracy)
        if self.verbose:
            logger.info(f"Initial Evaluation Accuracy: {100 * accuracy:.1f}%")
