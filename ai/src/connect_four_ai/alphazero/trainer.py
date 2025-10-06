"""
AlphaZero training implementation.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import logging

from .network import ResNet
from .mcts import MCTS

import time

logger = logging.getLogger(__name__)

class AlphaZeroTrainer:
    """AlphaZero training implementation with self-play and neural network updates."""
    
    def __init__(self, game, config, verbose=True):
        self.network = ResNet(game, config).to(config.device)
        self.mcts = MCTS(self.network, game, config)
        self.game = game
        self.config = config

        # Losses and optimizer
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, weight_decay=0.0001)

        # Pre-allocate memory on GPU
        state_shape = game.encode_state(game.reset()).shape
        self.max_memory = config.minibatch_size * config.n_minibatches
        self.state_memory = torch.zeros(self.max_memory, *state_shape).to(config.device)
        self.value_memory = torch.zeros(self.max_memory, 1).to(config.device)
        self.policy_memory = torch.zeros(self.max_memory, game.cols).to(config.device)
        self.current_memory_index = 0
        self.memory_full = False

        # MCTS search iterations
        self.search_iterations = config.mcts_start_search_iter
        
        # Logging
        self.verbose = verbose
        self.total_games = 0

    def train(self, training_epochs):
        """Train the AlphaZero agent for a specified number of training epochs."""
        # For each training epoch
        time_start = time.time()
        for epoch in range(int(training_epochs)):
            if self.verbose:
                logger.info(f"[{time.time() - time_start:.2f}s] Training epoch {epoch}/{training_epochs} ({self.config.games_per_epoch} games per epoch)")

            # Play specified number of games
            for _ in range(int(self.config.games_per_epoch)):
                self.self_play()
            
            # At the end of each epoch, increase the number of MCTS search iterations
            self.search_iterations = min(self.config.mcts_max_search_iter, self.search_iterations + self.config.mcts_search_increment)

            elapsed_time = time.time() - time_start
            time_per_epoch = elapsed_time / (epoch + 1)
            estimated_time_remaining = time_per_epoch * (training_epochs - epoch - 1)
            if self.verbose:
                logger.info(f"Estimated time remaining: {estimated_time_remaining:.2f}s ({estimated_time_remaining / 60:.2f}m) ({estimated_time_remaining / 3600:.2f}h)")


    def self_play(self):
        """Perform one episode of self-play."""
        state = self.game.reset()
        done = False
        while not done:
            # Search for a move
            action, root = self.mcts.search(state, self.search_iterations, use_root_dirichlet_noise=True)

            # Value target is the value of the MCTS root node
            value = root.get_value()

            # Visit counts used to compute policy target
            visits = np.zeros(self.game.cols)
            for child_action, child in root.children.items():
                visits[child_action] = child.n_visits
            # Softmax so distribution sums to 1
            visits /= np.sum(visits)

            # Append state + value & policy targets to memory
            self.append_to_memory(state, value, visits)

            # If memory is full, perform a learning step
            if self.memory_full:
                self.learn()

            # Perform action in game
            state, _, done = self.game.step(state, action)

            # Flip the board
            state = -state

        # Increment total games played
        self.total_games += 1

        # Logging if verbose
        if self.verbose and self.total_games % 100 == 0:
            logger.info(f"\nTotal Games: {self.total_games}, Items in Memory: {self.current_memory_index}, Search Iterations: {self.search_iterations}\n")

    def append_to_memory(self, state, value, visits):
        """
        Append state and MCTS results to memory buffers.
        Args:
            state (array-like): Current game state.
            value (float): MCTS value for the game state.
            visits (array-like): MCTS visit counts for available moves.
        """
        # Calculate the encoded states
        encoded_state = np.array(self.game.encode_state(state))
        encoded_state_augmented = np.array(self.game.encode_state(state[:, ::-1]))

        # Stack states and visits
        states_stack = np.stack((encoded_state, encoded_state_augmented), axis=0)
        visits_stack = np.stack((visits, visits[::-1]), axis=0)

        # Convert the stacks to tensors
        state_tensor = torch.tensor(states_stack, dtype=torch.float).to(self.config.device)
        visits_tensor = torch.tensor(visits_stack, dtype=torch.float).to(self.config.device)
        value_tensor = torch.tensor(np.array([value, value]), dtype=torch.float).to(self.config.device).unsqueeze(1)

        # Store in pre-allocated GPU memory
        self.state_memory[self.current_memory_index:self.current_memory_index + 2] = state_tensor
        self.value_memory[self.current_memory_index:self.current_memory_index + 2] = value_tensor
        self.policy_memory[self.current_memory_index:self.current_memory_index + 2] = visits_tensor

        # Increment index, handle overflow
        self.current_memory_index = (self.current_memory_index + 2) % self.max_memory

        # Set memory filled flag to True if memory is full
        if (self.current_memory_index == 0) or (self.current_memory_index == 1):
            self.memory_full = True

    def learn(self):
        """Update the neural network by extracting minibatches from memory and performing one step of optimization for each one."""
        self.network.train()

        # Create a randomly shuffled list of batch indices
        batch_indices = np.arange(self.max_memory)
        np.random.shuffle(batch_indices)

        for batch_index in range(self.config.n_minibatches):
            # Get minibatch indices
            start = batch_index * self.config.minibatch_size
            end = start + self.config.minibatch_size
            mb_indices = batch_indices[start:end]

            # Slice memory tensors
            mb_states = self.state_memory[mb_indices]
            mb_value_targets = self.value_memory[mb_indices]
            mb_policy_targets = self.policy_memory[mb_indices]

            # Network predictions
            value_preds, policy_logits = self.network(mb_states)

            # Loss calculation
            # Policy: match MCTS visit distribution using KL divergence
            policy_log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = nn.KLDivLoss(reduction="batchmean")(policy_log_probs, mb_policy_targets)
            # Value: MSE between predicted value and root value target
            value_loss = self.loss_mse(value_preds.view(-1), mb_value_targets.view(-1))
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory_full = False
        self.network.eval()
