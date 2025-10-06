"""
Monte Carlo Tree Search implementation for AlphaZero.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


class MCTS:
    """Monte Carlo Tree Search implementation."""
    
    def __init__(self, network, game, config):
        """Initialize Monte Carlo Tree Search with a given neural network, game instance, and configuration."""
        self.network = network
        self.game = game
        self.config = config

    def search(self, state, total_iterations, temperature=None, use_root_dirichlet_noise: bool = True):
        """Performs a search for the desired number of iterations, returns an action and the tree root."""
        # Create the root
        root = TreeNode(None, state, 1, self.game, self.config)

        # Expand the root, adding noise to each action
        # Get valid actions
        valid_actions = self.game.get_valid_actions(state)
        state_tensor = torch.tensor(self.game.encode_state(state), dtype=torch.float).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            self.network.eval()
            #print(f"MCTS calling network with state_tensor shape: {state_tensor.shape}")
            #print(f"MCTS network type: {type(self.network)}")
            value, logits = self.network(state_tensor)
            #print(f"MCTS got value shape: {value.shape}, logits shape: {logits.shape}")

        # Compute policy over all actions, then mix Dirichlet noise only on valid actions
        logits_flat = logits.reshape(-1)
        policy_probs = F.softmax(logits_flat, dim=0).detach().cpu().numpy()
        priors = np.zeros(self.game.cols, dtype=np.float32)
        priors[:] = policy_probs

        if use_root_dirichlet_noise and len(valid_actions) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_actions))
            priors_valid = priors[valid_actions]
            priors[valid_actions] = (1 - self.config.dirichlet_eps) * priors_valid + self.config.dirichlet_eps * noise

        # Normalize priors over valid actions only
        denom = np.sum(priors[valid_actions]) if len(valid_actions) > 0 else 1.0
        if denom == 0:
            # uniform over valid actions as fallback
            priors[valid_actions] = 1.0 / max(1, len(valid_actions))
        else:
            priors[valid_actions] /= denom

        # Create a child for each possible valid action
        for action in valid_actions:
            child_state = -self.game.get_next_state(state, action, to_play=1)
            root.children[action] = TreeNode(root, child_state, -1, self.game, self.config)
            root.children[action].prob = float(priors[action])

        # Since we're not backpropagating, manually increase visits
        root.n_visits = 1
        # Set value as neural network prediction also as it will slightly improve the accuracy of the value target later
        root.total_score = value.item()

        # Begin search
        for _ in range(total_iterations):
            current_node = root

            # Phase 1: Selection
            # While not currently on a leaf node, select a new node using PUCT score
            while not current_node.is_leaf():
                current_node = current_node.select_child()

            # Phase 2: Expansion
            # When a leaf node is reached and it's not terminal; expand it
            if not current_node.is_terminal():
                current_node.expand()
                # Convert node state to tensor and pass through network
                state_tensor = torch.tensor(self.game.encode_state(current_node.state), dtype=torch.float).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    self.network.eval()
                    value, logits = self.network(state_tensor)
                    value = value.item()

                # Mask invalid actions for this node, then set child priors
                node_valid_actions = self.game.get_valid_actions(current_node.state)
                logits_flat = logits.reshape(-1)
                masked_logits = logits_flat.clone()
                # set invalid actions to -inf so they get zero probability after softmax
                invalid_mask = torch.ones(self.game.cols, dtype=torch.bool, device=masked_logits.device)
                if len(node_valid_actions) > 0:
                    invalid_mask[torch.tensor(node_valid_actions, device=masked_logits.device)] = False
                masked_logits[invalid_mask] = float('-inf')
                node_action_probs = F.softmax(masked_logits, dim=0).detach().cpu().numpy()
                for action in node_valid_actions:
                    if action in current_node.children:
                        current_node.children[action].prob = float(node_action_probs[action])
            # If node is terminal, get the value of it from game instance
            else:
                value = self.game.evaluate(current_node.state)

            # Phase 3: Backpropagation
            # Backpropagate the value of the leaf to the root
            current_node.backpropagate(value)
        
        # Select action with specified temperature
        if temperature == None:
            temperature = self.config.temperature
        return self.select_action(root, temperature), root

    def select_action(self, root, temperature=None):
        """Select an action from the root based on visit counts, adjusted by temperature, 0 temp for greedy."""
        if temperature == None:
            temperature = self.config.temperature
        action_counts = {key: val.n_visits for key, val in root.children.items()}
        if temperature == 0:
            return max(action_counts, key=action_counts.get)
        elif temperature == np.inf:
            return np.random.choice(list(action_counts.keys()))
        else:
            distribution = np.array([*action_counts.values()]) ** (1 / temperature)
            return np.random.choice([*action_counts.keys()], p=distribution/sum(distribution))


class TreeNode:
    """Represents a node in the MCTS, holding the game state and statistics for MCTS to operate."""
    
    def __init__(self, parent, state, to_play, game, config):
        self.parent = parent
        self.state = state
        self.to_play = to_play
        self.config = config
        self.game = game

        self.prob = 0
        self.children = {}
        self.n_visits = 0
        self.total_score = 0

    def expand(self):
        """Create child nodes for all valid actions. If state is terminal, evaluate and set the node's value."""
        # Get valid actions
        valid_actions = self.game.get_valid_actions(self.state)

        # If there are no valid actions, state is terminal, so get value using game instance
        if len(valid_actions) == 0:
            self.total_score = self.game.evaluate(self.state)
            return

        # Create a child for each possible action
        for action in valid_actions:
            # Make move, then flip board to perspective of next player
            child_state = -self.game.get_next_state(self.state, action, to_play=self.to_play)
            self.children[action] = TreeNode(self, child_state, -self.to_play, self.game, self.config)

    def select_child(self):
        """Select the child node with the highest PUCT score."""
        best_puct = -np.inf
        best_child = None
        for child in self.children.values():
            puct = self.calculate_puct(child)
            if puct > best_puct:
                best_puct = puct
                best_child = child
        return best_child

    def calculate_puct(self, child):
        """Calculate the PUCT score for a given child node."""
        # Scale Q(s,a) so it's between 0 and 1 so it's comparable to a probability
        # Using 1 - Q(s,a) because it's from the perspectve of the child – the opposite of the parent
        exploitation_term = 1 - (child.get_value() + 1) / 2
        exploration_term = child.prob * math.sqrt(self.n_visits) / (child.n_visits + 1)
        return exploitation_term + self.config.exploration_constant * exploration_term

    def backpropagate(self, value):
        """Update the current node and its ancestors with the given value."""
        self.total_score += value
        self.n_visits += 1
        if self.parent is not None:
            # Backpropagate the negative value so it switches each level
            self.parent.backpropagate(-value)

    def is_leaf(self):
        """Check if the node is a leaf (no children)."""
        return len(self.children) == 0

    def is_terminal(self):
        """Check if the node represents a terminal state (no valid moves or game over)."""
        return len(self.game.get_valid_actions(self.state)) == 0

    def get_value(self):
        """Calculate the average value of this node."""
        if self.n_visits == 0:
            return 0
        return self.total_score / self.n_visits
    
    def __str__(self):
        """Return a string containing the node's relevant information for debugging purposes."""
        return (f"State:\n{self.state}\nProb: {self.prob}\nTo play: {self.to_play}" +
                f"\nNumber of children: {len(self.children)}\nNumber of visits: {self.n_visits}" +
                f"\nTotal score: {self.total_score}")
