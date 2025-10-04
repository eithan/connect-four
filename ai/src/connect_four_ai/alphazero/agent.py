"""
AlphaZero agent for playing games.
"""

import numpy as np
import torch
import torch.nn.functional as F


class AlphaZeroAgent:
    """AlphaZero agent for playing Connect Four games."""
    
    def __init__(self, alphazero_trainer):
        self.alphazero_trainer = alphazero_trainer
        self.alphazero_trainer.network.eval()
        
        # Remove noise from move calculations
        self.alphazero_trainer.config.dirichlet_eps = 0

    def select_action(self, state, search_iterations=200):
        """Select an action for the given state."""
        #print(f"AlphaZeroAgent.select_action() called with search_iterations={search_iterations}")
        #print(f"Network type: {type(self.alphazero_trainer.network)}")
        
        state_tensor = torch.tensor(self.alphazero_trainer.game.encode_state(state), dtype=torch.float).to(self.alphazero_trainer.config.device)
        #print(f"State tensor shape: {state_tensor.shape}")
        
        # Get action without using search
        if search_iterations == 0:
            #print("Using direct network prediction (no MCTS)")
            with torch.no_grad():
                _, logits = self.alphazero_trainer.network(state_tensor.unsqueeze(0))

            # Get action probs and mask for valid actions
            action_probs = F.softmax(logits.view(-1), dim=0).cpu().numpy()
            valid_actions = self.alphazero_trainer.game.get_valid_actions(state)
            valid_action_probs = action_probs[valid_actions]
            best_action = valid_actions[np.argmax(valid_action_probs)]
            #print(f"Selected action: {best_action}")
            return best_action
        # Else use MCTS 
        else:
            #print(f"Using MCTS with {search_iterations} iterations")
            action, _ = self.alphazero_trainer.mcts.search(state, search_iterations)
            #print(f"MCTS selected action: {action}")
            return action
