#!/usr/bin/env python3
"""
Example usage of the Connect Four AI library.
"""

from connect_four_ai import ConnectFourGame, RandomAI, MinimaxAI


def main():
    """Demonstrate the Connect Four AI library."""
    print("Connect Four AI Library Example")
    print("=" * 40)
    
    # Create a game
    game = ConnectFourGame(rows=6, cols=7)
    
    # Create AI players
    random_ai = RandomAI(game)
    minimax_ai = MinimaxAI(game, depth=3)
    
    print("Playing a game between Random AI and Minimax AI...")
    print()
    
    # Play a game
    move_count = 0
    while not game.game_over and move_count < 42:  # Max possible moves
        if game.current_player == 1:
            move = random_ai.get_move()
            player_name = "Random AI"
        else:
            move = minimax_ai.get_move()
            player_name = "Minimax AI"
        
        if move is None:
            print("No valid moves available!")
            break
        
        print(f"{player_name} plays column {move}")
        game.make_move(move)
        move_count += 1
        
        # Print board state
        print_board(game.get_board_state())
        print()
    
    # Game over
    if game.winner == 0:
        print("It's a draw!")
    else:
        winner_name = "Random AI" if game.winner == 1 else "Minimax AI"
        print(f"{winner_name} wins!")


def print_board(board):
    """Print the game board.
    
    Args:
        board: 2D numpy array representing the board
    """
    print("  0 1 2 3 4 5 6")
    for row in board:
        print("  " + " ".join(str(cell) if cell != 0 else "." for cell in row))


if __name__ == "__main__":
    main()
