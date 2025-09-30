"""
Command line interface for Connect Four AI.
"""

import argparse
import sys
from .game import ConnectFourGame
from .ai import RandomAI, MinimaxAI


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Connect Four AI")
    parser.add_argument("--ai-type", choices=["random", "minimax"], default="minimax",
                       help="Type of AI to use")
    parser.add_argument("--depth", type=int, default=4,
                       help="Search depth for minimax AI")
    parser.add_argument("--rows", type=int, default=6,
                       help="Number of rows in the board")
    parser.add_argument("--cols", type=int, default=7,
                       help="Number of columns in the board")
    
    args = parser.parse_args()
    
    # Create game
    game = ConnectFourGame(rows=args.rows, cols=args.cols)
    
    # Create AI
    if args.ai_type == "random":
        ai = RandomAI(game)
    else:
        ai = MinimaxAI(game, depth=args.depth)
    
    print("Connect Four AI")
    print("=" * 20)
    print("Commands:")
    print("  <column> - Make a move in the specified column (0-6)")
    print("  quit    - Exit the game")
    print("  reset   - Reset the game")
    print()
    
    while True:
        # Display board
        print_board(game.get_board_state())
        print(f"Current player: {game.current_player}")
        
        if game.game_over:
            if game.winner == 0:
                print("It's a draw!")
            else:
                print(f"Player {game.winner} wins!")
            break
        
        if game.current_player == 1:  # AI's turn
            move = ai.get_move()
            if move is not None:
                print(f"AI plays column {move}")
                game.make_move(move)
            else:
                print("No valid moves available!")
                break
        else:  # Human's turn
            try:
                command = input("Your move (column 0-6, 'quit', or 'reset'): ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'reset':
                    game.reset()
                    continue
                else:
                    col = int(command)
                    if game.make_move(col):
                        print(f"You played column {col}")
                    else:
                        print("Invalid move! Try again.")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input! Try again.")
                continue


def print_board(board):
    """Print the game board.
    
    Args:
        board: 2D numpy array representing the board
    """
    print("\n  0 1 2 3 4 5 6")
    for row in board:
        print("  " + " ".join(str(cell) if cell != 0 else "." for cell in row))
    print()


if __name__ == "__main__":
    main()
