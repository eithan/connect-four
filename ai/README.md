# Connect Four AI

A Python library for implementing AI strategies for the Connect Four game.

## Features

- Complete Connect Four game implementation
- Multiple AI strategies (Random, Minimax with alpha-beta pruning)
- Command-line interface for playing against AI
- Extensible architecture for adding new AI strategies

## Installation

This project uses Poetry for dependency management. To set up the development environment:

```bash
cd ai
poetry install
```

## Usage

### Command Line Interface

Play against the AI:

```bash
poetry run connect-four-ai
```

Available options:
- `--ai-type`: Choose between "random" or "minimax" AI
- `--depth`: Set search depth for minimax AI (default: 4)
- `--rows`: Number of rows in the board (default: 6)
- `--cols`: Number of columns in the board (default: 7)

### Python API

```python
from connect_four_ai import ConnectFourGame, ConnectFourAI, RandomAI, MinimaxAI

# Create a game
game = ConnectFourGame(rows=6, cols=7)

# Create an AI
ai = MinimaxAI(game, depth=4)

# Make moves
game.make_move(3)  # Human move
ai_move = ai.get_move()  # AI move
if ai_move is not None:
    game.make_move(ai_move)
```

## Development

Run tests:
```bash
poetry run pytest
```

Format code:
```bash
poetry run black src/
```

Lint code:
```bash
poetry run flake8 src/
```

Type checking:
```bash
poetry run mypy src/
```
