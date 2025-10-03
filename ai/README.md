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



----

Notes:

Connect 4 in Javascript -- contains all game logic; only server call needed is for AI moves
AI in Python --
  * Using AlphaZero code from https://www.kaggle.com/code/auxeno/alphazero-connect-4-rl/notebook
  * Trained the model locally with 50 epochs and 100 games/epoch (doesn't seem to play too well yet)
  * Trained the model locally with 100 epochs and 200 games/epoch (plays better but not perfectly)
    ** Took 23 hours on a Mac Pro M4 Max with 16 Core CPU, 40 Core GPU, and 64 GB of unified memory
  * Converted to TorchScript for faster loading (converts into a static graph that loads faster, doesn't need full Python model)
    ** This would be used on Google Cloud if I build a Python server for the AI player
  * Converted to Onnx model and implemented some javascript to load it
    ** TODO: Confirm whether the javascript is correct
    ** TODO: make sure to set a cache policy on the public model file
    ** TODO: See whether this is a sustainable method for browser-only playing (load-time, etc)
    ** TODO: Fully implement this in the game as an AI player
    ** TODO: unit tests!

  * TODO: Clean up the python code, remove all the cruft from Cursor (cli, game, play, etc)
  * TODO: stream-line the AI Python library to just build the Onnx file and copy it appropriately/etc if using Onnx in game