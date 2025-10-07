To train:
poetry run python train.py --action train --verbose --resume --epochs 10 --games_per_epoch 100
This will resume from its last checkpoint, if any. Recommended training size >= 100 epochs x 200 games/epoch = 20000 games.
Training summary is recorded in models/results_summary.txt.
Models are placed inside the models folder. There will be one regular pytorch model, one Torchscript model, and one ONNX model.
There will be a checkpoint model used only for training purposes, in case you want to train the same model iteratively.

To test:
poetry run python train.py --action play --play_mode weights
poetry run python train.py --action play --play_mode torchscript
poetry run python train.py --action play --play_mode onnx

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