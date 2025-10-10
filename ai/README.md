### AlphaZero training (short guide)

#### 1) Environment setup

```bash
poetry install
```

#### 2) Train (resumable)

```bash
poetry run python -m connect_four_ai.train --action train --verbose --resume --epochs 10 --games_per_epoch 100
```

- Outputs land in `src/connect_four_ai/models/`:
  - `alphazero-network-weights.pth` (PyTorch weights)
  - `alphazero-network-model-ts.pt` (TorchScript)
  - `alphazero-network-model.onnx` (+ `.onnx.data` if emitted)
  - `alphazero-checkpoint.pt` (resume + total time/config history)
  - `results_summary.txt` (aggregated metrics and paths)

Recommended for better play: ≥ 100 epochs × ≥ 200 games/epoch.

#### 3) Test Python-side inference

```bash
poetry run python -m connect_four_ai.train --action play --play_mode weights
poetry run python -m connect_four_ai.train --action play --play_mode torchscript
poetry run python -m connect_four_ai.train --action play --play_mode onnx
```

#### 4) Use the model in the web app

Copy the ONNX artifacts into the web app `public` folder, then run the site.

```bash
cp src/connect_four_ai/models/alphazero-network-model.onnx ../web/public/
cp src/connect_four_ai/models/alphazero-network-model.onnx.data ../web/public/  # if present
cd ../web && npm run dev
```

#### Credits:
- AlphaZero adapted from an amazing write-up and implementation by Kaggle user auxeno at https://www.kaggle.com/code/auxeno/alphazero-connect-4-rl/notebook

#### Notes:
- The shipped browser model (in `web/public`) was trained for 67 hours on a 16‑core M4 Max MacBook Pro with 64 GB RAM and a 40‑core GPU.
- ONNX is used for cross‑platform, serverless inference via `onnxruntime‑web`.
- TorchScript is useful for Python server deployments.
