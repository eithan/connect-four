"""
Refactored train_new.py using modular AlphaZero implementation.
"""

import argparse
import os
import torch
import logging
import random
import numpy as np
from connect_four_ai.alphazero import AlphaZeroConfig, config_dict, Connect4, ConnectFour, AlphaZeroTrainer, AlphaZeroAgent, Evaluator, ONNXAlphaZeroNetwork

# for ONNX model
import onnxruntime as ort
import numpy as np

config = AlphaZeroConfig.from_dict(config_dict)

def init_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s')

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_path(mode):
    """Get the correct model file path based on mode."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    
    model_files = {
        'weights': os.path.join(models_dir, 'alphazero-network-weights.pth'),
        'torchscript': os.path.join(models_dir, 'alphazero-network-model-ts.pt'),
        'onnx': os.path.join(models_dir, 'alphazero-network-model.onnx'),
        'checkpoint': os.path.join(models_dir, 'alphazero-checkpoint.pt')
    }
    return model_files.get(mode, model_files['weights'])

def train(alphazero, resume: bool = False):
    import json
    from datetime import datetime
    import time

    print(f"Starting training... ({config.training_epochs} training epochs x {config.games_per_epoch} games per epoch)")
    
    evaluator = Evaluator(alphazero)

    # Evaluate pre training
    evaluator.evaluate()

    # Main training/eval loop
    start_time = datetime.now()
    wall_start = time.time()
    alphazero.train(config.training_epochs)
    wall_elapsed = time.time() - wall_start
    end_time = datetime.now()
    evaluator.evaluate()
    # for _ in range(config.training_epochs):
    #     alphazero.train(1)
    #     evaluator.evaluate()

    # Create models directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # this is our pytorch model containing the weights
    models_dir = os.path.dirname(get_model_path('weights'))
    os.makedirs(models_dir, exist_ok=True)
    weights_path = get_model_path('weights')
    torch.save(alphazero.network.state_dict(), weights_path)
    
    # Calculate total training time and accumulate config history
    checkpoint_path = get_model_path('checkpoint')
    total_elapsed_time = wall_elapsed
    config_history = [config]
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'elapsed_time' in checkpoint:
                total_elapsed_time = checkpoint['elapsed_time'] + wall_elapsed
            if 'config_history' in checkpoint:
                config_history = checkpoint['config_history'] + [config]
        except Exception as e:
            print(f"Warning: Could not load previous checkpoint for time accumulation: {e}")
    
    # save checkpoint (model + optimizer + counters + config history + total elapsed time)
    alphazero.save_checkpoint(checkpoint_path, total_elapsed_time, config_history)
    print(f"Training completed! Model saved at {weights_path}.")

    # for Cloud deployment we have to use TorchScript and export to CPU
    convert_weights_to_ts_model(alphazero)

    # for client-side deployment we  use ONNX
    convert_weights_to_onnx_model(alphazero)

    # results summary
    write_results_summary(start_time, end_time, wall_elapsed, weights_path, models_dir)

def write_results_summary(start_time, end_time, wall_elapsed, weights_path, models_dir):
    import json
    results_path = os.path.join(models_dir, 'results_summary.txt')
    
    # Load checkpoint to get config history and accumulated training time
    checkpoint_path = get_model_path('checkpoint')
    total_training_time = wall_elapsed
    training_configs = [config]
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'elapsed_time' in checkpoint:
                total_training_time = checkpoint['elapsed_time']  # This now contains the total accumulated time
            if 'config_history' in checkpoint:
                training_configs = checkpoint['config_history']
            elif 'config' in checkpoint:
                training_configs.append(checkpoint['config'])
        except Exception as e:
            print(f"Warning: Could not load checkpoint for config history: {e}")
    
    # Calculate total games trained across all configs
    total_games_trained = sum(cfg.training_epochs * cfg.games_per_epoch for cfg in training_configs)
    
    # Format total training time nicely
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    summary = {
        "total_training_time_seconds": round(total_training_time, 3),
        "total_training_time_formatted": time_str,
        "total_games_trained": total_games_trained,
        "start_utc": start_time.isoformat() + "Z",
        "end_utc": end_time.isoformat() + "Z",
        "elapsed_seconds": round(wall_elapsed, 3),
        "outputs": {
            "weights_path": weights_path,
            "torchscript_path": get_model_path('torchscript'),
            "onnx_path": get_model_path('onnx'),
        },
        "training_configs": [
            {
                "n_filters": cfg.n_filters,
                "n_res_blocks": cfg.n_res_blocks,
                "exploration_constant": cfg.exploration_constant,
                "temperature": cfg.temperature,
                "dirichlet_alpha": cfg.dirichlet_alpha,
                "dirichlet_eps": cfg.dirichlet_eps,
                "learning_rate": cfg.learning_rate,
                "training_epochs": cfg.training_epochs,
                "games_per_epoch": cfg.games_per_epoch,
                "minibatch_size": cfg.minibatch_size,
                "n_minibatches": cfg.n_minibatches,
                "mcts_start_search_iter": cfg.mcts_start_search_iter,
                "mcts_max_search_iter": cfg.mcts_max_search_iter,
                "mcts_search_increment": cfg.mcts_search_increment,
                "seed": cfg.seed,
                "device": str(cfg.device),
            } for cfg in training_configs
        ],
    }

    with open(results_path, 'w') as f:
        # Human-friendly text with a JSON blob at the end for parsing
        f.write("AlphaZero Training Summary\n")
        f.write("==========================\n")
        f.write(f"Total Training Time (includes prior training runs, if any): {time_str}\n")
        f.write(f"Total Games Trained: {total_games_trained:,}\n\n")
        f.write(f"Start (UTC): {summary['start_utc']}\n")
        f.write(f"End   (UTC): {summary['end_utc']}\n")
        f.write(f"Elapsed Time: {summary['elapsed_seconds']}s ({summary['elapsed_seconds'] / 60:.2f}m) ({summary['elapsed_seconds'] / 3600:.2f}h)\n\n")
        f.write("Outputs:\n")
        f.write(f"- Weights: {summary['outputs']['weights_path']}\n")
        f.write(f"- TorchScript: {summary['outputs']['torchscript_path']}\n")
        f.write(f"- ONNX: {summary['outputs']['onnx_path']}\n\n")
        f.write("Training Configs:\n\n")
        for i, cfg in enumerate(summary['training_configs']):
            f.write(f"Config {i+1}:\n")
            for k, v in cfg.items():
                f.write(f"  - {k}: {v}\n")
            f.write("\n")
        f.write("JSON:\n")
        f.write(json.dumps(summary, indent=2))
    print(f"Wrote results summary to {results_path}")    
    
def convert_weights_to_ts_model(alphazero, reload=False):
    print("Converting weights to TorchScript model...")
    if reload:
        weights_path = get_model_path('weights')
        pre_trained_weights = torch.load(weights_path, map_location=config.device)
        alphazero.network.load_state_dict(pre_trained_weights)

    # for Cloud deployment we have to use CPU!
    alphazero.network.to("cpu")
    alphazero.network.eval()
    ts_path = get_model_path('torchscript')
    torch.jit.script(alphazero.network).save(ts_path)
    print(f"Export completed! Torchscript Model saved at {ts_path}.")

def convert_weights_to_onnx_model(alphazero, reload=False):
    # for dynamo mode we can ignore this warning because we don't use torchvision capabilities
    import warnings
    warnings.filterwarnings("ignore", message=".*torchvision is not installed.*")

    print("Converting weights to ONNX model...")
    if reload:
        weights_path = get_model_path('weights')
        pre_trained_weights = torch.load(weights_path, map_location=config.device)
        alphazero.network.load_state_dict(pre_trained_weights)
        alphazero.network.to("cpu")
        alphazero.network.eval()
    
    onnx_path = get_model_path('onnx')
    dummy_input = torch.zeros(1, 3, 6, 7, dtype=torch.float32).contiguous()  # match conv input
    # Ensure eval/CPU before export
    alphazero.network.to("cpu")
    alphazero.network.eval()
    # Export with static shapes (batch=1), using a newer opset and constant folding to help lower convolutions
    torch.onnx.export(
        alphazero.network,
        dummy_input,
        onnx_path,
        dynamo=True,
        export_params=True,
        do_constant_folding=True,
        opset_version=18,
        input_names=["input"],
        output_names=["value", "policy_logits"],
    )
    print(f"Export completed! ONNX Model saved at {onnx_path}.")
    print(f"DON'T FORGET TO COPY THE ONNX FILES TO THE WEB APP PUBLIC DIRECTORY!")

def load_weights_to_alphazero(alphazero):
    file_path = get_model_path('weights')
    pre_trained_weights = torch.load(file_path, map_location=config.device)
    print(f"Loaded weights from {file_path}")
    alphazero.network.load_state_dict(pre_trained_weights)

def load_ts_model_to_alphazero(alphazero):
    file_path = get_model_path('torchscript')
    ts_model = torch.jit.load(file_path, map_location=config.device)
    print(f"Loaded TorchScript model from {file_path}")
    alphazero.network = ts_model

def load_model_by_mode(alphazero, mode):
    """Load the appropriate model based on mode."""
    if mode == 'weights':
        load_weights_to_alphazero(alphazero)
    elif mode == 'torchscript':
        load_ts_model_to_alphazero(alphazero)
    elif mode == 'onnx':
        onnx_path = get_model_path(mode)
        alphazero.network = ONNXAlphaZeroNetwork(onnx_path)
        alphazero.mcts.network = alphazero.network
        print(f"Loaded ONNX model from {onnx_path}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

def play_human_vs_alphazero(alphazero, alphazero_first=False):
    agent = AlphaZeroAgent(alphazero)

    # Set to 1 for AlphaZero to play first
    turn = 1 if alphazero_first else 0

    # Reset the game
    state = Connect4().reset()
    done = False

    # Play loop
    while not done:
        print("Current Board:")
        print(state)

        if turn == 0:
            print("Human to move.")
            action = int(input("Enter a move:"))
        else:
            print("AlphaZero is thinking...")
            action = agent.select_action(state, 200)

        next_state, reward, done = Connect4().step(state, action)

        print("Board After Move:")
        print(next_state)

        if done == True:
            print("Game over")
        else:
            state = -next_state
            turn = 1 - turn

def load_model(path):
    return ort.InferenceSession(path)

def predict(session, state):
    x = np.array([state], dtype=np.float32)
    output = session.run(None, {"input": x})
    move = int(output[0].argmax())
    return move

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AlphaZero Connect Four Training and Playing')
    
    parser.add_argument('--play_mode', choices=['weights', 'torchscript', 'onnx'], 
                       default='weights', 
                       help='Model format to use (default: weights)')
    
    parser.add_argument('--action', choices=['train', 'play', 'export'], 
                       required=True,
                       help='Action to perform: train or play or export')

    parser.add_argument('--alphazero_first', choices=['true', 'false'],
                       default='false',
                       help='Play with AlphaZero first (default: false)')

    parser.add_argument('--export_mode', choices=['torchscript', 'onnx'], 
                       default='onnx', 
                       help='Model format to use (default: onnx)')

    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    parser.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    parser.add_argument('--games_per_epoch', type=int, default=None, help='Override number of games per epoch')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint or weights')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    init_logging(args.verbose)
    if args.seed is not None:
        seed_everything(args.seed)
    
    game = Connect4()
    # Apply CLI overrides to config
    if args.epochs is not None:
        config.training_epochs = int(args.epochs)
    if args.games_per_epoch is not None:
        config.games_per_epoch = int(args.games_per_epoch)

    alphazero = AlphaZeroTrainer(game, config)
    
    if args.action == 'train':
        # Optionally resume
        if args.resume:
            ckpt = get_model_path('checkpoint')
            weights = get_model_path('weights')
            if os.path.exists(ckpt):
                print(f"Resuming from checkpoint: {ckpt}")
                try:
                    alphazero.load_checkpoint(ckpt)
                except Exception as e:
                    print(f"Failed to load checkpoint ({e}), trying weights: {weights}")
                    if os.path.exists(weights):
                        load_weights_to_alphazero(alphazero)
            elif os.path.exists(weights):
                print(f"Resuming from weights: {weights}")
                load_weights_to_alphazero(alphazero)
            else:
                print("No checkpoint or weights found; starting fresh.")
        # Train the model
        train(alphazero, resume=args.resume)
    elif args.action == 'export':
        if args.export_mode == 'torchscript':
            convert_weights_to_ts_model(alphazero)
        elif args.export_mode == 'onnx':
            convert_weights_to_onnx_model(alphazero)
        else:
            print(f"Unknown export mode: {args.export_mode}")
            print("Available options:")
            print("1. Export to TorchScript with: python train.py --action export --export_mode torchscript")
            print("2. Export to ONNX with: python train.py --action export --export_mode onnx")
            exit(1)
    elif args.action == 'play':
        model_path = get_model_path(args.play_mode)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            print("Available options:")
            print("1. Train first with: python train.py --action train --play_mode <play_mode>")
            print("2. Use a different play_mode if the model exists in another format")
            exit(1)
        
        print(f"Loading model from {model_path} for playing...")
        load_model_by_mode(alphazero, args.play_mode)
        
        print(f"Starting game with {args.play_mode} model...")
        play_human_vs_alphazero(alphazero, args.alphazero_first == 'true')