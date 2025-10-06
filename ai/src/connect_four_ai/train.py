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
        'onnx': os.path.join(models_dir, 'alphazero-network-model.onnx')
    }
    return model_files.get(mode, model_files['weights'])

def train(alphazero):
    print(f"Starting training...")
        
    evaluator = Evaluator(alphazero)

    # Evaluate pre training
    evaluator.evaluate()

    # Main training/eval loop
    alphazero.train(config.training_epochs)
    evaluator.evaluate()
    # for _ in range(config.training_epochs):
    #     alphazero.train(1)
    #     evaluator.evaluate()

    # Create models directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # this is our pytorch model containing the weights
    weights_path = get_model_path('weights')
    torch.save(alphazero.network.state_dict(), weights_path)
    print(f"Training completed! Model saved at {weights_path}.")

    # for Cloud deployment we have to use TorchScript and export to CPU
    convert_weights_to_ts_model(alphazero)

    # for client-side deployment we  use ONNX
    convert_weights_to_onnx_model(alphazero)
    
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
    dummy_input = torch.zeros(1, 3, 6, 7, dtype=torch.float32)  # match conv input
    torch.onnx.export(
        alphazero.network,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["value", "policy_logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "value": {0: "batch"},
            "policy_logits": {0: "batch"},
        }
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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    init_logging(args.verbose)
    if args.seed is not None:
        seed_everything(args.seed)
    
    game = Connect4()
    alphazero = AlphaZeroTrainer(game, config)
    
    if args.action == 'train':
        # Train the model
        train(alphazero)
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