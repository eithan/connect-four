"""
Refactored train_new.py using modular AlphaZero implementation.
"""

import torch
from connect_four_ai.alphazero import Config, config_dict, Connect4, AlphaZeroTrainer, AlphaZeroAgent, Evaluator, ONNXAlphaZeroNetwork

# for ONNX model
import onnxruntime as ort
import numpy as np

config = Config(config_dict)

def train(alphazero):
    evaluator = Evaluator(alphazero)

    # Evaluate pre training
    evaluator.evaluate()

    # Main training/eval loop
    alphazero.train(config.training_epochs)
    evaluator.evaluate()
    # for _ in range(config.training_epochs):
    #     alphazero.train(1)
    #     evaluator.evaluate()

    # Save trained weights
    torch.save(alphazero.network.state_dict(), 'alphazero-network-weights.pth')

    # for Cloud deployment we have to use CPU!
    alphazero.network.to("cpu")
    alphazero.network.eval()
    torch.jit.script(alphazero.network).save("alphazero-network-model-ts.pt")
    
def convert_weights_to_ts_model(alphazero):
    file_path = "./alphazero-network-weights.pth"
    pre_trained_weights = torch.load(file_path, map_location=config.device)
    alphazero.network.load_state_dict(pre_trained_weights)

    # for Cloud deployment we have to use CPU!
    alphazero.network.to("cpu")
    alphazero.network.eval()
    torch.jit.script(alphazero.network).save("alphazero-network-model-ts.pt")

def convert_weights_to_onnx_model(alphazero):
    file_path = "./alphazero-network-weights.pth"
    pre_trained_weights = torch.load(file_path, map_location=config.device)
    alphazero.network.load_state_dict(pre_trained_weights)
    alphazero.network.to("cpu")
    alphazero.network.eval()
    dummy_input = torch.zeros(1, 3, 6, 7, dtype=torch.float32)  # match conv input
    torch.onnx.export(
        alphazero.network,
        dummy_input,
        "alphazero-network-model-onnx.onnx",
        #dynamo=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

def load_weights_to_alphazero(alphazero):
    file_path = "./alphazero-network-weights.pth"
    pre_trained_weights = torch.load(file_path, map_location=config.device)
    print(f"Loaded weights from {file_path}")
    alphazero.network.load_state_dict(pre_trained_weights)

def load_ts_model_to_alphazero(alphazero):
    file_path = "./alphazero-network-model-ts.pt"
    ts_model = torch.jit.load(file_path, map_location=config.device)
    print(f"Loaded TorchScript model from {file_path}")
    alphazero.network = ts_model

def play_human_vs_alphazero(alphazero):
    agent = AlphaZeroAgent(alphazero)

    # Set to 1 for AlphaZero to play first
    turn = 1

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

if __name__ == "__main__":
    game = Connect4()
    # Doubled our training last time! Took 23 hours (83000 seconds) (1:11pm --> 12:29pm the next day)
    #config.training_epochs = config.training_epochs * 2
    #config.games_per_epoch = config.games_per_epoch * 2
    alphazero = AlphaZeroTrainer(game, config)

    # --- Training ---
    #train(alphazero)
    #exit()

    # convert_weights_to_ts_model(alphazero)
    # convert_weights_to_onnx_model(alphazero)
    # exit()

    # play_with_weights = True

    # if play_with_weights:
    #     # --- Playing (without TorchScript)---
    #     load_weights_to_alphazero(alphazero)
    # else:
    #     # --- Playing (with TorchScript)---
    #     load_ts_model_to_alphazero(alphazero)
    
    # Update MCTS to use the new ONNX network
    alphazero.network = ONNXAlphaZeroNetwork("./models/alphazero-network-model-onnx.onnx")
    alphazero.mcts.network = alphazero.network

    print(f"Updated MCTS to use network type: {type(alphazero.mcts.network)}")

    play_human_vs_alphazero(alphazero)