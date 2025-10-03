import onnxruntime as ort
import numpy as np
import torch

class ONNXAlphaZeroNetwork:
    """ONNX AlphaZero network. For debugging purposes only because we load the model in javascript directly for playing."""
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"ONNX model loaded. Input: {self.input_name}, Outputs: {self.output_names}")

    # Dummy methods to mimic PyTorch API
    def eval(self):
        return self  # does nothing

    def train(self):
        return self  # does nothing

    def to(self, device):
        return self  # does nothing, ONNX doesn't use devices

    # PyTorch-compatible forward method
    def forward(self, state_tensor):
        """Forward pass that returns (value, policy_logits) like PyTorch network."""
        print(f"ONNXAlphaZeroNetwork.forward() called with state shape: {state_tensor.shape if hasattr(state_tensor, 'shape') else type(state_tensor)}")
        
        if isinstance(state_tensor, torch.Tensor):
            state = state_tensor.detach().cpu().numpy()
        else:
            state = np.array(state_tensor, dtype=np.float32)
        
        # Ensure correct shape
        if state.ndim == 3:
            state = np.expand_dims(state, axis=0)
        
        print(f"ONNX input shape: {state.shape}")
        
        inputs = {self.input_name: state.astype(np.float32)}
        outputs = self.session.run(self.output_names, inputs)
        
        print(f"ONNX outputs: {[output.shape for output in outputs]}")
        
        # Assuming outputs are [value, policy_logits]
        if len(outputs) == 2:
            value = torch.tensor(outputs[0], dtype=torch.float32)
            policy_logits = torch.tensor(outputs[1], dtype=torch.float32)
            print(f"Returning value shape: {value.shape}, policy_logits shape: {policy_logits.shape}")
            return value, policy_logits
        else:
            # Fallback if only one output
            output = torch.tensor(outputs[0], dtype=torch.float32)
            # Split into value and policy (assuming first element is value, rest is policy)
            value = output[:, :1]  # First element as value
            policy_logits = output[:, 1:]  # Rest as policy
            print(f"Fallback - value shape: {value.shape}, policy_logits shape: {policy_logits.shape}")
            return value, policy_logits

    # Callable interface (makes it work like PyTorch module)
    def __call__(self, state_tensor):
        return self.forward(state_tensor)

    # Legacy methods for backward compatibility
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Legacy predict method."""
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if state.ndim == 3:
            state = np.expand_dims(state, axis=0)
        inputs = {self.input_name: state.astype(np.float32)}
        outputs = self.session.run(self.output_names, inputs)
        return outputs[0]

    def get_move(self, state: np.ndarray) -> int:
        """Legacy get_move method."""
        _, policy_logits = self.forward(state)
        return int(torch.argmax(policy_logits, dim=1)[0].item())
