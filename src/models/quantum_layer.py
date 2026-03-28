import math
import pennylane as qml
import torch
import torch.nn as nn


class QuantumLayer(nn.Module):
    """
    Hybrid quantum layer for Python 3.11 + PennyLane.
    - AngleEmbedding + BasicEntanglerLayers
    - parameter-shift diff_method
    - Output explicitly cast to float32 to match PyTorch layers
    """

    def __init__(self, n_qubits=4, n_q_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def _circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = _circuit

        bound = 1.0 / math.sqrt(n_qubits)
        self.weights = nn.Parameter(
            torch.empty(n_q_layers, n_qubits).uniform_(-bound, bound)
        )

    def forward(self, x):
        """
        x: (batch_size, n_qubits), float32, tanh-bounded
        returns: (batch_size, n_qubits), float32
        """
        # Run circuit per sample, cast each output to float32
        outs = [
            torch.stack(self._circuit(x[i], self.weights)).float()
            for i in range(x.shape[0])
        ]
        return torch.stack(outs)  # (batch, n_qubits), float32

    def get_circuit_depth(self):
        return 1 + self.n_q_layers
