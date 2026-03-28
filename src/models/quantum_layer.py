import math
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class QuantumLayer(nn.Module):
    """
    Minimal hybrid quantum layer for Python 3.11 + PennyLane.
    - Uses interface='torch' with diff_method='parameter-shift'
    - Single-sample qnode called in a manual batch loop
    - Weights managed as nn.Parameter for full torch autograd support
    """

    def __init__(self, n_qubits=4, n_q_layers=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def _circuit(inputs, weights):
            # Angle encoding
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            # Variational layers
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = _circuit

        # Weights shape: (n_q_layers, n_qubits)
        bound = 1.0 / math.sqrt(n_qubits)
        self.weights = nn.Parameter(
            torch.empty(n_q_layers, n_qubits).uniform_(-bound, bound)
        )

    def forward(self, x):
        """
        x: (batch_size, n_qubits), tanh-bounded values
        returns: (batch_size, n_qubits)
        """
        out = torch.stack(
            [torch.stack(self._circuit(x[i], self.weights))
             for i in range(x.shape[0])]
        )
        return out  # (batch, n_qubits)

    def get_circuit_depth(self):
        return 1 + self.n_q_layers
