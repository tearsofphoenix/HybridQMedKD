import pennylane as qml
import torch
import torch.nn as nn
import math


class QuantumLayer(nn.Module):
    """
    Parameterized quantum circuit layer using PennyLane.
    Uses manual batch loop to avoid TorchLayer batch-dimension bugs
    across PennyLane versions (confirmed fix for 0.38+).

    Encodes input via angle encoding (RY), applies variational
    blocks (RY + RZ + CNOT entanglement), returns PauliZ expectation
    values as a (batch, n_qubits) tensor.
    """

    def __init__(self, n_qubits=4, n_q_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Angle encoding: one RY per qubit
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational blocks
            for l in range(n_q_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                # Linear entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Return stacked tensor: shape (n_qubits,)
            return qml.math.stack(
                [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            )

        self._circuit = circuit

        # Trainable weights: (n_q_layers, n_qubits, 2)
        bound = 1.0 / math.sqrt(n_qubits)
        self.weights = nn.Parameter(
            torch.empty(n_q_layers, n_qubits, 2).uniform_(-bound, bound)
        )

    def forward(self, x):
        """
        x: (batch_size, n_qubits)  — values should be in [-pi, pi]
        returns: (batch_size, n_qubits)
        """
        # Manual batch loop: safe across all PennyLane versions
        out = torch.stack([self._circuit(x[i], self.weights) for i in range(x.shape[0])])
        return out  # (batch, n_qubits)

    def get_circuit_depth(self):
        """Approximate circuit depth: encoding + n_q_layers * (rotations + entanglement)."""
        return 1 + self.n_q_layers * 2
