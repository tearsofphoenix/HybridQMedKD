import pennylane as qml
import torch
import torch.nn as nn


class QuantumLayer(nn.Module):
    """
    Parameterized quantum circuit layer using PennyLane.
    Encodes input via angle encoding (RY), applies variational
    blocks (RY + RZ + CNOT entanglement), and returns PauliZ
    expectation values.
    """

    def __init__(self, n_qubits=4, n_q_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_q_layers = n_q_layers

        dev = qml.device("default.qubit", wires=n_qubits)
        weight_shapes = {"weights": (n_q_layers, n_qubits, 2)}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Angle encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for l in range(n_q_layers):
                for i in range(n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # x: (batch, n_qubits)
        return self.qlayer(x)

    def get_circuit_depth(self):
        """Approximate circuit depth for reporting."""
        # 1 encoding layer + n_q_layers * (rotation + entanglement)
        return 1 + self.n_q_layers * 2
