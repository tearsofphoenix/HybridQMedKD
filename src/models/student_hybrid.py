import torch
import torch.nn as nn
from .quantum_layer import QuantumLayer


class StudentHybrid(nn.Module):
    """
    Hybrid quantum-classical student.
    quantum_position: 'front' | 'middle' | 'tail'
    """

    def __init__(self, input_dim, n_qubits=4, n_q_layers=1, quantum_position="middle"):
        super().__init__()
        self.quantum_position = quantum_position
        self.n_qubits = n_qubits

        self.front_proj = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU()
        )
        self.q_proj = nn.Sequential(
            nn.Linear(16, n_qubits),
            nn.Tanh()
        )
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_q_layers=n_q_layers)
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.tail_pre = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_qubits),
            nn.Tanh()
        )

    def forward(self, x):
        if self.quantum_position == "front":
            q_in = torch.tanh(x[:, :self.n_qubits])
        elif self.quantum_position == "middle":
            h = self.front_proj(x)
            q_in = self.q_proj(h)
        elif self.quantum_position == "tail":
            h = self.front_proj(x)
            q_in = self.tail_pre(h)
        else:
            raise ValueError(f"Unknown quantum_position: {self.quantum_position}")

        q_out = self.quantum(q_in)
        return self.head(q_out)
