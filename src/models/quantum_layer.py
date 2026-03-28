import math
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Pure-numpy PennyLane circuit (no torch interface, no TorchLayer)
# ---------------------------------------------------------------------------

def _make_circuit(n_qubits, n_q_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface=None)  # pure numpy, no autograd conflict
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(float(inputs[i]), wires=i)
        for l in range(n_q_layers):
            for i in range(n_qubits):
                qml.RY(float(weights[l, i, 0]), wires=i)
                qml.RZ(float(weights[l, i, 1]), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ---------------------------------------------------------------------------
# Numerical Jacobian for backward pass (parameter-shift rule)
# ---------------------------------------------------------------------------

def _param_shift_grad(circuit, inputs_np, weights_np, n_qubits, n_q_layers):
    """
    Compute d(output)/d(weights) via parameter-shift rule.
    Returns grad_weights: same shape as weights_np.
    """
    shift = np.pi / 2
    grad = np.zeros_like(weights_np)
    for l in range(n_q_layers):
        for i in range(n_qubits):
            for k in range(2):  # RY, RZ
                w_plus = weights_np.copy()
                w_plus[l, i, k] += shift
                w_minus = weights_np.copy()
                w_minus[l, i, k] -= shift
                out_p = np.array(circuit(inputs_np, w_plus), dtype=np.float32)
                out_m = np.array(circuit(inputs_np, w_minus), dtype=np.float32)
                grad[l, i, k] = 0  # filled below per output
    # We need full Jacobian: (n_qubits_out, n_q_layers, n_qubits, 2)
    J = np.zeros((n_qubits, n_q_layers, n_qubits, 2), dtype=np.float32)
    for l in range(n_q_layers):
        for i in range(n_qubits):
            for k in range(2):
                w_plus = weights_np.copy()
                w_plus[l, i, k] += shift
                w_minus = weights_np.copy()
                w_minus[l, i, k] -= shift
                out_p = np.array(circuit(inputs_np, w_plus), dtype=np.float32)
                out_m = np.array(circuit(inputs_np, w_minus), dtype=np.float32)
                J[:, l, i, k] = (out_p - out_m) / 2.0
    return J


# ---------------------------------------------------------------------------
# Custom torch.autograd.Function: forward = numpy circuit,
# backward = parameter-shift Jacobian
# ---------------------------------------------------------------------------

class _QuantumCircuitFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights, circuit, n_qubits, n_q_layers):
        """
        inputs:  (n_qubits,) torch tensor
        weights: (n_q_layers, n_qubits, 2) torch tensor
        """
        inp_np = inputs.detach().cpu().numpy().astype(np.float64)
        w_np   = weights.detach().cpu().numpy().astype(np.float64)

        out_np = np.array(circuit(inp_np, w_np), dtype=np.float32)
        ctx.save_for_backward(inputs, weights)
        ctx.circuit    = circuit
        ctx.inp_np     = inp_np
        ctx.w_np       = w_np
        ctx.n_qubits   = n_qubits
        ctx.n_q_layers = n_q_layers
        return torch.from_numpy(out_np)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (n_qubits,)
        returns: grad_inputs, grad_weights, None, None, None
        """
        n_qubits   = ctx.n_qubits
        n_q_layers = ctx.n_q_layers
        circuit    = ctx.circuit
        inp_np     = ctx.inp_np
        w_np       = ctx.w_np

        go = grad_output.detach().cpu().numpy()  # (n_qubits,)

        # Jacobian wrt weights: (n_qubits_out, n_q_layers, n_qubits, 2)
        J = _param_shift_grad(circuit, inp_np, w_np, n_qubits, n_q_layers)
        # grad_weights = J^T @ grad_output  -> (n_q_layers, n_qubits, 2)
        grad_w = np.einsum('o,oijk->ijk', go, J).astype(np.float32)

        grad_weights = torch.from_numpy(grad_w)
        grad_inputs  = None  # inputs not differentiable (angle encoding)
        return grad_inputs, grad_weights, None, None, None


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------

class QuantumLayer(nn.Module):
    """
    Hybrid quantum layer: pure-numpy PennyLane circuit with
    parameter-shift backward pass via custom autograd Function.
    Compatible with Python 3.14 + PennyLane 0.38+ + PyTorch 2.x.
    No TorchLayer, no interface conflict.
    """

    def __init__(self, n_qubits=4, n_q_layers=2):
        super().__init__()
        self.n_qubits   = n_qubits
        self.n_q_layers = n_q_layers
        self._circuit   = _make_circuit(n_qubits, n_q_layers)

        bound = 1.0 / math.sqrt(n_qubits)
        self.weights = nn.Parameter(
            torch.empty(n_q_layers, n_qubits, 2).uniform_(-bound, bound)
        )

    def forward(self, x):
        """
        x: (batch_size, n_qubits), values in [-pi, pi]
        returns: (batch_size, n_qubits)
        """
        outs = [
            _QuantumCircuitFunction.apply(
                x[i], self.weights, self._circuit, self.n_qubits, self.n_q_layers
            )
            for i in range(x.shape[0])
        ]
        return torch.stack(outs)  # (batch, n_qubits)

    def get_circuit_depth(self):
        return 1 + self.n_q_layers * 2
