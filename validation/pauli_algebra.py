"""Small NumPy helpers for constructing multi-qubit states and operators."""

from __future__ import annotations

from functools import reduce

import numpy as np


I_GATE = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)

ZERO = np.array([1.0, 0.0], dtype=complex)
ONE = np.array([0.0, 1.0], dtype=complex)


def kron_all(factors):
    """Return the Kronecker product of all factors in order."""
    factors = list(factors)
    if not factors:
        return np.array([[1.0]], dtype=complex)
    return reduce(np.kron, factors)


def tensor_power(operator, n_qubits: int):
    """Return ``operator`` tensored with itself ``n_qubits`` times."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive.")
    return kron_all([operator] * n_qubits)


def pauli_on_qubit(pauli, n_qubits: int, qubit: int):
    """Embed a single-qubit Pauli operator at the requested qubit index."""
    if not 0 <= qubit < n_qubits:
        raise ValueError("qubit index is out of range.")
    factors = [I_GATE] * n_qubits
    factors[qubit] = np.asarray(pauli, dtype=complex)
    return kron_all(factors)


def basis_bits_to_ket(bits):
    """Convert a sequence of computational-basis bits into a state vector."""
    bits = np.asarray(bits, dtype=int).ravel()
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must contain only 0 and 1.")
    return kron_all(ZERO if bit == 0 else ONE for bit in bits)
