import numpy as np

# Define Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Kronecker products for 2-qubit system
pauli_labels = [a+b for a in ['I', 'X', 'Y', 'Z'] for b in ['I', 'X', 'Y', 'Z']]
pauli_matrices = [np.kron(a, b) for a in [I, X, Y, Z] for b in [I, X, Y, Z]]

def decompose_unitary(U):
    """Decompose a 2-qubit unitary matrix into Pauli terms."""
    coeffs = []
    for P in pauli_matrices:
        # Compute the coefficient for each Pauli matrix
        coeff = np.trace(U @ P.conj().T) / 4
        coeffs.append(coeff)
    
    decomposition = {label: coeff for label, coeff in zip(pauli_labels, coeffs)}
    return decomposition

def reconstruct_from_decomposition(decomposition):
    """Reconstruct the unitary matrix from the Pauli term decomposition."""
    return sum(decomposition[label] * P for label, P in zip(pauli_labels, pauli_matrices))
