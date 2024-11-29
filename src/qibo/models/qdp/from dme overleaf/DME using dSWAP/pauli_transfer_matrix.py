import numpy as np

# Define Pauli matrices for single qubit
pauli_matrices = [
    np.eye(2, dtype=complex),  # Identity
    np.array([[0, 1], [1, 0]], dtype=complex),  # X
    np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
    np.array([[1, 0], [0, -1]], dtype=complex),  # Z
]

def hilbert_schmidt_inner(A, B):
    """Compute the Hilbert-Schmidt inner product of two matrices A and B."""
    return np.trace(np.dot(np.conjugate(A.T), B)).real

def n_qubit_paulis(n):
    """Generate the n-qubit Pauli matrices using Kronecker products."""
    if n == 1:
        return pauli_matrices
    else:
        smaller_paulis = n_qubit_paulis(n - 1)
        return [np.kron(p1, p2) for p1 in pauli_matrices for p2 in smaller_paulis]

def pauli_transfer_matrix(U):
    """
    Compute the Pauli Transfer Matrix (PTM) for a given unitary U.

    The PTM represents the action of a quantum channel in the Pauli basis. 
    For each pair of Pauli operators $P_i$ and $P_j$, we apply the 
    unitary transformation $U P_j U^\dagger$ to $P_j$, and compute 
    the Hilbert-Schmidt inner product with $P_i$. The result is 
    normalized by $1/2^n$, where $n$ is the number of qubits.
    
    Args:
        U (np.ndarray): The unitary operator as a NumPy array.

    Returns:
        np.ndarray: The Pauli Transfer Matrix.
    """

    # Determine the number of qubits from the dimension of U
    dim = U.shape[0]
    num_qubits = int(np.log2(dim))
    
    if dim not in [2, 4]:
        raise ValueError("Only 1-qubit (2x2) or 2-qubit (4x4) unitary matrices are supported.")
    
    # Generate Pauli matrices for n qubits
    pauli_ops = n_qubit_paulis(num_qubits)
    n_paulis = len(pauli_ops)

    # Initialize the PTM matrix
    PTM = np.zeros((n_paulis, n_paulis), dtype=complex)
    
    # Loop through all pairs of Pauli operators
    for i, p_i in enumerate(pauli_ops):
        for j, p_j in enumerate(pauli_ops):
            # Transform p_j by U: U * p_j * U†
            p_j_transformed = np.dot(np.dot(U, p_j), np.conjugate(U.T))
            
            # Compute the Hilbert-Schmidt inner product
            PTM[i, j] = hilbert_schmidt_inner(p_i, p_j_transformed)
    
    # Normalize by 1/2^n
    PTM /= 2**num_qubits
    
    return PTM.real
