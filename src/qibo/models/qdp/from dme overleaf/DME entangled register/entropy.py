import numpy as np

def von_neumann_entropy(rho, base=np.e):
    r"""
    Compute the von Neumann entropy of a density matrix.

    The entropy is given by the formula:

    .. math::

        S(\rho) = -\sum_j(\lambda_j * \log_{base}(\lambda_j))

    where lambda_j are the eigenvalues of the density matrix rho.

    Args:
        rho (numpy.ndarray): Density matrix.
        base (int): Logarithm base for entropy calculation. Default is 2 (bits).

    Returns:
        float: Von Neumann entropy of the density matrix.
    """
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 0]
    return -np.sum(evals * np.log(evals)) / np.log(base)


def purity(state):
    """Calculate the purity of a density matrix = Tr[\rho^2]"""
    return np.trace(np.dot(state, state)).real
