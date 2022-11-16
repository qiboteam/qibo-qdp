import numpy as np
import pytest

from qibo import gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import Circuit
from qibo.models.error_mitigation import CDR, sample_training_circuit
from qibo.noise import DepolarizingError, NoiseModel
from qibo.symbols import Z


def get_noise_model(error):
    noise = NoiseModel()
    noise.add(error, gates.CNOT)
    return noise


@pytest.mark.parametrize("nqubits", [3])
@pytest.mark.parametrize(
    "noise",
    [get_noise_model(DepolarizingError(0.1))],
)
@pytest.mark.parametrize("full_output", [False, True])
def test_cdr(nqubits, noise, full_output):
    """Test that CDR reduces the noise."""
    # Define the circuit
    hz = 0.5
    hx = 0.5
    dt = 0.25
    c = Circuit(nqubits, density_matrix=True)
    c.add(gates.RZ(q, theta=-2 * hz * dt - np.pi / 2) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-2 * hx * dt + np.pi) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-np.pi / 2) for q in range(nqubits))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(1, nqubits, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.M(q) for q in range(nqubits))
    # Define the observable
    obs = np.prod([Z(i) for i in range(nqubits)])
    obs = SymbolicHamiltonian(obs)
    # Noise-free expected value
    exact = obs.expectation(c().state())
    # Noisy expected value without mitigation
    noisy = obs.expectation(noise.apply(c)().state())
    # Mitigated expected value
    estimate = CDR(
        circuit=c,
        observable=obs,
        backend=None,
        noise_model=noise,
        nshots=10000,
        full_output=full_output,
    )
    if full_output:
        estimate = estimate[0]
    assert np.abs(exact - estimate) <= np.abs(exact - noisy)


@pytest.mark.parametrize("nqubits", [3])
def test_sample_training_circuit(nqubits):
    # Define the circuit
    hz = -2
    hx = 1
    dt = np.pi / 4
    c = Circuit(nqubits, density_matrix=True)
    c.add(gates.RZ(q, theta=-2 * hz * dt - np.pi / 2) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-2 * hx * dt + np.pi) for q in range(nqubits))
    c.add(gates.RX(q, theta=np.pi / 2) for q in range(nqubits))
    c.add(gates.RZ(q, theta=-np.pi / 2) for q in range(nqubits))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(0, nqubits - 1, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.RZ(q + 1, theta=-2 * dt) for q in range(1, nqubits, 2))
    c.add(gates.CNOT(q, q + 1) for q in range(1, nqubits, 2))
    c.add(gates.M(q) for q in range(nqubits))
    with pytest.raises(ValueError):
        sample_training_circuit(c)
