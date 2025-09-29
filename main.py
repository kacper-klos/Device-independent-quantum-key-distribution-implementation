import numpy as np
import numpy.typing as npt

# Constants valid for 2x2 states for Alice and Bob
STATES = 2
PROJECTION_SHAPE = (STATES, STATES)
STATE_SHAPE = (4, 4)
# How close should the proper value be
RTOL = 1e-8
ATOL = 1e-10


def pvm_check(pvm_stacked: npt.NDArray[np.complex128]) -> bool:
    """Check if passed set of pvm is valid

    Args:
        pvm_stacked: Stacked pvm, at every index there is a pvm.

    Returns:
        True if the set is the pvm.
    """
    # Check if it is valid set of pvm
    status = pvm_stacked.ndim == 3 and np.allclose(
        np.sum(pvm_stacked, axis=0), np.eye(STATES), rtol=RTOL, atol=ATOL
    )
    # Check each of pvm
    i = 0
    while status and i < pvm_stacked.shape[0]:
        pvm = pvm_stacked[i]
        status = (
            pvm.shape == PROJECTION_SHAPE
            and np.allclose(pvm, pvm.conj().T, rtol=RTOL, atol=ATOL)
            and np.allclose(pvm, pvm @ pvm, rtol=RTOL, atol=ATOL)
        )
        i += 1
    return status


def is_hermitian_psd(matrix: npt.NDArray[np.complex128]) -> bool:
    """Check if matrix is hermitian and positively semi-definite"""
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.allclose(matrix, matrix.conj().T, rtol=RTOL, atol=ATOL)
    ):
        return False
    # Better stability for checking if psd
    eigenvalues = np.linalg.eigvalsh(
        (matrix + matrix.conj().T) / 2
    )
    return float(eigenvalues.min()) >= -ATOL


def density_matix_check(rho: np.ndarray) -> bool:
    """Check if passed quantum state is valid"""
    trace = np.abs(np.trace(rho))
    return (
        rho.shape == STATE_SHAPE
        and abs(trace.imag) <= ATOL
        and np.isclose(trace, 1.0, rtol=RTOL, atol=ATOL)
        and is_hermitian_psd(rho)
    )


def probability_vector(
    rho: npt.NDArray[np.complex128],
    pvm_alice_set: npt.NDArray[np.complex128],
    pvm_bob_set: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """Create a probability vector of Alice and Bob measurements

    Args:
        rho: Density matrix of a stat.
        pvm_alice_set: Set of Alice pvm.
        pvm_bob_set: Set of Bob pvm.

    Returns:
        Vector with probabilities of achieving the state.
    """
    # Check validity of input
    assert pvm_check(pvm_alice_set) and pvm_check(pvm_bob_set)
    assert density_matix_check(rho)

    probabilities = np.zeros(pvm_alice_set.shape[0] * pvm_bob_set.shape[0])
    i = 0
    for pvm_alice in pvm_alice_set:
        for pvm_bob in pvm_bob_set:
            probabilities[i] = np.trace(rho @ np.kron(pvm_alice, pvm_bob))
            i += 1
    return probabilities
