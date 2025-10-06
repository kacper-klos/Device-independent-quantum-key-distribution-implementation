import numpy as np
import numpy.typing as npt
from itertools import product

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
    if pvm_stacked.ndim != 4:
        return False
    for pvm_set in pvm_stacked:
        if not np.allclose(
            np.sum(pvm_set, axis=0), np.eye(STATES), rtol=RTOL, atol=ATOL
        ):
            return False
        # Check each of pvm in set
        for pvm in pvm_set:
            if not (
                pvm.shape == PROJECTION_SHAPE
                and np.allclose(pvm, pvm.conj().T, rtol=RTOL, atol=ATOL)
                and np.allclose(pvm, pvm @ pvm, rtol=RTOL, atol=ATOL)
            ):
                return False
    return True


def is_hermitian_psd(matrix: npt.NDArray[np.complex128]) -> bool:
    """Check if matrix is hermitian and positively semi-definite"""
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.allclose(matrix, matrix.conj().T, rtol=RTOL, atol=ATOL)
    ):
        return False
    # Better stability for checking if psd
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2)
    return float(eigenvalues.min()) >= -ATOL


def density_matix_check(rho: npt.NDArray[np.complex128]) -> bool:
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
    pvm_alice_stacked: npt.NDArray[np.complex128],
    pvm_bob_stacked: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """Create a probability vector of Alice and Bob measurements

    Args:
        rho: Density matrix of a stat.
        pvm_alice_set: Set of Alice pvm.
        pvm_bob_set: Set of Bob pvm.

    Returns:
        Vector with probabilities of achieving the state where index a,b,x,y corresponds to p(ab|xy).
    """
    # Check validity of input
    assert pvm_check(pvm_alice_stacked) and pvm_check(pvm_bob_stacked)
    assert density_matix_check(rho)

    probabilities = np.zeros(
        (
            pvm_alice_stacked.shape[0],
            pvm_bob_stacked.shape[0],
            pvm_alice_stacked.shape[1],
            pvm_bob_stacked.shape[1],
        )
    )
    for x, alice_pvm_set in enumerate(pvm_alice_stacked):
        for y, bob_pvm_set in enumerate(pvm_bob_stacked):
            for a, alice_pvm in enumerate(alice_pvm_set):
                for b, bob_pvm in enumerate(bob_pvm_set):
                    probabilities[a, b, x, y] = np.trace(
                        rho @ np.kron(alice_pvm, bob_pvm)
                    )

    return probabilities


def parametrixed_state(theta: float, psi: float) -> npt.NDArray[np.float64]:
    """Return state parametrized by theta"""
    zero_ket = np.array([1.0, 0.0], dtype=np.float64)
    one_ket = np.array([0.0, 1.0], dtype=np.float64)
    return np.astype(
        np.cos(theta) * zero_ket + np.exp(psi * 1j) * np.sin(theta) * one_ket,
        np.float64,
    )


def parametrized_projections(theta: float, psi: float) -> npt.NDArray[np.float64]:
    """Returns pair of pvm parametrized by theta"""
    state = parametrixed_state(theta, psi)
    pvm = np.astype(np.outer(state, state.conj()), np.float64)
    return np.stack([pvm, np.eye(STATES) - pvm])


def max_quadratic_form_vector(
    matrix: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """For the given matrix returns the unit vector that gives the highest possible value"""
    # Check if the input is hermitian matrix
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert np.allclose(matrix, matrix.conj().T, rtol=RTOL, atol=ATOL)
    # Better stability
    matrix = (matrix + matrix.conj().T) / 2
    eigenvalue, eigenvectors = np.linalg.eigh(matrix)
    vector = eigenvectors[np.argmax(eigenvalue)]
    return np.astype(vector, np.complex128)


def reduced_visibility_matrix(
    rho: npt.NDArray[np.complex128], visibility: float
) -> npt.NDArray[np.complex128]:
    """Set the visibility of a density matrix"""
    assert visibility >= 0 and visibility <= 1 and density_matix_check(rho)
    return visibility * rho + (1 - visibility) * np.eye(rho.shape[0])


def bell_expression(
    probabilities_coefficients: npt.NDArray[np.float64],
    alice_pvm_stack: npt.NDArray[np.float64],
    bob_pvm_stack: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Creates a matrix representing a bell expression."""
    bell_matrix = np.zeros(
        (
            alice_pvm_stack.shape[-2] + bob_pvm_stack.shape[-2],
            alice_pvm_stack.shape[-1] + bob_pvm_stack.shape[-1],
        )
    )
    for a, b, x, y in product(*probabilities_coefficients.shape):
        bell_matrix += probabilities_coefficients[a, b, x, y] * (
            np.kron(alice_pvm_stack[x, a], bob_pvm_stack[x, b])
        )
    return np.astype(bell_matrix, np.float64)


def maximum_bell_expression(
    input_angles: npt.NDArray[np.float64],
    probabilities_coefficients: npt.NDArray[np.float64],
) -> float:
    """Finds optimal state for the bell expression with given angles

    Args:
        input_angles: Composition of angles for Alice (up to 2*STATE index) and bob
    """
    assert input_angles.size == 2 * 2 * STATES
    alice_pvm_stacked = np.stack(
        [
            parametrized_projections(input_angles[2 * i], input_angles[2 * i + 1])
            for i in range(STATES)
        ],
    )
    bob_pvm_stacked = np.stack(
        [
            parametrized_projections(
                input_angles[2 * STATES + 2 * i], input_angles[2 * STATES + 2 * i + 1]
            )
            for i in range(STATES)
        ],
    )
    bell_matrix = bell_expression(
        probabilities_coefficients, alice_pvm_stacked, bob_pvm_stacked
    )
    # For stability
    eigenvalues, eigenvectors = np.linalg.eigh((bell_matrix + bell_matrix.conj().T) / 2)
    state = eigenvectors[:, np.argmax(eigenvalues)]
    violation = state.conj().T @ bell_matrix @ state
    assert violation.size == 1
    return float(violation[0])
