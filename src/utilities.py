# In this file are defined a constants and functions that check whether operations are valid quantum operations and compute basic objects.
import numpy as np
import numpy.typing as npt
from itertools import product

# Constants valid for 2x2 states for Alice and Bob
INPUTS = 2
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
        print(f"Dim is wrong: {pvm_stacked.ndim}")
        return False
    for pvm_set in pvm_stacked:
        if not np.allclose(
            np.sum(pvm_set, axis=0), np.eye(STATES), rtol=RTOL, atol=ATOL
        ):
            print(f"Not to identity: {pvm_set}")
            return False
        # Check each of pvm in set
        for pvm in pvm_set:
            if not pvm.shape == PROJECTION_SHAPE:
                print(f"Not proper shape of a {pvm.shape}")
                return False
            if not np.allclose(pvm, pvm.conj().T, rtol=RTOL, atol=ATOL):
                print(f"Not hermitian {pvm}")
                return False
            if not np.allclose(pvm, pvm @ pvm, rtol=RTOL, atol=ATOL):
                print(f"Not projection {pvm} != {pvm @ pvm}")
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


def density_matrix_check(rho: npt.NDArray[np.complex128]) -> bool:
    """Check if passed quantum state is valid"""
    trace = np.abs(np.trace(rho))
    return (
        rho.shape == STATE_SHAPE
        and abs(trace.imag) <= ATOL
        and np.isclose(trace, 1.0, rtol=RTOL, atol=ATOL)
        and is_hermitian_psd(rho)
    )


def parametrized_state(theta: float, psi: float) -> npt.NDArray[np.complex128]:
    """Return a state parametrized by θ and ψ."""
    zero_ket = np.array([1.0, 0.0], dtype=np.float64)
    one_ket = np.array([0.0, 1.0], dtype=np.float64)
    return np.astype(
        np.cos(theta) * zero_ket + np.exp(psi * 1j) * np.sin(theta) * one_ket,
        np.complex128,
    )


def parametrized_projections(theta: float, psi: float) -> npt.NDArray[np.complex128]:
    """Return a pair of PVMs parametrized by θ and ψ."""
    state = parametrized_state(theta, psi)
    pvm = np.astype(np.outer(state, state.conj()), np.complex128)
    return np.stack([pvm, np.eye(STATES) - pvm])


def pvm_from_angle_array(
    angles: npt.NDArray[np.float64], inputs: int, outputs: int
) -> npt.NDArray[np.complex128]:
    """Creates a set of pvm from the list of angles.

    Args:
        angles: List where set of two indexes will be use to create a projection.
        inputs: Number of input settings per party.
        outputs: Number of possible outputs per measurement.

    Returns:
        Array where at the index [x,a] is matrix corresponding to pvm with input x and output a.
    """
    assert angles.size % (inputs * outputs) == 0
    pvm_stacked = np.stack(
        [
            parametrized_projections(angles[outputs * i], angles[outputs * i + 1])
            for i in range(inputs)
        ],
    )
    return pvm_stacked


def reduced_visibility_matrix(
    rho: npt.NDArray[np.complex128], visibility: float
) -> npt.NDArray[np.complex128]:
    """Apply reduced visibility to a density matrix."""
    assert visibility >= 0 and visibility <= 1
    assert density_matrix_check(rho)
    return np.astype(
        visibility * rho + (1 - visibility) / rho.shape[0] * np.eye(rho.shape[0]),
        np.complex128,
    )


def bell_expression(
    probabilities_coefficients: npt.NDArray[np.float64],
    alice_pvm_stack: npt.NDArray[np.complex128],
    bob_pvm_stack: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """Create a matrix representing a Bell expression.

    Args:
        probabilities_coefficients: Coefficient at index [a,b,x,y] corresponds to the projection p(ab|xy) in bell expression.
        alice_pvm_stacked: Set of pvms from Alice.
        bob_pvm_stacked: Set of pvms from Bob.

    Returns:
        Matrix which after being multiplied with the density matrix and taken trace gives violation of Bell expression.
    """
    bell_matrix = np.zeros(
        (
            alice_pvm_stack.shape[-2] + bob_pvm_stack.shape[-2],
            alice_pvm_stack.shape[-1] + bob_pvm_stack.shape[-1],
        ),
        dtype=np.complex128,
    )
    shape = probabilities_coefficients.shape
    for a, b, x, y in product(
        range(shape[0]), range(shape[1]), range(shape[2]), range(shape[3])
    ):
        bell_matrix += probabilities_coefficients[a, b, x, y] * (
            np.kron(alice_pvm_stack[x, a], bob_pvm_stack[y, b])
        )
    return np.astype(bell_matrix, np.float64)


def probability_array(
    rho: npt.NDArray[np.complex128],
    pvm_alice_stacked: npt.NDArray[np.complex128],
    pvm_bob_stacked: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    """Create a probability vector of Alice and Bob measurements

    Args:
        rho: Density matrix of a state.
        pvm_alice_set: Set of Alice pvm.
        pvm_bob_set: Set of Bob pvm.

    Returns:
        Array of joint probabilities where index [a,b,x,y] corresponds to p(ab|xy).
    """
    # Check validity of input
    assert pvm_check(pvm_alice_stacked) and pvm_check(pvm_bob_stacked)
    assert density_matrix_check(rho)

    probabilities = np.zeros(
        (
            pvm_alice_stacked.shape[1],
            pvm_bob_stacked.shape[1],
            pvm_alice_stacked.shape[0],
            pvm_bob_stacked.shape[0],
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
