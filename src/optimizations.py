# This file consists of low-level optimizations independent of other modules.
import ncpol2sdpa as ncpol
import numpy as np
import numpy.typing as npt
import utilities
from typing import Tuple


def GuessingProbability(
    probabilities: npt.NDArray[np.float64], x_star: int = 0, level: int = 1
) -> Tuple[float, npt.NDArray[np.float64]]:
    """Create and solve npa hierarchy in order to find the maximum achievable guessing probability for x_star state.

    Args:
        probabilities: Array for which index [a,b,x,y] corresponds to probability p(ab|xy).
        x_star: Special input for which the guessing probability of eavesdropper will be maximized.
        level: Level of the npa hierarchy used for solving the problem.

    Returns:
        The optimal guessing probability and dual variables indexed the same as probability, corresponding to bell inequality coefficients.
    """
    assert (
        len(probabilities.shape) == 4
        and all(d > 0 for d in probabilities.shape)
        and x_star < probabilities.shape[2]
    )
    P = ncpol.Probability([2, 2], [2, 2])
    behaviour_constraints = [
        P([0], [0], "A") - np.sum(probabilities[0, :, 0, 0]),
        P([0], [1], "A") - np.sum(probabilities[1, :, 0, 0]),
        P([0], [0], "B") - np.sum(probabilities[:, 0, 0, 0]),
        P([0], [1], "B") - np.sum(probabilities[:, 1, 0, 0]),
        P([0, 0], [0, 0]) - probabilities[0, 0, 0, 0],
        P([0, 0], [0, 1]) - probabilities[0, 0, 0, 1],
        P([0, 0], [1, 0]) - probabilities[0, 0, 1, 0],
        P([0, 0], [1, 1]) - probabilities[0, 0, 1, 1],
    ]
    behaviour_constraints.append("-0[0,0]+1.0")
    sdp = ncpol.SdpRelaxation(P.get_all_operators(), normalized=False, verbose=0)
    sdp.get_relaxation(
        level,
        objective=-P([0], [x_star], "A"),
        momentequalities=behaviour_constraints,
        substitutions=P.substitutions,
    )
    sdp.solve()
    return -sdp.primal, np.zeros([1])


def alice_bob_entrophy(
    density_matrix: npt.NDArray[np.complex128],
    alice_pvm_star_input: npt.NDArray[np.complex128],
    bob_pvm_star_input: npt.NDArray[np.complex128],
) -> float:
    probability_array = np.zeros(
        (alice_pvm_star_input.shape[0], bob_pvm_star_input.shape[0])
    )
    for i, alice_pvm in enumerate(alice_pvm_star_input):
        for j, bob_pvm in enumerate(bob_pvm_star_input):
            val = np.trace(np.kron(alice_pvm, bob_pvm) @ density_matrix)
            probability_array[i, j] = np.real(val)
    # Fix rounding error
    probability_array = np.clip(probability_array, 0.0, 1.0)
    probability_array = probability_array / np.sum(probability_array)
    probability_mask = probability_array > 0
    entrophy_AB = -np.sum(
        probability_array[probability_mask]
        * np.log(probability_array[probability_mask])
        / np.log(utilities.STATES)
    )
    bob_probability = np.sum(probability_array, axis=0)
    # Fix rounding error
    bob_probability = np.clip(bob_probability, 0.0, 1.0)
    bob_probability = bob_probability / np.sum(bob_probability)
    bob_mask = bob_probability > 0
    entrophy_B = -np.sum(
        bob_probability[bob_mask]
        * np.log(bob_probability[bob_mask])
        / np.log(utilities.STATES)
    )
    return float(entrophy_AB - entrophy_B)


def max_quadratic_form_vector(
    matrix: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Return the unit vector that yields the maximum quadratic form value for the given matrix."""
    # Check if the input is hermitian matrix
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    assert np.allclose(
        matrix, matrix.conj().T, rtol=utilities.RTOL, atol=utilities.ATOL
    )
    # Better stability
    matrix = (matrix + matrix.conj().T) / 2
    eigenvalue, eigenvectors = np.linalg.eigh(matrix)
    vector = eigenvectors[np.argmax(eigenvalue)]
    return np.astype(vector, np.complex128)


def maximum_bell_expression(
    input_angles: npt.NDArray[np.float64],
    probabilities_coefficients: npt.NDArray[np.float64],
) -> Tuple[float, npt.NDArray[np.complex128]]:
    """Finds optimal state for the bell expression with given angles

    Args:
        input_angles: Composition of angles for Alice (up to 2*STATE index) and bob.
        probabilities_coefficients: Coefficients for Bell state where position at index [a,b,x,y] corresponds to coefficient of p(ab|xy).

    Return:
        The maximum possible violation achievable and the matrix of bell expression.
    """
    assert input_angles.size == 2 * 2 * utilities.STATES
    alice_pvm_stacked, bob_pvm_stacked = (
        utilities.pvm_from_angle_array(
            input_angles[: int(input_angles.size / 2)],
            utilities.INPUTS,
            utilities.STATES,
        ),
        utilities.pvm_from_angle_array(
            input_angles[int(input_angles.size / 2) :],
            utilities.INPUTS,
            utilities.STATES,
        ),
    )
    bell_matrix = utilities.bell_expression(
        probabilities_coefficients, alice_pvm_stacked, bob_pvm_stacked
    )
    # For stability
    eigenvalues, eigenvectors = np.linalg.eigh((bell_matrix + bell_matrix.conj().T) / 2)
    state = eigenvectors[:, np.argmax(eigenvalues)]
    violation = state.conj().T @ bell_matrix @ state
    assert violation.size == 1
    return float(violation), np.outer(state, state.conj())
