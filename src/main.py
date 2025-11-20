import numpy as np
import numpy.typing as npt
import scipy
import optimizations
import utilities
from typing import Tuple

import ipdb


def alice_eve_entrophy_maximalization(
    visibility: float,
    density_matrix: npt.NDArray[np.complex128],
    angles: npt.NDArray[np.float64],
    x_star: int = 0,
    level: int = 1,
    difference: float = 10e-3,
    max_iters: int = 100,
) -> Tuple[float, npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Use npa hierarchy and basinhopping to finding the maximal value of H(A|x=x*,E) with values at which it occurred.

    Args:
        visibility: Visibility of quantum state.
        density_matrix: Initial density matrix of a state.
        angles: Initial set of angles used to generate the measurements.
        x_star: Input at which the key is generated.
        level: Level of npa hierarchy for guessing probability.
        difference: The difference between the entropy, above which the optimization continues running.
        max_iters: How many should the problem be optimized, ignoring the difference.

    Returns:
        Value of entropy, set of angles and reduced density matrix for which the value was found.
    """
    assert utilities.density_matrix_check(density_matrix)
    assert angles.ndim == 1 and angles.size % 2 == 0
    old_value = np.inf
    new_value = 100.0

    initial_values = np.tile(
        [2 * np.pi * i / utilities.STATES for i in range(utilities.STATES)], 2 * 2
    )
    i = 0
    # Minimazation loop
    while np.abs(old_value - new_value) > difference and max_iters > i:
        i += 1
        low_visibility_density_matrix = utilities.reduced_visibility_matrix(
            density_matrix, visibility
        )
        alice_pvm_stacked, bob_pvm_stacked = (
            utilities.pvm_from_angle_array(
                angles[: int(angles.size / 2)], utilities.INPUTS, utilities.STATES
            ),
            utilities.pvm_from_angle_array(
                angles[int(angles.size / 2) :], utilities.INPUTS, utilities.STATES
            ),
        )
        probabilities = utilities.probability_array(
            low_visibility_density_matrix, alice_pvm_stacked, bob_pvm_stacked
        )
        old_value = new_value
        guessing_probability, probabilities_coefficients = (
            optimizations.GuessingProbability(probabilities, x_star, level)
        )
        print(guessing_probability)
        guessing_probability = np.clip(guessing_probability, 0.0, 1.0)
        new_value = (
            1.0
            if guessing_probability < utilities.ATOL
            else -np.log(guessing_probability) / np.log(utilities.STATES)
        )
        # Optimal angle selection
        angles = scipy.optimize.basinhopping(
            lambda x: -optimizations.maximum_bell_expression(
                x, probabilities_coefficients
            )[0],
            initial_values,
            T=np.pi / 5,
            niter_success=50,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": initial_values.size * [(0.0, 2 * np.pi)],
            },
        ).x
        density_matrix = optimizations.maximum_bell_expression(
            angles, probabilities_coefficients
        )[1]
        print(new_value)
        print(old_value)
        print(np.abs(old_value - new_value))

    return (
        new_value,
        angles,
        utilities.reduced_visibility_matrix(density_matrix, visibility),
    )


def alice_bob_entropy_minimalization(
    density_matrix: npt.NDArray[np.complex128],
    alice_pvm_star_input: npt.NDArray[np.complex128],
) -> float:
    """Find the minimum value of entropy H(A|B, x=x*) through selecting Bob measurement settings.

    Args:
        density_matrix: Matrix of the shared quantum state.
        alice_pvm_star_input: Set of Alice pvm after she received x=x*.

    Returns:
        Minimal value of H(A|B, x=x*).
    """
    initial_values = np.tile(
        [2 * np.pi * i / utilities.STATES for i in range(utilities.STATES)], 1
    )
    bob_angles = scipy.optimize.basinhopping(
        lambda x: optimizations.alice_bob_entrophy(
            density_matrix, alice_pvm_star_input, utilities.parametrized_projections(*x)
        ),
        initial_values,
        T=np.pi / 5,
        niter_success=50,
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "bounds": initial_values.size * [(0.0, 2 * np.pi)],
        },
    ).x
    return optimizations.alice_bob_entrophy(
        density_matrix,
        alice_pvm_star_input,
        utilities.parametrized_projections(*bob_angles),
    )


def keyrate(
    visibility: float = 1.0,
    x_star: int = 0,
    verbose: bool = True,
    guessing_level: int = 1,
) -> float:
    """Find the key rate for the selected visibility.

    Args:
        visibility: Value for the state visibility.
        x_star: Input used for generating key.
        verbose: Should solver describe what it is doing.
        guessing_level: Level of npa hierarchy for guessing probability.

    Return:
        Key rate.
    """
    ipdb.set_trace()

    initial_state = np.ones(utilities.STATE_SHAPE[0]) / np.sqrt(
        utilities.STATE_SHAPE[0]
    )
    density = utilities.reduced_visibility_matrix(
        np.outer(initial_state, initial_state.conj().T), visibility
    )
    angles = np.tile(
        [2 * np.pi * i / utilities.STATES for i in range(utilities.STATES)],
        2 * utilities.STATES,
    )
    entrophy_AE, angles, density = alice_eve_entrophy_maximalization(
        visibility, density, angles, x_star, guessing_level
    )
    entrophy_AB = alice_bob_entropy_minimalization(
        density, utilities.parametrized_projections(angles[x_star], angles[x_star + 1])
    )
    key_rate_value = entrophy_AE - entrophy_AB
    if verbose:
        print(f"For V = {visibility}: r = {key_rate_value}")
    return key_rate_value


if __name__ == "__main__":
    keyrate(visibility=1.0)
