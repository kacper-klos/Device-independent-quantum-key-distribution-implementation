# This file consists of low-level optimizations independent of other modules.
import ncpol2sdpa as ncpol
import numpy as np
import numpy.typing as npt
import sympy
import utilities
from typing import List, Tuple
from itertools import product


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
    a_size, b_size, x_size, y_size = probabilities.shape
    # Create operator for Alice, Bob and Eve
    A = [
        ncpol.generate_operators(f"A_({i})", a_size, hermitian=True)
        for i in range(x_size)
    ]
    B = [
        ncpol.generate_operators(f"B_({i})", b_size, hermitian=True)
        for i in range(y_size)
    ]
    E = [ncpol.generate_operators(f"E_({0})", a_size, hermitian=True)]
    # Create constraints satisfied by quantum systems
    quantum_equalities: List[sympy.core.expr.Expr] = []
    quantum_inequalities: List[sympy.core.expr.Expr] = []

    def QuantumBehaviorBounds(
        operators: List[sympy.physics.quantum.operator.HermitianOperator],
    ) -> None:
        for input_operators_set in operators:
            for i, operator in enumerate(input_operators_set):
                quantum_inequalities.append(operator)
                # Idempotent
                quantum_equalities.append(operator * operator - operator)
                for other_operator in input_operators_set[:i]:
                    # Orthogonality
                    quantum_equalities.append(operator * other_operator)
            # Summing to identity
            quantum_equalities.append(
                sum(input_operators_set[i] for i in range(len(input_operators_set))) - 1
            )

    QuantumBehaviorBounds(A)
    QuantumBehaviorBounds(B)
    QuantumBehaviorBounds(E)
    # Commutativity
    for a, b, x, y in product(
        range(a_size), range(b_size), range(x_size), range(y_size)
    ):
        quantum_equalities.append(A[x][a] * B[y][b] - B[y][b] * A[x][a])
    for a, c, x in product(range(a_size), range(a_size), range(x_size)):
        quantum_equalities.append(E[0][c] * A[x][a] - A[x][a] * E[0][c])
    for b, c, y in product(range(b_size), range(a_size), range(y_size)):
        quantum_equalities.append(E[0][c] * B[y][b] - B[y][b] * E[0][c])
    # Set bounds for probabilities
    probabilities_equalities: List[
        sympy.physics.quantum.operator.HermitianOperator
    ] = []
    raw_equalities: List[
        Tuple[
            Tuple[int, int, int, int], sympy.physics.quantum.operator.HermitianOperator
        ]
    ] = []
    for a, b, x, y in product(
        range(a_size), range(b_size), range(x_size), range(y_size)
    ):
        probabilities_equalities.append(
            A[x][a] * B[y][b] - float(probabilities[a, b, x, y])
        )
        raw_equalities.append(((a, b, x, y), A[x][a] * B[y][b]))
    # Gather operators
    operators: List[sympy.physics.quantum.operator.HermitianOperator] = []
    for a, x in product(range(a_size), range(x_size)):
        operators.append(A[x][a])
    for b, y in product(range(b_size), range(y_size)):
        operators.append(B[y][b])
    for a in range(a_size):
        operators.append(E[0][a])
    # Creator optimization
    sdp = ncpol.SdpRelaxation(verbose=1, variables=operators)
    sdp.get_relaxation(
        level=level,
        objective=-sum(A[x_star][a] * E[0][a] for a in range(a_size)),
        equalities=quantum_equalities,
        inequalities=quantum_inequalities,
        momentequalities=probabilities_equalities,
    )
    sdp.solve(solver="mosek")
    # Get the best output
    list_dual_value = [sdp.extract_dual_value(raw[1]) for raw in raw_equalities]
    probabilities_coefficients = np.zeros(probabilities.shape)
    for i, indexes in enumerate(raw_equalities):
        probabilities_coefficients[indexes[0]] = list_dual_value[i]
    # Return the optimal value and dual variables corresponding to each probability.
    return -sdp.primal, probabilities_coefficients


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
