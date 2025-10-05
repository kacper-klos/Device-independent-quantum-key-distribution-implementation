import ncpol2sdpa as ncpol
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple
from itertools import product


def GuessingProbability(
    probabilities: npt.NDArray[np.float64], x_star: int = 0, level: int = 2
) -> Tuple[float, List[float]]:
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
    # Create constraints satisfied by quantum systems
    quantum_equalities: List[ncpol.nc_utils.Operator] = []

    def QuantumBehaviorBounds(
        operators: List[ncpol.nc_utils.Operator],
        equalities: List[ncpol.nc_utils.Polynomial],
    ) -> None:
        for input_operators_set in operators:
            for operator in input_operators_set:
                # Indepotent
                equalities.append(operator * operator - operator)
                for other_operator in input_operators_set:
                    if operator != other_operator:
                        # Orthogonality
                        equalities.append(operator * other_operator)
            # Summing to identity
            equalities.append(
                sum(input_operators_set for _ in range(len(input_operators_set))) - 1
            )

    QuantumBehaviorBounds(A, quantum_equalities)
    QuantumBehaviorBounds(B, quantum_equalities)
    # Commutivity
    for a, b, x, y in product(
        range(a_size), range(b_size), range(x_size), range(y_size)
    ):
        quantum_equalities.append(A[x][a] * B[y][b] - B[y][b] * A[x][a])
    # Set bounds for probabilities
    probabilities_equalities: List[ncpol.nc_utils.Operator] = []
    probabilities_equalities_input: List[Tuple[int, int, int, int]] = []
    for a, b, x, y in product(
        range(a_size), range(b_size), range(x_size), range(y_size)
    ):
        probabilities_equalities.append(
            A[x][a] * B[y][b] - float(probabilities[a, b, x, y])
        )
        probabilities_equalities_input.append((a, b, x, y))
    # Creator optimization
    a_value: Dict[int, float] = {}
    operators: List[ncpol.nc_utils.Operator] = []
    for a in range(a_size):
        for x in range(x_size):
            operators.append(A[x][a])
    for b in range(b_size):
        for y in range(y_size):
            operators.append(B[y][b])
    for a in range(a_size):
        sdp = ncpol.SdpRelaxation(verbose=0, variables=operators)
        sdp.get_relaxation(
            level=level,
            objective=A[x_star][a],
            equalities=quantum_equalities,
            momentequalities=probabilities_equalities,
            removeequalities=True,
        )
        sdp.solve(solver="mosek")
        a_value[a] = sdp.primal
    # Get the dual variables os probabilities
    dual_variables = sdp.y[len(quantum_equalities) :]
    probabilities_coefficients = np.zeros(probabilities.shape)
    for i, coefficient in enumerate(probabilities_equalities):
        probabilities_coefficients[probabilities_equalities_input[i]] = coefficient
    # Return biggest value and dual variables for each probability
    return max(a_value.values()), dual_variables
