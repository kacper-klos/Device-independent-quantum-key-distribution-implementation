import ncpol2sdpa as ncpol
import numpy as np
import numpy.typing as npt
import sympy
from typing import List, Dict, Tuple
from itertools import product


def GuessingProbability(
    probabilities: npt.NDArray[np.float64], x_star: int = 0, level: int = 2
) -> Tuple[float, npt.NDArray[np.float64]]:
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
    quantum_equalities: List[sympy.physics.quantum.operator.HermitianOperator] = []

    def QuantumBehaviorBounds(
        operators: List[sympy.physics.quantum.operator.HermitianOperator],
        equalities: List[sympy.core.expr.Expr],
    ) -> None:
        for input_operators_set in operators:
            for i, operator in enumerate(input_operators_set):
                # Indepotent
                equalities.append(operator * operator - operator)
                for other_operator in input_operators_set[:i]:
                    # Orthogonality
                    equalities.append(operator * other_operator)
            # Summing to identity
            equalities.append(
                sum(input_operators_set[i] for i in range(len(input_operators_set))) - 1
            )

    QuantumBehaviorBounds(A, quantum_equalities)
    QuantumBehaviorBounds(B, quantum_equalities)
    # Commutivity
    for a, b, x, y in product(
        range(a_size), range(b_size), range(x_size), range(y_size)
    ):
        quantum_equalities.append(A[x][a] * B[y][b] - B[y][b] * A[x][a])
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
    # Creator optimization
    a_value: Dict[int, Tuple[float, List[float]]] = {}
    operators: List[sympy.physics.quantum.operator.HermitianOperator] = []
    for a in range(a_size):
        for x in range(x_size):
            operators.append(A[x][a])
    for b in range(b_size):
        for y in range(y_size):
            operators.append(B[y][b])
    for a in range(a_size):
        sdp = ncpol.SdpRelaxation(verbose=1, variables=operators)
        sdp.get_relaxation(
            level=level,
            objective=-A[x_star][a],
            equalities=quantum_equalities,
            momentequalities=probabilities_equalities,
        )
        sdp.solve(solver="mosek")
        type(sdp.y_mat)
        a_value[a] = (
            -sdp.primal,
            [sdp.extract_dual_value(raw[1]) for raw in raw_equalities],
        )
    # Get the best solution
    max_tuple = max(a_value.items(), key=lambda x: x[1][0])
    probabilities_coefficients = np.zeros(probabilities.shape)
    for i, indexes in enumerate(raw_equalities):
        probabilities_coefficients[indexes[0]] = max_tuple[1][1][i]
    # Return biggest value and dual variables for each probability
    return max_tuple[1][0], probabilities_coefficients
