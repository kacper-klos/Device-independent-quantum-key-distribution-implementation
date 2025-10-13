import ncpol2sdpa as ncpol
import numpy as np
import numpy.typing as npt
import sympy
from typing import List, Tuple
from itertools import product


def GuessingProbability(
    probabilities: npt.NDArray[np.float64], x_star: int = 0, level: int = 1
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
                # Indepotent
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
    # Commutivity
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
    # Return biggest value and dual variables for each probability
    return -sdp.primal, probabilities_coefficients
