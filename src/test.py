import math
from typing import Tuple

import numpy as np
import pytest

import utilities


@pytest.fixture
def product_state() -> np.ndarray:
    """|00> state."""
    vec = np.zeros(utilities.STATE_SHAPE[0], dtype=np.complex128)
    vec[0] = 1.0
    return vec


@pytest.fixture
def product_density(product_state: np.ndarray) -> np.ndarray:
    return np.outer(product_state, product_state.conj())


@pytest.fixture
def pvm_z() -> np.ndarray:
    return utilities.parametrized_projections(0.0, 0.0)


@pytest.fixture
def pvm_x() -> np.ndarray:
    return utilities.parametrized_projections(math.pi / 4, 0.0)


@pytest.fixture
def valid_pvm_stack(pvm_z: np.ndarray, pvm_x: np.ndarray) -> np.ndarray:
    return np.stack([pvm_z, pvm_x])


@pytest.fixture
def bell_measurements() -> Tuple[np.ndarray, np.ndarray]:
    angles = np.array([0.0, 0.0, math.pi / 4, 0.0])
    alice = utilities.pvm_from_angle_array(angles, utilities.INPUTS, utilities.STATES)
    bob = utilities.pvm_from_angle_array(angles, utilities.INPUTS, utilities.STATES)
    return alice, bob


# --- pvm_check ---
def test_pvm_check_accepts_valid_set(valid_pvm_stack: np.ndarray) -> None:
    assert utilities.pvm_check(valid_pvm_stack)


def test_pvm_check_rejects_invalid_set(valid_pvm_stack: np.ndarray) -> None:
    broken = valid_pvm_stack.copy()
    broken[0, 0, :, :] = np.eye(utilities.STATES) * 0.5
    assert not utilities.pvm_check(broken)


# --- is_hermitian_psd ---
def test_is_hermitian_psd_accepts_identity() -> None:
    assert utilities.is_hermitian_psd(np.eye(2, dtype=np.complex128))


def test_is_hermitian_psd_rejects_negative_eigenvalue() -> None:
    matrix = np.diag([1.0, -0.2]).astype(np.complex128)
    assert not utilities.is_hermitian_psd(matrix)


# --- density_matrix_check ---
def test_density_matrix_check_valid(product_density: np.ndarray) -> None:
    assert utilities.density_matrix_check(product_density)


def test_density_matrix_check_invalid_trace(product_density: np.ndarray) -> None:
    invalid = product_density * 1.2
    assert not utilities.density_matrix_check(invalid.astype(np.complex128))


# --- parametrized_state ---
def test_parametrized_state_is_normalized() -> None:
    state = utilities.parametrized_state(theta=0.27, psi=0.13)
    assert pytest.approx(np.vdot(state, state).real) == 1.0


def test_parametrized_state_matches_expected_amplitudes() -> None:
    theta = math.pi / 4
    psi = math.pi / 3
    state = utilities.parametrized_state(theta, psi)
    expected = np.array(
        [math.cos(theta), np.exp(1j * psi) * math.sin(theta)], dtype=np.complex128
    )
    np.testing.assert_allclose(state, expected, atol=1e-12)


# --- parametrized_projections ---
def test_parametrized_projections_are_projectors() -> None:
    pvm = utilities.parametrized_projections(theta=0.17, psi=0.93)
    for projector in pvm:
        np.testing.assert_allclose(projector, projector.conj().T, atol=1e-10)
        np.testing.assert_allclose(projector @ projector, projector, atol=1e-10)


def test_parametrized_projections_sum_to_identity() -> None:
    pvm = utilities.parametrized_projections(theta=0.42, psi=1.1)
    np.testing.assert_allclose(
        np.sum(pvm, axis=0), np.eye(utilities.STATES), atol=1e-10
    )


# --- pvm_from_angle_array ---
def test_pvm_from_angle_array_shape_and_validity() -> None:
    angles = np.array([0.0, 0.0, math.pi / 5, math.pi / 7])
    pvm_stack = utilities.pvm_from_angle_array(angles, inputs=2, outputs=2)
    assert pvm_stack.shape == (2, 2, utilities.STATES, utilities.STATES)
    assert utilities.pvm_check(pvm_stack)


def test_pvm_from_angle_array_raises_for_invalid_length() -> None:
    with pytest.raises(AssertionError):
        utilities.pvm_from_angle_array(np.array([0.0, 0.0, 1.0]), inputs=2, outputs=2)


# --- reduced_visibility_matrix ---
def test_reduced_visibility_matrix_limits(product_density: np.ndarray) -> None:
    full = utilities.reduced_visibility_matrix(product_density, 1.0)
    depolarized = utilities.reduced_visibility_matrix(product_density, 0.0)
    np.testing.assert_allclose(full, product_density, atol=1e-12)
    np.testing.assert_allclose(
        depolarized,
        np.eye(product_density.shape[0], dtype=np.complex128)
        / product_density.shape[0],
        atol=1e-12,
    )


def test_reduced_visibility_matrix_preserves_density(
    product_density: np.ndarray,
) -> None:
    reduced = utilities.reduced_visibility_matrix(product_density, 0.35)
    assert utilities.density_matrix_check(reduced)


# --- bell_expression ---
def test_bell_expression_single_coefficient(
    bell_measurements: Tuple[np.ndarray, np.ndarray],
) -> None:
    alice, bob = bell_measurements
    coefficients = np.zeros((2, 2, 2, 2))
    coefficients[0, 0, 0, 0] = 1.0
    bell = utilities.bell_expression(coefficients, alice, bob)
    expected = np.kron(alice[0, 0], bob[0, 0])
    np.testing.assert_allclose(bell, expected, atol=1e-10)


def test_bell_expression_is_linear(
    bell_measurements: Tuple[np.ndarray, np.ndarray],
) -> None:
    alice, bob = bell_measurements
    coefficients = np.zeros((2, 2, 2, 2))
    coefficients[0, 0, 0, 0] = 0.6
    coefficients[1, 1, 1, 1] = -0.4
    bell = utilities.bell_expression(coefficients, alice, bob)
    expected = 0.6 * np.kron(alice[0, 0], bob[0, 0]) - 0.4 * np.kron(
        alice[1, 1], bob[1, 1]
    )
    np.testing.assert_allclose(bell, expected, atol=1e-10)


# --- probability_array ---
def test_probability_array_for_product_state(
    product_density: np.ndarray, valid_pvm_stack: np.ndarray
) -> None:
    alice = np.stack([valid_pvm_stack[0], valid_pvm_stack[0]])
    bob = np.stack([valid_pvm_stack[0], valid_pvm_stack[0]])
    probs = utilities.probability_array(product_density, alice, bob)
    expected = np.zeros_like(probs)
    expected[0, 0, :, :] = 1.0
    np.testing.assert_allclose(probs, expected, atol=1e-12)


def test_probability_array_normalization(
    product_density: np.ndarray, valid_pvm_stack: np.ndarray
) -> None:
    alice = valid_pvm_stack
    bob = valid_pvm_stack
    probs = utilities.probability_array(product_density, alice, bob)
    totals = np.sum(probs, axis=(0, 1))
    np.testing.assert_allclose(
        totals, np.ones((alice.shape[0], bob.shape[0])), atol=1e-12
    )
