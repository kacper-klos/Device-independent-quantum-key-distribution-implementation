import numpy as np
import numpy.typing as npt
from optimizations import GuessingProbability

level = 3


def perfect_chsh() -> npt.NDArray[np.float64]:
    probabilities = np.zeros([2, 2, 2, 2])
    p_win = np.cos(np.pi / 8) ** 2
    p_lose = 1 - p_win
    for x in range(2):
        for y in range(2):
            for a in range(2):
                for b in range(2):
                    if (a ^ b) == (x * y):
                        probabilities[a, b, x, y] = p_win / 2.0
                    else:
                        probabilities[a, b, x, y] = p_lose / 2.0
    return probabilities


def print_guessing_result(
    input_probabilites: np.ndarray, output_guessing_probability: float, expect: float
) -> None:
    print(f"Problem output: {output_guessing_probability}")
    print(f"Expected : {expect}")


def even_probability_test() -> None:
    probabilities = np.ones([2, 2, 2, 2])
    probabilities /= np.sum(probabilities[:, :, 0, 0])
    print_guessing_result(
        probabilities, GuessingProbability(probabilities, level=level)[0], 0.5
    )


def deterministic_probability_test() -> None:
    probabilities = np.zeros([2, 2, 2, 2])
    for x_i in range(2):
        for y_i in range(2):
            probabilities[0, 0, x_i, y_i] = 1
    print_guessing_result(
        probabilities, GuessingProbability(probabilities, level=level)[0], 1.0
    )


def random_probability_test(p: float) -> None:
    assert p <= 1.0 and p >= 0
    probabilities = np.zeros([2, 2, 2, 2])
    for x_i in range(2):
        for y_i in range(2):
            probabilities[0, 0, x_i, y_i] = p
            probabilities[1, 1, x_i, y_i] = 1 - p
    print_guessing_result(
        probabilities, GuessingProbability(probabilities, level=level)[0], 1.0
    )


def noisy_chsh_probability_test(v: float) -> None:
    if v <= 1.0 and v > 1 / np.sqrt(2):
        expected = (1 + np.sqrt(2 - 2 * (v**2))) / 2
    else:
        expected = 1

    probabilities = v * perfect_chsh() + (1 - v) / 4 * np.ones([2, 2, 2, 2])
    print_guessing_result(
        probabilities, GuessingProbability(probabilities, level=level)[0], expected
    )


if __name__ == "__main__":
    even_probability_test()
    deterministic_probability_test()
    random_probability_test(0.9)
    random_probability_test(0.6)
    random_probability_test(0.2)
    noisy_chsh_probability_test(1.0)
    noisy_chsh_probability_test(0.9)
    noisy_chsh_probability_test(0.8)
    noisy_chsh_probability_test(0.75)
    noisy_chsh_probability_test(0.70)
    noisy_chsh_probability_test(0.40)
