import numpy as np
from optimizations import GuessingProbability


def print_guessing_result(
    input_probabilites: np.ndarray, output_guessing_probability: float
) -> None:
    print(f"Problem input:\n{input_probabilites}")
    print(f"Problem output: {output_guessing_probability}")


def even_probability_test() -> None:
    probabilities = np.ones([2, 2, 2, 2])
    probabilities /= np.sum(probabilities[:, :, 0, 0])
    print_guessing_result(probabilities, GuessingProbability(probabilities)[0])


def deterministic_probability_test() -> None:
    probabilities = np.zeros([2, 2, 2, 2])
    for x_i in range(2):
        for y_i in range(2):
            probabilities[0, 0, x_i, y_i] = 1
    print_guessing_result(probabilities, GuessingProbability(probabilities)[0])


def ideal_chsh_probability_test() -> None:
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
    # Expect result around 1/2
    print_guessing_result(probabilities, GuessingProbability(probabilities)[0])


if __name__ == "__main__":
    even_probability_test()
    deterministic_probability_test()
    ideal_chsh_probability_test()
