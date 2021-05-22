import numpy as np


class Cost:
    pass


class MSE(Cost):

    @classmethod
    def eval(cls, y_prime: np.ndarray, y: np.ndarray) -> float:
        return ((y_prime - y) ** 2).sum()

    @classmethod
    def grad(cls, y_prime: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * (y_prime - y)
