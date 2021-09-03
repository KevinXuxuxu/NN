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

class LogSoftmax(Cost):

    @classmethod
    def eval(cls, y_prime: np.ndarray, y: np.ndarray) -> float:
        return -y[y_prime == 1][0] + np.log(np.exp(y).sum())

    @classmethod
    def grad(cls, y_prime: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -1 + np.exp(y[y_prime == 1][0]) / np.exp(y).sum()
