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
        return (-y_prime[y == 1] + np.log(np.exp(y_prime).sum(axis=0))).sum()

    @classmethod
    def grad(cls, y_prime: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -(y == 1).astype('float') + np.exp(y_prime) / np.exp(y_prime).sum(axis=0)
