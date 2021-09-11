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
        return (-y_prime[y == 1] + np.log(np.exp(y_prime).sum(axis=1))).sum()

    @classmethod
    def grad(cls, y_prime: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -(y == 1).astype('float') + np.exp(y_prime) / np.exp(y_prime).sum(axis=1)[:, np.newaxis]

class CrossEntropyWithSoftmax(Cost):
    '''
        reference: https://deepnotes.io/softmax-crossentropy#cross-entropy-loss    
    '''

    @classmethod
    def _softmax(cls, input: np.ndarray) -> np.ndarray:
        exp = np.exp(input)
        return exp / exp.sum(axis=1)[:, np.newaxis]

    @classmethod
    def eval(cls, y_prime: np.ndarray, y: np.ndarray) -> float:
        m = y.shape[1]
        p = cls._softmax(y_prime)
        log_likelihood = -np.log(np.max(p*y, axis=1))
        return np.sum(log_likelihood) / m
    
    @classmethod
    def grad(cls, y_prime: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (cls._softmax(y_prime) - y) / y.shape[1]
