import numpy as np

from typing import Union


class Activation:
    pass


class Sigmoid(Activation):

    @classmethod
    def eval(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # use np.clip to avoid exp overflow
        return 1. / (1. + np.exp(np.clip(-x, -700., 700.)))

    @classmethod
    def grad(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return cls.eval(x) * (1. - cls.eval(x))


class Identity(Activation):

    @classmethod
    def eval(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x

    @classmethod
    def grad(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1.

