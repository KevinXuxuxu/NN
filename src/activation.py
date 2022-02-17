from typing import Union

import numpy as np


class Activation:
    pass


class Sigmoid(Activation):
    @classmethod
    def eval(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # use np.clip to avoid exp overflow
        return 1.0 / (1.0 + np.exp(np.clip(-x, -700.0, 700.0)))

    @classmethod
    def grad(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return cls.eval(x) * (1.0 - cls.eval(x))


class Identity(Activation):
    @classmethod
    def eval(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return x

    @classmethod
    def grad(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 1.0


class ReLU(Activation):
    @classmethod
    def eval(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        rtn = x.copy()
        rtn[x < 0.0] = 0.0
        return rtn

    @classmethod
    def grad(cls, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (x > 0).astype("float")
