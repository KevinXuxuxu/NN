import logging
import numpy as np

from src.cost import Cost, MSE
from src.layer import Layer
from src.activation import Activation, Sigmoid

DEFAULT_LEARNING_RATE = 0.001

class Model:

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE, cost: Cost = MSE):
        self._cost = cost
        self._rate = learning_rate
        self._layers = []

    def layer(self, n: int, act: Activation = Sigmoid):
        new_layer = Layer(n, self._rate, act)
        if self._layers:
            new_layer.connect(self._layers[-1])
        self._layers.append(new_layer)
        return self

    def inference(self, _input: np.ndarray) -> np.ndarray:
        if _input.shape[0] != self._layers[0].n:
            raise Exception('input shape {} does not match first layer dimension {}'.format(
                _input.shape, self._layers[0].n))
        self._layers[0].a = _input
        self._layers[0].forward_pass()
        return self._layers[-1].a

    def get_cost(self, ground_truth: np.ndarray) -> float:
        if ground_truth.shape != self._layers[-1].a.shape:
            raise Exception('ground truth shape {} does not match last layer shape {}'.format(
                ground_truth.shape, self._layers[-1].a.shape))
        return self._cost.eval(self._layers[-1].a, ground_truth)

    def back_prop(self, ground_truth: np.ndarray):
        if ground_truth.shape != self._layers[-1].a.shape:
            raise Exception('ground truth shape {} does not match last layer shape {}'.format(
                ground_truth.shape, self._layers[-1].a.shape))
        y_grad = self._cost.grad(self._layers[-1].a, ground_truth)
        self._layers[-1].back_prop(y_grad)
