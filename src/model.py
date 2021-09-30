import numpy as np
from time import time
from typing import Callable, Tuple

from src.activation import Activation, Sigmoid
from src.cost import Cost, MSE
from src.layer import Layer
from src.utils import check_shape, print_progress_bar

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
        else:
            self._input_dim = new_layer.n
        self._output_dim = new_layer.n
        self._layers.append(new_layer)
        return self

    def inference(self, _input: np.ndarray) -> np.ndarray:
        check_shape(_input.shape, (None, self._input_dim), axis=1)
        self._layers[0].a = _input
        self._layers[0].forward_pass()
        return self._layers[-1].a

    def get_cost(self, ground_truth: np.ndarray) -> float:
        check_shape(self._layers[-1].a.shape, ground_truth.shape)
        return self._cost.eval(self._layers[-1].a, ground_truth)

    def back_prop(self, ground_truth: np.ndarray):
        check_shape(self._layers[-1].a.shape, ground_truth.shape)
        y_grad = self._cost.grad(self._layers[-1].a, ground_truth)
        self._layers[-1].back_prop(y_grad)

    def train(self,
              train_set: Tuple[np.ndarray, np.ndarray],
              test_set: Tuple[np.ndarray, np.ndarray],
              validation_set: Tuple[np.ndarray, np.ndarray] = None,
              learning_rate: float = None,
              batch_size: int = 64,
              epochs: int = 10,
              get_accuracy: Callable[[np.ndarray, np.ndarray], float] = None):
        # check input dimension etc. and initialize parameters
        check_shape(train_set[0].shape, (None, self._input_dim), axis=1)
        check_shape(train_set[1].shape, (None, self._output_dim), axis=1)
        check_shape(test_set[0].shape, (None, self._input_dim), axis=1)
        check_shape(test_set[1].shape, (None, self._output_dim), axis=1)
        if not validation_set:
            validation_set = test_set
        check_shape(validation_set[0].shape, (None, self._input_dim), axis=1)
        check_shape(validation_set[1].shape, (None, self._output_dim), axis=1)
        if learning_rate is not None:
            self._rate = learning_rate
        if not get_accuracy:
            raise NotImplemented('get_accuracy')

        # break train_set into batch
        batched_train_set = [
            (train_set[0][i:i+batch_size], train_set[1][i:i+batch_size])
            for i in range(0, train_set[0].shape[0], batch_size)
        ]
        if batched_train_set[-1][0].shape[0] != batch_size:
            batched_train_set.pop()
        
        # start training
        cost, accuracy = [], []
        num_batches = len(batched_train_set)
        start_time = time()
        for epoch_idx in range(1, epochs+1):
            print("Training epoch {}: ".format(epoch_idx), end='', flush=True)
            percent = 1
            epoch_start_time = time()
            for batch_idx in range(num_batches):
                x, y = batched_train_set[batch_idx]
                result = self.inference(x)
                accuracy.append(get_accuracy(result, y))
                cost.append(self.get_cost(y))
                self.back_prop(y)
                percent = print_progress_bar(percent, batch_idx, num_batches)
            print()
            epoch_time = time() - epoch_start_time
            validation_result = self.inference(validation_set[0])
            validation_accuracy = get_accuracy(validation_result, validation_set[1])
            print('Epoch training time: {:.2f}, validation accuracy: {:.4f}'.format(
                epoch_time, validation_accuracy))
        total_time = time() - start_time
        test_result = self.inference(test_set[0])
        test_accuracy = get_accuracy(test_result, test_set[1])
        print('Total training time: {:.2f}, test accuracy: {:.4f}'.format(
            total_time, test_accuracy))
        return cost, accuracy
