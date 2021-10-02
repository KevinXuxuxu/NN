import numpy as np

from src.operator.operator import Operator

SIGMOID_INPUT_CAP = 700.


class BroadcastOperator(Operator):

    def __init__(self):
        super().__init__([], [])

    def register(self, pred: Operator):
        if len(self.preds) == 1:
            raise Exception('Multiple input not supported for broadcat operator')
        # pass the input shape as output shape
        self.oshape = pred.oshape
        return super().register(pred)

    def _eval(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented('_eval')

    def _grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented('_grad')

    def _forward_pass(self):
        self.output = self._eval(self.preds[0].output)
        super()._forward_pass()

    def _back_prop(self):
        dL_do = self._aggregate_grad()
        do_dh = self._grad(self.output)
        self.grad = dL_do * do_dh
        super()._back_prop()


class Sigmoid(BroadcastOperator):

    def _eval(self, x: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(np.clip(-x, -SIGMOID_INPUT_CAP, SIGMOID_INPUT_CAP)))

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return self._eval(x) * (1. - self._eval(x))


class Identity(BroadcastOperator):

    def _eval(cls, x: np.ndarray) -> np.ndarray:
        return x

    def _grad(cls, x: np.ndarray) -> np.ndarray:
        return 1.


class ReLU(BroadcastOperator):

    def _eval(cls, x: np.ndarray) -> np.ndarray:
        rtn = x.copy()
        rtn[x < 0.] = 0.
        return rtn

    def _grad(cls, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype('float')
