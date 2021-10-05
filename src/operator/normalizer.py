import numpy as np

from src.operator.operator import Operator

class Normalizer(Operator):

    def __init__(self):
        super().__init__([], [])

    def add_pred(self, pred: Operator):
        if len(self.preds) == 1:
            raise Exception('Multiple input not supported for normalizer')
        # pass the input shape as output shape
        self.oshape = pred.oshape
        return super().add_pred(pred)

    def _eval(self, x: np.ndarray) -> np.ndarray:
        raise NotImplemented('_eval')

    def _forward_pass(self):
        self.output = self._eval(self.preds[0].output)
        super()._forward_pass()


class Softmax(Normalizer):

    def add_pred(self, pred: Operator):
        if len(pred.oshape) > 1:
            raise ValueError('Multi-dimensional input not supported for softmax')
        super().add_pred(pred)

    def _eval(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / exp.sum(axis=1)[:, np.newaxis]

    def _back_prop(self):
        n, m = self.output.shape
        dL_do = self._aggregate_grad()
        # do_j/dx_i = o_i * (δ_ij - o_j), δ is Kronecker delta
        # dL/dx_i = ∑(dL/do_j * do_j/dx_i)
        # do/dx is tensor over softmax
        # tensor product of o
        oij_tensor = np.einsum('...i,...j', self.output, self.output)
        # TODO: is there any way to simplify the following part?
        # kronecker delta tensor
        kd_tensor = np.repeat(np.diag(np.ones((m))).reshape((1, m, m)), n, axis=0)
        # o_i * δ_ij tensor
        o_kd_tensor = np.repeat(self.output.reshape((n, m, 1)), m, axis=2) * kd_tensor
        do_dx = o_kd_tensor - oij_tensor
        self.grad = np.sum(dL_do.reshape((n, m, 1)) * do_dx, axis=1) # dL_dx
        super()._back_prop()
