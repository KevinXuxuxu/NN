import numpy as np

from typing import List, Tuple
from src.utils import check_shape


class Operator:

    def __init__(self, ishape: List[int], oshape: List[int]):
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        self.preds, self.succs= [], []
        self.forward_pass_count, self.back_prod_count = 0, 0

    def _forward_pass(self):
        # override in subclass with this as callback
        for succ in self.succs:
            succ.forward_pass()

    def _back_prop(self):
        # override in subclass with this as callback
        for pred in self.preds:
            pred.back_prop()

    def forward_pass(self):
        self.forward_pass_count += 1
        if self.forward_pass_count == len(self.preds):
            self._forward_pass()
            self.forward_pass_count = 0

    def back_prop(self):
        self.back_prod_count += 1
        if self.back_prod_count == len(self.succs):
            self._back_prop()
            self.back_prod_count = 0

    def register(self, pred: 'Operator'):
        self.preds.append(pred)
        pred.succs.append(self)


class ParameterizedOperator(Operator):

    rate = 0.001

    def __init__(self, ishape: List[int], oshape: List[int], rate: float = None):
        super().__init__(ishape, oshape)
        self._rate = rate if rate is not None else self.__class__.rate


class MatMul(ParameterizedOperator):
    
    def __init__(self, ishape: List[int], oshape: List[int]):
        super().__init__(ishape, oshape)
        # weights initialized to [-0.1, 0.1]
        self._w = np.random.random((ishape[0], oshape[0])) / 5. - .1

    def register(self, pred: 'Operator'):
        check_shape(pred.oshape, self.ishape)
        super().register(pred)

    def _forward_pass(self):
        self.output = np.matmul(self.preds[0].output, self._w)
        super()._forward_pass()

    def _back_prop(self):
        dL_do = np.sum(np.array([succ.grad for succ in self.succs]))
        dL_dw = np.einsum('ki,kj', self.preds[0].output, dL_do)
        self._w -= self._rate * dL_dw
        self.grad = np.einsum('ik,jk', dL_do, self._w)
        super()._back_prop()
