import numpy as np

from typing import List


class Operator:

    def __init__(self, ishape: List[int], oshape: List[int]):
        self.ishape, self.oshape = tuple(ishape), tuple(oshape)
        self.preds, self.succs = [], []
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

    def _aggregate_grad(self) -> np.ndarray:
        # sum up gradients from all successors for back prop
        return np.sum(np.array([succ.grad for succ in self.succs]), axis=0)


class ParameterizedOperator(Operator):

    rate = 0.001

    def __init__(self, ishape: List[int], oshape: List[int], rate: float = None):
        super().__init__(ishape, oshape)
        self._rate = rate if rate is not None else self.__class__.rate
