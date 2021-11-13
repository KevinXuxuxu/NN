import numpy as np

from typing import List
from src.operator.operator import Operator, ParameterizedOperator
from src.utils import check_shape


class Linear(ParameterizedOperator):
    
    def __init__(self, ishape: List[int], oshape: List[int]):
        super().__init__(ishape, oshape)
        # weights initialized to [-0.1, 0.1]
        self._w = np.random.random((ishape[0], oshape[0])) / 5. - .1

    def add_pred(self, pred: Operator):
        if len(self.preds) == 1:
            raise Exception('Multiple input not supported for Linear')
        check_shape(pred.oshape, self.ishape)
        super().add_pred(pred)

    def _forward_pass(self):
        self.output = np.matmul(self.preds[0].output, self._w)
        super()._forward_pass()

    def _back_prop(self):
        dL_do = self._aggregate_grad()
        dL_dw = np.einsum('ki,kj', self.preds[0].output, dL_do)
        self.grad = np.einsum('ik,jk', dL_do, self._w)
        self._w -= self._rate * dL_dw
        super()._back_prop()


class Bias(ParameterizedOperator):

    def __init__(self, ishape: List[int], oshape: List[int]):
        super().__init__(ishape, oshape)
        # bias initialized to 0.1
        self._b = np.ones(ishape) * .1

    def add_pred(self, pred: Operator):
        if len(self.preds) == 1:
            raise Exception('Multiple input not supported for bias')
        check_shape(pred.oshape, self.ishape)
        super().add_pred(pred)

    def _forward_pass(self):
        self.output = self.preds[0].output + self._b
        super()._forward_pass()

    def _back_prop(self):
        dL_do = self._aggregate_grad()
        dL_db = np.sum(dL_do, axis=0)
        self._b -= self._rate * dL_db
        self.grad = dL_do
        super()._back_prop()


class DataInput(Operator):

    def __init__(self, data: np.ndarray):
        self.output = data
        super().__init__([], data.shape[1:])

    def forward_pass(self):
        self._forward_pass()

    def _back_prop(self):
        self.grad_on_data = self._aggregate_grad()

    def add_pred(self, pred: 'Operator'):
        raise Exception('Add pred is not supported for DataInput operator')
