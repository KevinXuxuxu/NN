import numpy as np

from typing import List
from src.operator.operator import Operator
from src.utils import check_shape


class Loss(Operator):

    def __init__(self, ishape: List[int]):
        super().__init__(ishape, [])
        self.preds = [None, None]

    def register(self, pred: Operator):
        raise NotImplemented(
            'Regular register not supported for loss function, ' +
            'use register_output and register_ground_truth')

    def register_output(self, pred: Operator):
        if self.preds[0] is not None:
            raise Exception('Output for loss is already set')
        check_shape(self.ishape, pred.oshape)
        self.preds[0] = pred
        pred.succs.append(self)

    def register_ground_truth(self, pred: Operator):
        if self.preds[1] is not None:
            raise Exception('Ground truth for loss is already set')
        check_shape(self.ishape, pred.oshape)
        self.preds[1] = pred
        pred.succs.append(self)

    @property
    def y(self):
        return self.preds[0].output

    @property
    def gt(self):
        return self.preds[1].output


class MSE(Loss):

    def _forward_pass(self):
        if self.y.shape != self.gt.shape:
            raise ValueError('Model output and ground truth have different shape')
        self.output = ((self.gt - self.y) ** 2).sum()
        # super()._forward_pass()

    def _back_prop(self):
        if self.y.shape != self.gt.shape:
            raise ValueError('Model output and ground truth have different shape')
        self.grad = 2 * (self.gt - self.y)
        super()._back_prop()
