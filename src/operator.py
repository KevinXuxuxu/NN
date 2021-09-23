import numpy as np

from typing import List


class Operator:

    def __init__(self):
        self.preds, self.succs= [], []
        self.forward_pass_count, self.back_prod_count = 0, 0

    def _forward_pass(self):
        raise NotImplemented('forward_pass')

    def _back_prop(self) -> np.ndarray:
        raise NotImplemented('back_prop')

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
