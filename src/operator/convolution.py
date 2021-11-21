import numpy as np

from typing import List
from src.operator.operator import Operator
from src.utils import check_shape


class Conv2D(Operator):

    def __init__(self,
                 ishape: List[int],
                 oshape: List[int],
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 0):
        # ishape = (in_channels, image_height, image_width)
        # oshape = (out_channels, image_height, image_width)
        super().__init__(ishape, oshape)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._w = 1e-3 * np.random.randn(oshape[0], ishape[0], kernel_size, kernel_size)
        self._b = np.zeros(oshape[0])

    def add_pred(self, pred: Operator):
        if len(self.preds) == 1:
            raise Exception('Multiple input not supported for Conv2D')
        check_shape(pred.oshape, self.ishape)
        super().add_pred(pred)

    def _get_sub_matrices(self, x: np.ndarray) -> np.ndarray:
        k, s = self.kernel_size, self.stride
        t1 = (x.shape[-2] - k) // s + 1
        t2 = (x.shape[-1] - k) // s + 1
        vshape = x.shape[:-2] + (t1, t2) + (k, k)
        strides = x.strides[:2] + (x.strides[-2] * s, x.strides[-1] * s) + x.strides[-2:]
        return np.lib.stride_tricks.as_strided(x, vshape, strides)

    def _forward_pass(self):
        self.output = np.pad(
            self.preds[0].output,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        sub_matrics = self._get_sub_matrices(self.preds[0].output)
        super()._forward_pass()