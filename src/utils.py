import numpy as np
from typing import Tuple

def check_shape(x: np.ndarray, expected_shape: Tuple[int, int], axis: int = None):
    if axis is None and x.shape == expected_shape:
        return
    if axis is not None and x.shape[axis] == expected_shape[axis]:
        return
    raise Exception('tested shape {} does not match expected shape {}'.format(
        x.shape, expected_shape))

def print_progress_bar(percent: int, i: int, n: int) -> int:
    if i > percent * n / 100.:
        if percent % 10 == 0:
            print('{}%'.format(percent), end='', flush=True)
        else:
            print('.', end='', flush=True)
        while i > percent * n / 100:
            percent += 1
    return percent

