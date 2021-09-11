import numpy as np

from typing import Tuple

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return flattend data and label separately
        with shape (n, 784), (n, 1) where n is number of data points
    '''
    # get # lines in the dataset
    with open(file_path) as f:
        n = sum(1 for line in f)
    # pre-allocate result ndarray
    rtn = np.zeros((n, 785))
    # load data line by line
    with open(file_path) as f:
        for i in range(n):
            line = f.readline()
            for j, x in enumerate(line.strip().split(',')):
                rtn[i, j] = float(x)
    return rtn[:, 1:], rtn[:, :1]
