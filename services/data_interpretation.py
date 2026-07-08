# ###################
# Ian Janes
# Prof. Don Lipkin
# Data interpretation service file
# ###################

from numpy.typing import NDArray
import numpy as np


def report_chop_results(times: NDArray[np.float64], temps: NDArray[np.float64]):

    if len(times.shape) != 1:
        raise ValueError("Time array must be one-dimensional.")
    if temps.shape[0] != times.shape[0]:
        raise ValueError("Times and temperatures arrays must have matching first dimension length.")
    for i in range(temps.shape[1]):
        max_mask = (temps[1:-1, i] > temps[:-2, i]) & (temps[1:-1, i] > temps[2:, i])
        max_indices = np.where(max_mask)[0]+1
        max_dev = abs(temps[max_indices[:-1]] - temps[max_indices[1:]])