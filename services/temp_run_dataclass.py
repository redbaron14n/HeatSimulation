# ###################
# Ian Janes
# Prof. Don Lipkin
# Data graphics service file
# ###################

from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


@dataclass(slots=True)
class TemperatureRun:

    times: NDArray[np.float64]
    temperatures: NDArray[np.float64]
    positions: NDArray[np.float64]

    @property
    def min_temp(self) -> float:

        return float(np.min(self.temperatures))
    

    @property
    def max_temp(self) -> float:

        return float(np.max(self.temperatures))
    

    @property
    def n_times(self) -> int:

        return self.temperatures.shape[0]
    

    @property
    def x_res(self) -> int:

        return self.temperatures.shape[1]
    

    def snapshot(self, index: int) -> NDArray[np.float64]:

        """
        :return: The temperature distribution at a given time index.
        """

        return self.temperatures[index]
    

    def point_history(self, x_indx: int, r_indx: int | None = None) -> NDArray[np.float64]:

        """
        :return: The temperature evolution over time of a given spatial index.
        """

        if r_indx is not None:
            return self.temperatures[: r_indx, x_indx]
        return self.temperatures[:, x_indx]