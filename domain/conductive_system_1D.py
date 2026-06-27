# ###################
# Ian Janes
# Prof. Don Lipkin
# Conductive System 1D class file
# ###################

import numpy as np
from numpy.typing import NDArray

class ConductiveSystem1D:

    """
    Class for 1D heat conduction system. Given diffusivity, conductivity, emissivity, and length.
    """

    def __init__(
            self,
            diff: float,
            cond: float,
            emis: NDArray[np.float64],
            length: float,
        ):

        """
        :param float diff: The thermal diffusivity of the material [m^2/s].
        :param float cond: The thermal conductivity of the material [W/m/K].
        :param NDArray[np.float64] emis: The emissivities of the material at and above corresponding temperatures. The first column represents the temperatures [K] and the second column represents the emissivities [dimensionless].
        :param float length: The length of the system [m].
        """

        self.diffusivity = diff
        self.conductivity = cond
        self.emissivities = emis
        self.length = length


    ########################################
    # Getters and Setters
    ########################################


    @property
    def diffusivity(self) -> float:

        """
        :return: The thermal diffusivity [m^2/s].
        """

        return self._diff
    

    @diffusivity.setter
    def diffusivity(self, diff: float):

        if diff < 0:
            raise ValueError("Diffusivity must be non-negative.")
        self._diff = diff


    @property
    def conductivity(self) -> float:

        """
        :return: The thermal conductivity [W/m/K].
        """

        return self._cond


    @conductivity.setter
    def conductivity(self, cond: float):

        if cond < 0:
            raise ValueError("Conductivity must be non-negative.")
        self._cond = cond


    @property
    def emissivities(self) -> NDArray[np.float64]:

        """
        :return: The emissivities at the boundaries [dimensionless].
        """

        return self._emis


    @emissivities.setter
    def emissivities(self, emis: NDArray[np.float64]):

        if emis.shape[1] != 2:
            raise ValueError("Emissivities must be a 2D array with 2 columns.")
        if not np.all(emis[:, 0] > 0):
            raise ValueError("Each temperature must be positive.")
        if not (np.all(emis[:, 1] >= 0) and np.all(emis[:, 1] <= 1)):
            raise ValueError("Each emissivity must be between 0 and 1 inclusive.")
        self._emis = emis


    @property
    def length(self) -> float:

        """
        :return: The length of the system [m].
        """

        return self._length


    @length.setter
    def length(self, length: float):

        if length <= 0:
            raise ValueError("Length must be positive.")
        self._length = length