# ###################
# Ian Janes
# Prof. Don Lipkin
# Conductive System 1D class file
# ###################

import numpy as np
from numpy.typing import NDArray

class ConductiveSystem1D:

    """
    Class for 1D heat conduction system. Given diffusivity, conductivity, emissivity, heat transfer coefficients, length, and temperatures.
    """

    def __init__(
            self,
            peri: float,
            area: float,
            cphc: float,
            dens: float,
            diff: float,
            cond: float,
            emis: NDArray[np.float64],
            htcs: tuple[float, float],
            length: float,
        ):

        """
        :param float peri: The perimeter of the system [m].
        :param float area: The cross-sectional area of the system [m^2].
        :param float cphc: The constant-pressure heat capacity of the material [J/kg/K].
        :param float dens: The density of the material [kg/m^3].
        :param float diff: The thermal diffusivity of the material [m^2/s].
        :param float cond: The thermal conductivity of the material [W/m/K].
        :param NDArray[np.float64] emis: The emissivities of the material at and above corresponding temperatures. The first column represents the temperatures [K] and the second column represents the emissivities [dimensionless]. Default emissivity for low temperatures is 0.05.
        :param tuple[float, float] htcs: The heat transfer coefficients at the boundaries [W/m^2/K].
        :param float length: The length of the system [m].
        """

        self.perimeter = peri
        self.area = area
        self.heat_capacity = cphc
        self.density = dens
        self.diffusivity = diff
        self.conductivity = cond
        self.emissivities = emis
        self.heat_transfer_coefs = htcs
        self.length = length


    ########################################
    # Getters and Setters
    ########################################


    @property
    def perimeter(self) -> float:

        """
        :return: The perimeter of the system [m].
        """

        return self._peri
    

    @perimeter.setter
    def perimeter(self, peri: float):

        if peri <= 0:
            raise ValueError("Perimeter must be positive.")
        self._peri = peri


    @property
    def area(self) -> float:

        """
        :return: The cross-sectional area of the system [m^2].
        """

        return self._area


    @area.setter
    def area(self, area: float):

        if area <= 0:
            raise ValueError("Cross-sectional area must be positive.")
        self._area = area


    @property
    def heat_capacity(self) -> float:

        """
        :return: The constant-pressure heat capacity of the material [J/kg/K].
        """

        return self._cphc
    

    @heat_capacity.setter
    def heat_capacity(self, cphc: float):

        if cphc <= 0:
            raise ValueError("Heat capacity must be positive.")
        self._cphc = cphc


    @property
    def density(self) -> float:

        """
        :return: The density of the material [kg/m^3].
        """

        return self._dens
    

    @density.setter
    def density(self, dens: float):

        if dens <= 0:
            raise ValueError("Density must be positive.")
        self._dens = dens


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
        if not emis[:, 0].all() > 0:
            raise ValueError("Each temperature must be positive.")
        if not (0 <= emis[:, 1].all() <= 1):
            raise ValueError("Each emissivity must be between 0 and 1 inclusive.")
        self._emis_time_points = emis[:, 0]
        self._emis = emis


    def get_emissivity_at_temp(self, temp: float) -> float:

        """
        Returns the emissivity of the material at a given temperature [K].

        :param float temperature: The temperature at which to get the emissivity [K].
        :return: The emissivity at that temperature.
        """

        idx = np.searchsorted(self._emis_time_points, temp, side="right") - 1
        if idx >= 0:
            return self._emis[idx, 1]
        return 0.05 # Default emissivity


    @property
    def heat_transfer_coefs(self) -> tuple[float, float]:

        """
        :return: The heat transfer coefficients at the boundaries [W/m^2/K].
        """

        return self._htcs


    @heat_transfer_coefs.setter
    def heat_transfer_coefs(self, htcs: tuple[float, float]):

        for h in htcs:
            if h < 0:
                raise ValueError("Each heat transfer coefficient must be non-negative.")
        self._htcs = htcs


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