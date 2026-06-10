# ###################
# Ian Janes
# Prof. Don Lipkin
# Conductive System 1D class file
# ###################

class ConductiveSystem1D:

    """
    Class for 1D heat conduction system. Given diffusivity, conductivity, emissivity, heat transfer coefficients, length, and temperatures.
    """

    def __init__(
            self,
            diff: float,
            cond: float,
            emis: tuple[float, float],
            htcs: tuple[float, float],
            length: float,
        ):

        """
        :param float diff: The thermal diffusivity of the material [m^2/s].
        :param float cond: The thermal conductivity of the material [W/m/K].
        :param tuple[float, float] emis: The emissivities of the material at the boundaries [dimensionless].
        :param tuple[float, float] htcs: The heat transfer coefficients at the boundaries [W/m^2/K].
        :param float length: The length of the system [m].
        """

        self._diff = diff
        self._cond = cond
        self._emis = emis
        self._htcs = htcs
        self._length = length


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
    def emissivities(self) -> tuple[float, float]:

        """
        :return: The emissivities at the boundaries [dimensionless].
        """

        return self._emis


    @emissivities.setter
    def emissivities(self, emis: tuple[float, float]):

        for e in emis:
            if e < 0 or e > 1:
                raise ValueError("Each emissivity must be between 0 and 1 inclusive.")
        self._emis = emis


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