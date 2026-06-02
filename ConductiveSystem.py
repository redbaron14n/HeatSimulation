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
            tran: tuple[float, float],
            length: float,
            temp: tuple[float, float, float, float],
        ):

        """
        :param float diff: The thermal diffusivity of the material [m^2/s].
        :param float cond: The thermal conductivity of the material [W/m/K].
        :param tuple[float, float] emis: The emissivities of the material at the boundaries [dimensionless].
        :param tuple[float, float] tran: The heat transfer coefficients at the boundaries [W/m^2/K].
        :param float length: The length of the system [m].
        :param tuple[float, float, float, float] temp: The surrounding and initial boundary temperatures (surr1, bound1, bound2, surr2) [K].
        """

        self._set_diffusivity(diff)
        self._set_conductivity(cond)
        self._set_emissivity(emis)
        self._set_heat_transfer_coefs(tran)
        self._set_length(length)
        self._set_temps(temp)
    

    def _set_diffusivity(self, diff: float):

        if diff < 0:
            raise ValueError("Diffusivity must be non-negative.")
        self._diffusivity = diff


    @property
    def diffusivity(self) -> float:

        """
        :return: The thermal diffusivity [m^2/s].
        """

        return self._diffusivity


    def _set_conductivity(self, cond: float):

        if cond < 0:
            raise ValueError("Conductivity must be non-negative.")
        self._conductivity = cond


    def _set_emissivity(self, emis: tuple[float, float]):

        if len(emis) != 2:
            raise ValueError("Emissivity input must be a tuple of length 2.")
        for e in emis:
            if e < 0 or e > 1:
                raise ValueError("Each emissivity must be between 0 and 1 inclusive.")
        self._emissivities = emis


    def _set_heat_transfer_coefs(self, tran: tuple[float, float]):

        if len(tran) != 2:
            raise ValueError("Heat transfer coefficient input must be a tuple of length 2.")
        for h in tran:
            if h < 0:
                raise ValueError("Each heat transfer coefficient must be non-negative.")
        self._heat_transfer_coefs = tran


    def _set_length(self, length: float):

        if length <= 0:
            raise ValueError("Length must be positive.")
        self._length = length


    @property
    def length(self) -> float:

        """
        :return: The length of the system [m].
        """

        return self._length

    
    def _set_temps(self, temp: tuple[float, float, float, float]):

        if len(temp) != 4:
            raise ValueError("Temperature input must be a tuple of length 4.")
        for t in temp:
            if t < 0:
                raise ValueError("Each temperature must be non-negative.")
        self._temperatures = temp