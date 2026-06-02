# ###################
# Ian Janes
# Prof. Don Lipkin
# Conductive System class file
# ###################

SIGMA = 5.670374419e-8 # Stefan-Boltzmann constant [W/m^2/K^4]

class ConductiveSystem1D:

    """
    Class for 1D heat conduction system. Given diffusivity, conductivity, emissivity, heat transfer coefficients, length, and temperatures, calculates how the temperature distribution across the system evolves over time.
    """

    def __init__(
            self,
            diff: float,
            cond: float,
            emis: tuple[float, float],
            tran: tuple[float, float],
            length: float,
            temp: tuple[float, float, float, float],
            num_points: int = 100,
            max_runtime: float = 100.0,
            diff_num: float = 0.5,
            convergence_tol: float = 1e-6
        ):

        """
        :param float diff: The thermal diffusivity of the material [m^2/s].
        :param float cond: The thermal conductivity of the material [W/m/K].
        :param tuple[float, float] emis: The emissivities of the material at the boundaries [dimensionless].
        :param tuple[float, float] tran: The heat transfer coefficients at the boundaries [W/m^2/K].
        :param float length: The length of the system [m].
        :param tuple[float, float, float, float] temp: The surrounding and initial boundary temperatures (surr1, bound1, bound2, surr2) [K].
        :param int num_points: The number of spatial points to discretize the system into for numerical calculations.
        :param float diff_num: The diffusion number of the system [dimensionless]. Relates to time step and numerical stability. Less than or equal to 0.5.
        :param float max_runtime: The maximum runtime for the simulation [s]. Used to determine how long to run the simulation before stopping if it hasn't converged yet.
        :param float convergence_tol: The tolerance for convergence of the simulation [dimensionless]. Used to determine when the simulation has converged.
        """

        self._set_diffusivity(diff)
        self._set_conductivity(cond)
        self._set_emissivity(emis)
        self._set_heat_transfer_coefs(tran)
        self._set_length(length)
        self._set_temps(temp)
        self._set_spatial_res(num_points)
        self._set_diffusion_number(diff_num)
        self._set_max_runtime(max_runtime)
        self._set_convergence_tol(convergence_tol)
        self._set_x_step()
        self._set_t_step()
    

    def _set_diffusivity(self, diff: float):

        if diff < 0:
            raise ValueError("Diffusivity must be non-negative.")
        self._diffusivity = diff


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

    
    def _set_temps(self, temp: tuple[float, float, float, float]):

        if len(temp) != 4:
            raise ValueError("Temperature input must be a tuple of length 4.")
        for t in temp:
            if t < 0:
                raise ValueError("Each temperature must be non-negative.")
        self._temperatures = temp


    def _set_spatial_res(self, points: int):

        if not isinstance(points, int):
            raise TypeError("Number of spatial points must be an integer.")
        if points <= 0:
            raise ValueError("Number of spatial points must be positive.")
        self._spatial_res = points


    def _set_diffusion_number(self, diff_num: float):

        if diff_num <= 0 or diff_num > 0.5:
            raise ValueError("Diffusion number must be greater than 0 and less than or equal to 0.5 for numerical stability.")
        self._diffusion_number = diff_num


    def _set_max_runtime(self, max_runtime: float):

        if max_runtime <= 0:
            raise ValueError("Maximum runtime must be greater than 0.")
        self._max_runtime = max_runtime


    def _set_convergence_tol(self, convergence_tol: float):

        if convergence_tol <= 0:
            raise ValueError("Convergence tolerance must be greater than 0.")
        self._convergence_tol = convergence_tol


    def _set_x_step(self):

        self._x_step = self._length / (self._spatial_res - 1)


    def _set_t_step(self):

        self._t_step = self._diffusion_number * (self._x_step ** 2) / self._diffusivity

test = ConductiveSystem1D(1.0, 1.0, (0.5, 0.5), (10.0, 10.0), 1.0, (300.0, 300.0, 300.0, 300.0))
