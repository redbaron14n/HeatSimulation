# ###################
# Ian Janes
# Prof. Don Lipkin
# Finite Difference Solver class file
# ###################

from ConductiveSystem import ConductiveSystem1D
from numpy import abs, array, complexfloating, float64, floating, full, linspace, roots, searchsorted
from numpy.typing import NDArray

BOLTZ = 5.670374419e-8 # Stefan-Boltzmann constant [W/m^2/K^4]

class FiniteDiffSolver1D:

    def __init__(
            self,
            system: ConductiveSystem1D,
            initial_temps: NDArray[float64] | None = None,
            torch_fluxs: NDArray[float64] = array([[0.0, 0.0, 0.0]]),
            gas_temps: tuple[float, float] = (298.15, 298.15),
            ambient_temp: float = 298.15,
            spatial_res: int = 25,
            min_sim_time: float = 0.0,
            max_sim_time: float = 100.0,
            diff_num: float = 0.5,
            conv_tol: float = 1e-6
        ):

        """
        :param ConductiveSystem1D system: The conductive system to be solved.
        :param NDArray[float64] | None initial_temps: The initial temperatures for the finite difference grid [K]. Default is None, which will initialize based on the system's ambient temperature.
        :param NDArray[float64] torch_fluxs: The heat fluxes from the torch at the boundaries [W/m^2] where positive is into the material. The first column is time and the second and third columns are the new constant heat fluxes at the left and right boundaries introduced at that time. Default is no torch flux.
        :param tuple[float, float] gas_temps: The temperatures of the gas at the boundaries [K].
        :param float ambient_temp: The ambient temperature for the simulation [K].
        :param int spatial_res: The number of spatial points to use in the finite difference grid.
        :param float min_sim_time: The minimum simulation time to run the solver for [s].
        :param float max_sim_time: The maximum simulation time to run the solver for [s].
        :param float diff_num: The diffusion number (alpha * dt / dx^2) to use for numerical stability.
        :param float conv_tol: The tolerance for convergence of the iterative solver.
        """

        self.system = system
        self._set_initial_temps(initial_temps)
        self.torch_heat_fluxes = torch_fluxs
        self.gas_temperatures = gas_temps
        self.ambient_temperature = ambient_temp
        self.spatial_resolution = spatial_res
        self.min_simulation_time = min_sim_time
        self.max_simulation_time = max_sim_time
        self.diffusion_number = diff_num
        self.convergence_tol = conv_tol


    @property
    def system(self) -> ConductiveSystem1D:

        """
        :return: The conductive system under analysis.
        """

        return self._system
    

    @system.setter
    def system(self, system: ConductiveSystem1D):

        self._system = system
        self._update_x_step()
    

    @property
    def torch_heat_fluxes(self) -> NDArray[float64]:

        """
        :return: The heat fluxes from the torch at the boundaries [W/m^2].
        """

        return self._torch_fluxs
    

    @torch_heat_fluxes.setter
    def torch_heat_fluxes(self, fluxs: NDArray[float64]):

        if fluxs.shape[1] != 3:
            raise ValueError("Torch heat fluxes must be a 2D array with shape (N, 3).")
        if not all(fluxs[:, 0] >= 0):
            raise ValueError("Torch heat flux times must be non-negative.")
        self._torch_fluxs = fluxs
        self._flux_time_points: NDArray[float64] = fluxs[:, 0]


    def _get_current_torch_flux(self, time: float) -> tuple[float, float]:

        idx = searchsorted(self._flux_time_points, time, side='right') - 1
        if idx >= 0:
            return self._torch_fluxs[idx, 1], self._torch_fluxs[idx, 2]
        return 0.0, 0.0 # No torch flux before the first time point      
    

    @property
    def gas_temperatures(self) -> tuple[float, float]:

        """
        :return: The temperatures of the gas at the boundaries [K].
        """

        return self._gas_temps
    

    @gas_temperatures.setter
    def gas_temperatures(self, temps: tuple[float, float]):

        if len(temps) != 2:
            raise ValueError("Gas temperatures must be a tuple of two values (T_gas1, T_gas2).")
        if temps[0] < 0 or temps[1] < 0:
            raise ValueError("Gas temperatures must be non-negative.")
        self._gas_temps = temps
    

    @property
    def ambient_temperature(self) -> float:

        """
        :return: The ambient temperature for the simulation [K].
        """

        return self._ambient_temp
    

    @ambient_temperature.setter
    def ambient_temperature(self, temp: float):

        if temp < 0:
            raise ValueError("Ambient temperature must be non-negative.")
        self._ambient_temp = temp

    
    @property
    def spatial_resolution(self) -> int:

        """
        :return: The number of spatial points in the finite difference grid.
        """

        return self._x_res
    

    @spatial_resolution.setter
    def spatial_resolution(self, points: int):

        if points < 2:
            raise ValueError("Number of spatial points must be at least 2.")
        self._x_res = points
        self._update_x_step() # Update x_step based on new spatial resolution


    def _set_initial_temps(self, initial_temps: NDArray[float64] | None):

        if initial_temps is not None:
            if len(initial_temps) != self._x_res:
                raise ValueError("Length of initial temperatures array must match spatial resolution.")
            self._init_temps = initial_temps
        else:
            self._init_temps: NDArray[float64] = full(self._x_res, self._ambient_temp) # Default to uniform ambient temperature


    @property
    def min_simulation_time(self) -> float:

        """
        :return: The minimum simulation time [s].
        """

        return self._min_time
    

    @min_simulation_time.setter
    def min_simulation_time(self, min_time: float):

        if min_time < 0.0:
            raise ValueError("Minimum simulation time must be non-negative.")
        self._min_time = min_time


    @property
    def max_simulation_time(self) -> float:

        """
        :return: The maximum simulation time [s].
        """

        return self._max_time
    

    @max_simulation_time.setter
    def max_simulation_time(self, max_time: float):

        if max_time <= 0.0:
            raise ValueError("Maximum simulation time must be greater than 0.")
        elif max_time <= self._min_time:
            raise ValueError("Maximum simulation time must be greater than minimum simulation time.")
        self._max_time = max_time
        self._update_t_step() # Update t_step based on new max_time


    @property
    def diffusion_number(self) -> float:

        """
        :return: The diffusion number (alpha * dt / dx^2) used for numerical stability.
        """

        return self._diff_num
    

    @diffusion_number.setter
    def diffusion_number(self, diff_num: float):

        if diff_num <= 0 or diff_num > 0.5:
            raise ValueError("Diffusion number must be greater than 0 and less than or equal to 0.5 for numerical stability.")
        self._diff_num = diff_num
        self._update_t_step() # Update t_step based on new diffusion number


    @property
    def convergence_tol(self) -> float:

        """
        :return: The tolerance for convergence of the iterative solver.
        """

        return self._conv_tol
    

    @convergence_tol.setter
    def convergence_tol(self, conv_tol: float):

        if conv_tol <= 0:
            raise ValueError("Convergence tolerance must be greater than 0.")
        self._conv_tol = conv_tol


    def _update_x_step(self):

        if not hasattr(self, '_x_res'):
            return # Allows initilization to complete before calculating x_step
        self._x_step = self._system.length / (self._x_res - 1)
        self._update_t_step() # Update t_step based on new x_step


    def _update_t_step(self):

        if not hasattr(self, '_diff_num'):
            return
        self._t_step = self._diff_num * (self._x_step ** 2) / self._system.diffusivity
        self._update_tick_count()


    def _update_tick_count(self):

        self._tick_count = int(self._max_time / self._t_step) + 1
    

    def _init_x_grid(self) -> NDArray[float64]:

        """
        Initialize the spatial grid for the finite difference solver.

        :return: A 1D array of spatial points along the length of the system.
        """

        return linspace(0, self._system.length, self._x_res)
    

    def _root_selector(self, roots_T: NDArray[complexfloating] | NDArray[floating]) -> float:

        """
        Select the appropriate root from the quartic equation based on physical constraints (real and positive).

        :param NDArray[complex] roots_T: The array of roots from the quartic equation.
        :return: The selected root that represents the boundary temperature [K].
        """

        real_roots = roots_T[abs(roots_T.imag) < 1e-10].real
        positive_real_roots = real_roots[real_roots > 0]
        if len(positive_real_roots) == 0:
            raise ValueError("No valid positive real roots found for boundary temperature.")
        if len(positive_real_roots) > 1:
            raise ValueError("Multiple positive real roots found for boundary temperature, unable to select unique solution.")
        return positive_real_roots[0] # Should only be one valid root based on physical constraints


    def _solve_boundary_temp(self, k: float, epsilon: float, h: float, T_inside: float, T_gas: float) -> float:

        """
        Solve for the boundary temperature based on the heat transfer coefficients and emissivity.

        :param float k: The thermal conductivity of the material [W/m/K].
        :param float epsilon: The emissivity of the material at the boundary [dimensionless].
        :param float h: The heat transfer coefficient at the boundary [W/m^2/K].
        :param float T_inside: The temperature just inside the boundary [K].
        :param float T_gas: The temperature of the gas adjacent to the boundary [K].
        :return T_bound: The calculated boundary temperature [K].
        """

        T_ambient = self._ambient_temp
        dz = self._x_step
        a4 = epsilon * BOLTZ * dz
        a1 = h * dz + k
        a0 = -k * T_inside - h * dz * T_gas - a4 * (T_ambient ** 4)
        coeffs = [a4, 0, 0, a1, a0] # Coefficients for the quartic equation a4*T^4 + a1*T + a0 = 0
        T_bound = self._root_selector(roots(coeffs))
        return T_bound


    def _iterate_internal_temps(self, T_new: NDArray[float64], temps: NDArray[float64]) -> NDArray[float64]:

        """
        Perform one iteration of the finite difference solver to update the internal temperatures based on the diffusion equation.

        :param NDArray[float64] temps: The current temperature array at all spatial points.
        :return: A new temperature array with updated internal temperatures.
        """

        for i in range(1, self._x_res - 1):
            T_new[i] = temps[i] + self._diff_num * (temps[i+1] - 2*temps[i] + temps[i-1])
        return T_new        


    def _check_convergence(self, T_new: NDArray[float64], T_old: NDArray[float64], tick: int) -> bool:

        """
        Check for convergence of the iterative solver based on the sum of squared differences between new and old temperature arrays.

        :param NDArray[float64] T_new: The new temperature array after an iteration.
        :param NDArray[float64] T_old: The old temperature array before the iteration.
        :return: True if converged, False otherwise.
        """

        sum_square = ((T_new - T_old) ** 2).sum()
        if tick % 1000 == 0:
                print(f"Tick {tick}: [{T_new[0]:.3f}, {T_new[1]:.3f}, {T_new[2]:.3f}, ..., {T_new[-2]:.3f}, {T_new[-1]:.3f}], Sum Square: {sum_square:.2e}")
        return sum_square <= self._conv_tol


    def _simulation_summary(self, tick: int, converged: bool):

        """
        Print a summary of the simulation results after completion.

        :param int tick: The number of ticks (iterations) taken to complete the simulation.
        :param bool converged: Whether the simulation converged or reached maximum time without convergence.
        """

        if converged:
            print(f"Simulation converged in {tick} ticks.")
        else:
            print(f"Simulation reached maximum time without convergence.")


    def run_simulation(self):

        """
        Run the finite difference simulation for the given system and solver parameters.
        """

        temps = self._init_temps
        x_grid = self._init_x_grid()
        k = self._system.conductivity
        epsilon1, epsilon2 = self._system.emissivities
        h1, h2 = self._system.heat_transfer_coefs
        T_gas1, T_gas2 = self._gas_temps[0], self._gas_temps[1]
        converged = False
        tick = 0
        while (not converged) and (tick < self._tick_count):
            tick += 1
            T_new = temps.copy()
            T_inside1, T_inside2 = temps[1], temps[-2]
            T_new[0] = self._solve_boundary_temp(k, epsilon1, h1, T_inside1, T_gas1)
            T_new[-1] = self._solve_boundary_temp(k, epsilon2, h2, T_inside2, T_gas2)
            T_new = self._iterate_internal_temps(T_new, temps)
            converged = self._check_convergence(T_new, temps, tick)
            temps = T_new
        self._simulation_summary(tick, converged)
        self._final_temps = temps

test_system = ConductiveSystem1D(1.58e-4, 40.0, (0.5, 0.5), (316.227766, 100.0), 0.0035)

test = FiniteDiffSolver1D(test_system, gas_temps=(2500.0, 300.0), ambient_temp=300.0, spatial_res=25, max_sim_time=100.0, diff_num=0.5, conv_tol=1e-6)