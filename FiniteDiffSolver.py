# ###################
# Ian Janes
# Prof. Don Lipkin
# Finite Difference Solver class file
# ###################

import numpy as np
from ConductiveSystem import ConductiveSystem1D, ConductiveSystemAxial
from DataHandling import load_init_temps, save_procedure
from numpy.typing import NDArray
from pathlib import Path
from typing import NamedTuple

BOLTZ = 5.670374419e-8 # Stefan-Boltzmann constant [W/m^2/K^4]
MAX_TEMP = 6000 # Maximum temperature in the simulation [K]

class LookupTables(NamedTuple):

    gas_temp_lookup: tuple[NDArray[np.float64], NDArray[np.float64]]
    htcs_lookup: tuple[NDArray[np.float64], NDArray[np.float64]]
    emis_lookup: NDArray[np.float64]

class FiniteDiffSolver1D:

    def __init__(
            self,
            system: ConductiveSystem1D,
            initial_temps: NDArray[np.float64] | None = None,
            gas_temps: NDArray[np.float64] = np.array([[0.0, 298.15, 298.15]]),
            htcs: NDArray[np.float64] = np.array([[0.0, 100., 100.]], dtype=np.float64),
            ambient_temp: float = 298.15,
            spatial_res: int = 25,
            min_sim_time: float = 0.0,
            max_sim_time: float = 100.0,
            diff_num: float = 0.1
        ):

        """
        :param system: The conductive system to be solved.
        :param initial_temps: The initial temperatures for the finite difference grid [K]. Default is None, which will initialize based on the system's ambient temperature.
        :param gas_temps: The temperatures of the gas at the boundaries [K]. The first column is time and the second and third columns are the new constant gas temperatures introduced at that time. Default is room temperature (298.15 K) on both sides.
        :param htcs: The heat transfer coefficients [W/m^2/K] at the boundaries. The first column is time and the second and third columns are the new constant coefficients introduced at that time. Default is 100 on both sides.
        :param ambient_temp: The ambient temperature for the simulation [K].
        :param spatial_res: The number of spatial points to use in the finite difference grid.
        :param min_sim_time: The minimum simulation time to run the solver for [s].
        :param max_sim_time: The maximum simulation time to run the solver for [s].
        :param diff_num: The diffusion number (alpha * dt / dx^2) to use for numerical stability.
        """

        self.system = system
        self.gas_temperatures = gas_temps
        self.heat_transfer_coefs = htcs
        self.ambient_temperature = ambient_temp
        self.spatial_resolution = spatial_res
        self.initial_temperatures = initial_temps
        self.min_simulation_time = min_sim_time
        self.max_simulation_time = max_sim_time
        self.diffusion_number = diff_num


    ########################################
    # Getters and Setters
    ########################################


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
    def gas_temperatures(self) -> NDArray[np.float64]:

        """
        :return: The temperatures of the gas at the boundaries [K].
        """

        return self._gas_temps
    

    @gas_temperatures.setter
    def gas_temperatures(self, temps: NDArray[np.float64]):

        if temps.shape[1] != 3:
            raise ValueError("Gas temperatures must be a 2D array with shape (N, 3).")
        if not np.all(temps[:, 0] >= 0):
            raise ValueError("Gas temperature times must be non-negative.")
        if not np.all(temps[:, 1:] > 0):
            raise ValueError("Gas temperatures must be positive.")
        self._gas_temps = temps


    @property
    def heat_transfer_coefs(self) -> NDArray[np.float64]:

        """
        :return: The heat transfer coefficients [W/m^2/K] at the boundaries. The first column is time and the second and third columns are the new constant coefficients introduced at that time.
        """

        return self._htcs
    

    @heat_transfer_coefs.setter
    def heat_transfer_coefs(self, htcs: NDArray[np.float64]):

        if htcs.shape[1] != 3:
            raise ValueError("Heat transfer coefficient input must be a 2D array with shape (N, 3).")
        if not np.all(htcs[:, 0] >= 0.):
            raise ValueError("All given times must be non-negative.")
        if not np.all(htcs[:, 1:] >= 0.):
            raise ValueError("All given heat transfer coefficients must be non-negative.")
        self._htcs = htcs


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


    @property
    def initial_temperatures(self) -> NDArray[np.float64]:

        """
        :return: The initial temperatures for the finite difference grid [K].
        """

        return self._init_temps
    

    @initial_temperatures.setter
    def initial_temperatures(self, initial_temps: NDArray[np.float64] | None):

        if initial_temps is None:
            self._init_temps: NDArray[np.float64] = np.full(self._x_res, self._ambient_temp) # Default to uniform ambient temperature
            return
        if len(initial_temps) != self._x_res:
            raise ValueError("Length of initial temperatures array must match spatial resolution.")
        self._init_temps = initial_temps


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


    ########################################
    # Private Methods
    ########################################


    def _validate_init_temps(self): # Call this before running the simulation to catch if x_res was changed without updating initial temperatures

        if len(self._init_temps) != self._x_res:
            raise ValueError("Length of initial temperatures array must match spatial resolution.")
        

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

        if not hasattr(self, '_t_step'):
            return
        self._tick_count = int(self._max_time / self._t_step) + 1
    

    def _root_selector(self, roots_T: NDArray[np.complexfloating] | NDArray[np.floating]) -> float:

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


    def _solve_boundary_temp(self, epsilon: float, h: float, T_inside: float, T_gas: float) -> float:

        """
        Solve for the boundary temperature based on the heat transfer coefficients and emissivity.

        :param float epsilon: The emissivity of the material at the boundary [dimensionless].
        :param float h: The heat transfer coefficient at the boundary [W/m^2/K].
        :param float T_inside: The temperature just inside the boundary [K].
        :param float T_gas: The temperature of the gas adjacent to the boundary [K].
        :return T_bound: The calculated boundary temperature [K].
        """

        k = self._system.conductivity
        a4 = epsilon * BOLTZ * self._x_step
        a1 = h * self._x_step + k
        a0 = -k * T_inside - h * self._x_step * T_gas - a4 * (self._ambient_temp ** 4)
        coeffs = [a4, 0., 0., a1, a0] # Coefficients for the quartic equation a4*T^4 + a1*T + a0 = 0
        T_bound = self._root_selector(np.roots(coeffs))
        return T_bound


    def _iterate_internal_temps(self, T_new: NDArray[np.float64], temps: NDArray[np.float64]) -> NDArray[np.float64]:

        """
        Perform one iteration of the finite difference solver to update the internal temperatures based on the diffusion equation.

        :param T_new: The array to store the new temperatures after the iteration.
        :param temps: The current temperature array at all spatial points.

        :return: A new temperature array with updated internal temperatures.
        """

        internal_t = temps[1:-1]
        left = temps[:-2]
        right = temps[2:]
        cond_terms = self._diff_num * (right - 2 * internal_t + left)
        T_new[1: -1] = internal_t + cond_terms
        return T_new


    def _check_convergence(self, T_new: NDArray[np.float64], T_old: NDArray[np.float64], tick: int, print_every: int, conv_tol: float) -> bool:

        """
        Check for convergence of the iterative solver based on the sum of squared differences between new and old temperature arrays.

        :param T_new: The new temperature array after an iteration.
        :param T_old: The old temperature array before the iteration.
        :param tick: The current iteration number.
        :param print_every: The interval at which to print the simulation progress. 0 means no printing.
        :return: True if converged, False otherwise.
        """

        sum_square = ((T_new - T_old) ** 2).sum()
        if print_every > 0 and tick % print_every == 0:
            print(f"Tick {tick}: [{T_new[0]:.3f}, {T_new[1]:.3f}, {T_new[2]:.3f}, ..., {T_new[-2]:.3f}, {T_new[-1]:.3f}], Sum Square: {sum_square:.2e}")
        finished = (sum_square <= conv_tol) and (tick * self._t_step >= self._min_time)
        return finished


    def _simulation_summary(self, tick: int, saved: int, converged: bool):

        """
        Print a summary of the simulation results after completion.

        :param tick: The number of ticks (iterations) taken to complete the simulation.
        :param saved: The number of saved data points.
        :param converged: Whether the simulation converged or reached maximum time without convergence.
        """

        if converged:
            print(f"Simulation converged in {tick} ticks with {saved} saved data points.")
        else:
            print(f"Simulation reached maximum time without convergence with {saved} saved data points.")


    def _build_lookup_tables(self, times: NDArray[np.float64]) -> LookupTables:

        """
        Build lookup tables for the simulation data. Drastically reduces the computational cost of interpolation during the simulation.

        :param times: The time points for which to build the lookup tables.
        :return: A tuple of the lookup tables  gas temperature, heat transfer coefs, and emissivity.
        """

        gas_data = self._gas_temps
        htcs_data = self._htcs
        emis_data = self._system.emissivities

        gas_temp0_lookup = np.interp(times, gas_data[:, 0], gas_data[:, 1], left=self._ambient_temp)
        gas_temp1_lookup = np.interp(times, gas_data[:, 0], gas_data[:, 2], left=self._ambient_temp)
        gasses_lookup = (gas_temp0_lookup, gas_temp1_lookup)

        htcs0_lookup = np.interp(times, htcs_data[:, 0], htcs_data[:, 1], left=100.)
        htcs1_lookup = np.interp(times, htcs_data[:, 0], htcs_data[:, 2], left=100.)
        htcs_lookup = (htcs0_lookup, htcs1_lookup)

        emis_lookup = np.interp(np.arange(0, MAX_TEMP + 1), emis_data[:, 0], emis_data[:, 1])

        return LookupTables(gasses_lookup, htcs_lookup, emis_lookup)


    ########################################
    # Public Methods
    ########################################


    def run_simulation(self, conv_tol: float = 1e-6, print_every: int = 1000, save_tol: float = 1e-3):

        """
        Run the finite difference simulation for the given system and solver parameters.

        :param conv_tol: The convergence tolerance of the simulation. The simulation will declare steady-state if the sum of the squares of the differences between iterations falls to or below this tolerance. Default is 1e-6.
        :param print_every: The interval at which to print the simulation progress. Default is 1000. 0 means no printing.
        :param save_tol: How large the sum square of differences between latest and last saved temperature distributions must be before it is saved. Default is 1e-3.
        """

        self._validate_init_temps() # Ensure initial temperatures are valid before starting simulation
        temps = self._init_temps
        last_saved = np.hstack((np.array([0.0]), temps))
        self._raw_temp_list: list[NDArray[np.float64]] = [last_saved]
        times = np.arange(0, self._max_time + self._t_step, self._t_step, dtype=np.float64)
        gasses, htcs, emis_lookup = self._build_lookup_tables(times)
        converged = False
        tick = 0
        saved = 0
        while (not converged) and (tick < self._tick_count):
            tick += 1
            T_new = temps.copy()
            T_inside0, T_inside1 = temps[1], temps[-2] # Finds the previous temperatures just inside the boundaries
            time = times[tick]
            gas_temp0, gas_temp1 = gasses[0][tick], gasses[1][tick]
            htc0, htc1 = htcs[0][tick], htcs[1][tick]
            emis = emis_lookup[temps.astype(int)]
            emis0, emis1 = emis[0], emis[-1]
            T_new[0] = self._solve_boundary_temp(emis0, htc0, T_inside0, gas_temp0)
            T_new[-1] = self._solve_boundary_temp(emis1, htc1, T_inside1, gas_temp1)
            T_new = self._iterate_internal_temps(T_new, temps)
            converged = self._check_convergence(T_new, temps, tick, print_every, conv_tol)
            if (((T_new - last_saved[1:])**2).sum() >= save_tol) or converged:
                last_saved = np.hstack((np.array([time]), T_new))
                self._raw_temp_list.append(last_saved)
                saved += 1
            temps = T_new
        self._final_temps = temps
        self._raw_temp_map = np.array(self._raw_temp_list)
        self._simulation_summary(tick, saved, converged)
        save_procedure(self._raw_temp_map, self._final_temps)


diff = 1.58e-4
cond = 40.
emis = np.array([[0.001, 0.5]], dtype=np.float64)
length = 0.0035

test_system = ConductiveSystem1D(diff, cond, emis, length)

x_res = 100
initial_temps = load_init_temps(Path("copper_steadystate.csv"))
gas_temps = np.array([[0.0, 300., 300.],
                      [0.0095, 300., 300.],
                      [0.1005, 2500., 300.]], dtype=np.float64)
htcs = np.array([[0., 100., 100.],
                 [0.0095, 100., 100.],
                 [0.1005, 316.227766, 100.]], dtype=np.float64)
ambient_temp = 300.

test_solver = FiniteDiffSolver1D(test_system, initial_temps, gas_temps, htcs, ambient_temp, x_res, min_sim_time=1.0)

test_solver.run_simulation(conv_tol=1e-6, print_every=10000)


class FiniteDiffSolverAxial:

    def __init__(
            self,
            system: ConductiveSystemAxial,
            initial_temps: NDArray[np.float64] | None = None,
            torch_fluxs: NDArray[np.float64] = np.array([[0.0, 0.0, 0.0]]),
            gas_temps: NDArray[np.float64] = np.array([[0.0, 298.15, 298.15]]),
            env_cutoff: float = 0.0,
            ambient_temp: float = 298.15,
            x_res: int = 25,
            r_res: int = 25,
            min_sim_time: float = 0.0,
            max_sim_time: float = 100.0,
            diff_num: float = 0.1,
            conv_tol: float = 1e-6
    ):
        

        """
        :param system: The conductive system to be solved.
        :param initial_temps: The initial temperatures for the finite difference grid [K]. Default is None, which will initialize based on the system's ambient temperature.
        :param torch_fluxs: The heat fluxes from the torch at the boundaries [W/m^2] where positive is into the material. The first column is time and the second and third columns are the new constant heat fluxes at the left and right boundaries introduced at that time. Default is no torch flux.
        :param gas_temps: The temperatures of the gas at the boundaries [K]. The first column is time and the second and third columns are the new constant gas temperatures surrounding the left and right sections introduced at that time. Default is room temperature (298.15 K) on both sides.
        :param env_cutoff: The cutoff distance (as a proportion of the system length) where left side gas temperatures, heat transfer coefficients switch to right side values. Defaults to 0.0, meaning left side values are used for the left face only.
        :param ambient_temp: The ambient temperature for the simulation [K].
        :param x_res: The number of spatial points to use in the finite difference grid along the length of the system.
        :param r_res: The number of spatial points to use in the finite difference grid along the radius of the system.
        :param min_sim_time: The minimum simulation time to run the solver for [s].
        :param max_sim_time: The maximum simulation time to run the solver for [s].
        :param diff_num: The diffusion number to use for numerical stability.
        :param conv_tol: The tolerance for convergence of the iterative solver.
        """

        self.system = system
        self.torch_heat_fluxes = torch_fluxs
        self.gas_temperatures = gas_temps
        self.env_cutoff = env_cutoff
        self.ambient_temperature = ambient_temp
        self.x_resolution = x_res
        self.r_resolution = r_res
        self.initial_temperatures = initial_temps
        self.min_simulation_time = min_sim_time
        self.max_simulation_time = max_sim_time
        self.diffusion_number = diff_num
        self.convergence_tol = conv_tol


    ########################################
    # Getters and Setters
    ########################################


    @property
    def system(self) -> ConductiveSystemAxial:

        """
        :return: The conductive system under analysis.
        """

        return self._system
    

    @system.setter
    def system(self, system: ConductiveSystemAxial):

        self._system = system
        self._update_x_step()
        self._update_r_step()


    @property
    def torch_heat_fluxes(self) -> NDArray[np.float64]:

        """
        :return: The heat fluxes from the torch at the boundaries [W/m^2].
        """

        return self._torch_fluxs
    

    @torch_heat_fluxes.setter
    def torch_heat_fluxes(self, fluxs: NDArray[np.float64]):

        if fluxs.shape[1] != 3:
            raise ValueError("Torch heat fluxes must be a 2D array with shape (N, 3).")
        if not np.all(fluxs[:, 0] >= 0):
            raise ValueError("Torch heat flux times must be non-negative.")
        self._torch_fluxs = fluxs


    @property
    def gas_temperatures(self) -> NDArray[np.float64]:

        """
        :return: The temperatures of the gas at the boundaries [K].
        """

        return self._gas_temps
    

    @gas_temperatures.setter
    def gas_temperatures(self, temps: NDArray[np.float64]):

        if temps.shape[1] != 3:
            raise ValueError("Gas temperatures must be a 2D array with shape (N, 3).")
        if not np.all(temps[:, 0] >= 0):
            raise ValueError("Gas temperature times must be non-negative.")
        if not np.all(temps[:, 1:] > 0):
            raise ValueError("Gas temperatures must be positive.")
        self._gas_temps = temps


    @property
    def env_cutoff(self) -> float:

        """
        :return: The cutoff distance (as a proportion of the system length) where left side gas temperatures and heat transfer coefficients switch to right side values.
        """

        return self._env_cutoff
    

    @env_cutoff.setter
    def env_cutoff(self, cutoff: float):

        if not 0 <= cutoff <= 1:
            raise ValueError("Environment cutoff must be a proportion between 0 and 1 inclusive.")
        self._env_cutoff = cutoff
        self._update_cutoff_index()


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
    def x_resolution(self) -> int:

        """
        :return: The number of spatial points in the finite difference grid along the length of the system.
        """

        return self._x_res
    

    @x_resolution.setter
    def x_resolution(self, points: int):

        if points < 2:
            raise ValueError("Number of spatial points along the length must be at least 2.")
        self._x_res = points
        self._update_x_step()
        self._update_cutoff_index()


    @property
    def r_resolution(self) -> int:

        """
        :return: The number of spatial points in the finite difference grid along the radius of the system.
        """

        return self._r_res
    

    @r_resolution.setter
    def r_resolution(self, points: int):

        if points < 2:
            raise ValueError("Number of spatial points along the radius must be at least 2.")
        self._r_res = points
        self._update_r_step()


    @property
    def initial_temperatures(self) -> NDArray[np.float64]:

        """
        :return: The initial temperatures for the finite difference grid [K].
        """

        return self._init_temps
    

    @initial_temperatures.setter
    def initial_temperatures(self, initial_temps: NDArray[np.float64] | None):

        shape = (self._r_res, self._x_res)
        if initial_temps is None:
            initial_temps = np.full(shape, self._ambient_temp)
        elif initial_temps.shape != shape:
            raise ValueError(f"Initial temperatures must be a 2D array with shape {shape}.")
        self._init_temps = initial_temps


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
        self._update_t_step()


    @property
    def diffusion_number(self) -> float:

        """
        :return: The diffusion number to use for numerical stability.
        """

        return self._diff_num
    

    @diffusion_number.setter
    def diffusion_number(self, diff_num: float):

        if diff_num <= 0 or diff_num > 0.5:
            raise ValueError("Diffusion number must be greater than 0 and less than or equal to 0.5 for numerical stability.")
        self._diff_num = diff_num
        self._update_t_step()


    @property
    def convergence_tol(self) -> float:

        """
        :return: The tolerance for convergence of the iterative solver.
        """

        return self._conv_tol


    @convergence_tol.setter
    def convergence_tol(self, conv_tol: float):

        if conv_tol <= 0:
            raise ValueError("Convergence tolerance must be positive.")
        self._conv_tol = conv_tol


    ########################################
    # Private Methods
    ########################################


    def _validate_init_temps(self):

        if self._init_temps.shape != (self._x_res, self._r_res):
            raise ValueError(f"Initial temperatures must be a 2D array with shape ({self._x_res}, {self._r_res}).")
        

    def _update_x_step(self):

        if not hasattr(self, "_x_res"):
            return
        self._x_step = self._system.length / (self._x_res - 1)
        self._update_t_step()


    def _update_r_step(self):

        if not hasattr(self, "_r_res"):
            return
        self._r_step = self._system.radius / (self._r_res - 1)
        self._r_map = np.linspace(0, self._system.radius, self._r_res)
        self._update_t_step()


    def _update_t_step(self):

        if not hasattr(self, "_diff_num"):
            return
        self._t_step = self._diff_num / (self._system.diffusivity * (self._x_step**(-2) + 2*self._r_step**(-2)))
        self._volcp = self._system.density * self._system.heat_capacity / self._t_step # Used constantly in edge and internal temp calculations


    def _update_tick_count(self):

        if not hasattr(self, "_t_step"):
            return
        self._tick_count = int(self._max_time / self._t_step)


    def _update_cutoff_index(self):

        if not hasattr(self, "_x_res"):
            return
        self._cutoff_index = round(self._env_cutoff * (self._x_res - 1))


    def _select_root(self, roots: NDArray[np.complexfloating] | NDArray[np.floating]) -> float:

        """
        Selects the appropriate root from the quartic equation based on physical constraints (real and positive).

        :param roots: The array of roots from the quartic equation.
        :return: The selected root that represents the boundary temperature [K].
        """

        real_roots = roots[abs(roots.imag) < 1e-10].real
        positive_real_roots = real_roots[real_roots > 0]
        if len(positive_real_roots) == 0:
            raise ValueError("No valid positive real roots found for boundary temperature.")
        if len(positive_real_roots) > 1:
            raise ValueError("Multiple positive real roots found for boundary temperature; unable to select unique solution.")
        return positive_real_roots[0]
    

    def _check_convergence(self, T_new: NDArray[np.float64], T_old: NDArray[np.float64], tick: int, print_every: int) -> bool:

        """
        Determines if the simulation converged after completing another tick by computing the sum of the squares of the difference between the cells. Also prints update statements.

        :param T_new: The newest 2D array of temperatures.
        :param T_old: The 2D array of temperatures from last tick.
        :param tick: The current calculation iteration.
        :param print_every: How many ticks should pass before printing another update.
        """

        sum_square = ((T_new - T_old)**2).sum()
        if print_every > 0 and tick % print_every == 0:
            print(f"Tick {tick}: [{T_new[0]:.3f}, {T_new[1]:.3f}, {T_new[2]:.3f}, ..., {T_new[-2]:.3f}, {T_new[-1]:.3f}], Sum Square: {sum_square:.2e}")
        finished = (sum_square <= self._conv_tol) and (tick * self._t_step >= self._min_time)
        return finished
    

    def _build_lookup_tables(self, times) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:

        """
        Build lookup tables for the simulation data. Drastically reduces the computational cost of interpolation during the simulation.

        :param times: The time points for which to build the lookup tables.
        :return: A tuple of the lookup tables for flux, gas temperature, and emissivity.
        """

        flux_data = self._torch_fluxs
        gas_data = self._gas_temps
        emis_data = self._system.emissivities

        flux0_lookup = np.interp(times, flux_data[:, 0], flux_data[:, 1], left=0.0)
        flux1_lookup = np.interp(times, flux_data[:, 0], flux_data[:, 2], left=0.0)
        gas_temp0_lookup = np.interp(times, gas_data[:, 0], gas_data[:, 1], left=298.15)
        gas_temp1_lookup = np.interp(times, gas_data[:, 0], gas_data[:, 2], left=298.15)
        emis_lookup = np.interp(np.arange(0, MAX_TEMP + 1), emis_data[:, 0], emis_data[:, 1])

        fluxs = (flux0_lookup, flux1_lookup)
        gasses = (gas_temp0_lookup, gas_temp1_lookup)

        return fluxs, gasses, emis_lookup
    

    def _calc_top_left_corner(self, emis: float, h: float, temp_gas: float, temp_a: float, temp_b: float, torch_flux: float) -> float:

        """
        Solves for the temperature [K] subject to both the left face and radial out boundary conditions using a weighted face-area method. Ax*(Ta - T)/dx + Ar*(T - Tb) = dr*Tx + dx*Tr

        :param emis: The emissivity of the material at its temperature.
        :param h: The heat transfer coefficient between the material and exterior gas at this point [W/m^2/K].
        :param temp_gas: The temperature of the gas outside this point [K].
        :param temp_a: The temperature of the material one node over in the x-axis [K].
        :param temp_b: The temperature of the material one node below in the r-axis [K].
        :param torch_flux: The heat flux entering the system from the torch [W/m^2].
        :return: The temperature at the node resulting from heat transfer [K].
        """

        dy = self._x_step * self._r_step**2 - self._x_step**2 * self._r_step # Convenience factor
        a4 = emis * BOLTZ * dy

        k = self._system.conductivity
        a1 = h * dy + k * (self._r_step**2 - self._x_step**2)

        a00 = h * dy * temp_gas
        a01 = emis * BOLTZ * dy * self._ambient_temp**4
        a02 = torch_flux * self._x_step * self._r_step**2
        a03 = k * self._r_step**2 * temp_a
        a04 = k * self._x_step**2 * temp_b
        a0 = -a00 - a01 - a02 - a03 + a04

        temp_corner = self._select_root(np.roots([a4, 0, 0, a1, a0]))
        return temp_corner
    

    def _calc_top_right_corner(self, emis: float, h: float, temp_gas: float, temp_a: float, temp_b: float, torch_flux: float) -> float:

        """
        Solves for the temperature [K] subject to both the right face and radial out boundary conditions using a weighted face-area method. Largely the same as the top-left corner but with a few sign changes. Ax*(T - Ta)/dx + Ar*(T - Tb) = dr*Tx + dx*Tr

        :param emis: The emissivity of the material at its temperature.
        :param h: The heat transfer coefficient between the material and exterior gas at this point [W/m^2/K].
        :param temp_gas: The temperature of the gas outside this point [K].
        :param temp_a: The temperature of the material one node before in the x-axis [K].
        :param temp_b: The temperature of the material one node below in the r-axis [K].
        :param torch_flux: The heat flux entering the system from the torch [W/m^2].
        :return: The temperature at the node resulting from heat transfer [K].
        """

        dy = self._x_step * self._r_step**2 + self._x_step**2 * self._r_step
        a4 = -emis * BOLTZ * dy

        k = self._system.conductivity
        a1 = -h * dy - k * (self._x_step**2 + self._r_step**2)

        a00 = h * dy * temp_gas
        a01 = emis * BOLTZ * dy * self._ambient_temp**4
        a02 = torch_flux * self._x_step * self._r_step**2
        a03 = k * self._r_step**2 * temp_a
        a04 = k * self._x_step**2 * temp_b
        a0 = a00 + a01 + a02 + a03 + a04

        temp_corner = self._select_root(np.roots([a4, 0, 0, a1, a0]))
        return temp_corner
    

    def _calc_bottom_left_corner(self, emis: float, h: float, temp_gas: float, temp_a: float, temp_b: float, torch_flux: float) -> float:

        """
        Solves for the temperature [K] subject to both the left face and inner zero slope symmetry boundary conditions using a weighted face-area method. Ax*(Ta - T)/dx + Ar*(Tb - T) = dr*Tx

        :param emis: The emissivity of the material at its temperature.
        :param h: The heat transfer coefficient between the material and exterior gas at this point [W/m^2/K].
        :param temp_gas: The temperature of the gas outside this point [K].
        :param temp_a: The temperature of the material one node over in the x-axis [K].
        :param temp_b: The temperature of the material one node above in the r-axis [K].
        :param torch_flux: The heat flux entering the system from the torch [W/m^2].
        :return: The temperature at the node resulting from heat transfer [K].
        """

        dy = self._x_step * self._r_step**2
        a4 = emis * BOLTZ * dy

        k = self._system.conductivity
        a1 = h * dy + k * (self._r_step**2 + 2 * self._x_step**2)

        a00 = h * dy * temp_gas
        a01 = emis * BOLTZ * dy * self._ambient_temp**4
        a02 = torch_flux * dy
        a03 = k * self._r_step**2 * temp_a
        a04 = 2 * k * self._x_step**2 * temp_b
        a0 = -a00 - a01 - a02 - a03 - a04

        temp_corner = self._select_root(np.roots([a4, 0, 0, a1, a0]))
        return temp_corner
    

    def _calc_bottom_right_corner(self, emis: float, h: float, temp_gas: float, temp_a: float, temp_b: float, torch_flux: float) -> float:

        """
        Solves for the temperature [K] subject to both the right face and inner zero slope symmetry boundary conditions using a weighted face-area method. Largely the same as the bottom-left corner but with a few sign changes. Ax*(T - Ta)/dx + Ar*(Tb - T) = dr*Tx

        :param emis: The emissivity of the material at its temperature.
        :param h: The heat transfer coefficient between the material and exterior gas at this point [W/m^2/K].
        :param temp_gas: The temperature of the gas outside this point [K].
        :param temp_a: The temperature of the material one node before in the x-axis [K].
        :param temp_b: The temperature of the material one node above in the r-axis [K].
        :param torch_flux: The heat flux entering the system from the torch [W/m^2].
        :return: The temperature at the node resulting from heat transfer [K].
        """

        dy = self._x_step * self._r_step**2
        a4 = -emis * BOLTZ * dy

        k = self._system.conductivity
        a1 = -h * dy - k * (self._r_step**2 - 2 * self._x_step**2)

        a00 = h * dy * temp_gas
        a01 = emis * BOLTZ * dy * self._ambient_temp**4
        a02 = torch_flux * dy
        a03 = k * self._r_step**2 * temp_a
        a04 = 2 * k * self._x_step**2 * temp_b
        a0 = a00 + a01 + a02 + a03 - a04

        temp_corner = self._select_root(np.roots([a4, 0, 0, a1, a0]))
        return temp_corner
    

    def _calc_left_edge(self, prev_temps_c01: NDArray[np.float64], h: float, temp_gas: float, emis_data: NDArray[np.float64], torch_flux: float) -> NDArray[np.float64]:

        """
        Calculates the temperatures subject to the left face boundary condition only.

        :param prev_temps_c01: The temperatures of the first two columns in the array at the previous time-step.
        :param h: The heat transfer coefficient of the left boundary.
        :param temp_gas: The temperature of the gas outside the left face.
        :param emis_data: The emissivities of the non-corner edge temps at the prior temperatures.
        :param torch_flux: The heat flux through the left boundary provided by the torch.
        """

        temps_b = prev_temps_c01[1:-1, 0]
        temps_h = prev_temps_c01[1:-1, 1]
        temps_alpha = prev_temps_c01[:-2, 0]
        temps_beta = prev_temps_c01[2:, 0]

        dy = 2 / self._x_step
        k = self._system.conductivity
        r_data = self._r_map[1:-1]

        a4 = -emis_data * BOLTZ * dy

        c1 = (-(k / self._x_step + h) * dy
              - 2 * k / (self._r_step**2) - self._volcp)

        a00 = (k * temps_h / self._x_step
               + h * temp_gas
               + emis_data * BOLTZ * self._ambient_temp**4
               + torch_flux) * dy
        a01 = ((temps_alpha + temps_beta) / self._r_step
               + (temps_beta - temps_alpha) / (2 * r_data)) * k / self._r_step
        a02 = self._system.density * self._system.heat_capacity * temps_b / self._t_step
        a0 = a00 + a01 + a02

        edge_temps = np.zeros_like(temps_b)
        for i, (c4, c0) in enumerate(zip(a4, a0)):
            edge_temps[i] = self._select_root(np.roots([c4, 0, 0, c1, c0]))
        return edge_temps
    

    def _calc_top_edge(self, prev_temps_rtop2: NDArray[np.float64], h_data: NDArray[np.float64], emis_data: NDArray[np.float64], gas_temps: NDArray[np.float64]) -> NDArray[np.float64]:

        """
        Calculates the temperatures subject to the lateral face boundary condition only.

        :param prev_temps_rtop2: The temperatures of the first last two rows in the array at the previous time-step.
        :param h: The heat transfer coefficients of each point.
        :param emis_data: The emissivities of the non-corner edge temps at the prior temperatures.
        :param gas_temps: The gas temperatures outside each node.
        """

        temps_b = prev_temps_rtop2[1, 1:-1]
        temps_g = prev_temps_rtop2[1, :-2]
        temps_h = prev_temps_rtop2[1, 2:]
        temps_alpha = prev_temps_rtop2[0, 1:-1]

        dy = 2 / self._r_step + 1 / self._system.radius
        k = self._system.conductivity

        a4 = -emis_data * BOLTZ * dy

        a1 = (-2 * k / (self._x_step**2)
              - 2 * k / (self._r_step**2)
              - h_data * dy - self._volcp)
        
        a0 = (k * (temps_h + temps_g) / (self._x_step**2)
              + 2 * k * temps_alpha / (self._r_step**2)
              + h_data * dy * gas_temps
              + emis_data * BOLTZ * dy * self._ambient_temp**4
              + self._volcp * temps_b)
        
        edge_temps = np.zeros_like(temps_b)
        for i, (c4, c1, c0) in enumerate(zip(a4, a1, a0)):
            edge_temps[i] = self._select_root(np.roots([c4, 0, 0, c1, c0]))
        return edge_temps


emis_a = np.array([[298.15, 0.21], [383.15, 0.33]])
htcs_a = (50.0, 50.0)

test_system_a = ConductiveSystemAxial(
    diff = 1.58e-4,
    cond = 400.0,
    emis = emis_a,
    htcs = htcs_a,
    dens = 8960.0,
    cphc = 418.0,
    length = 0.01,
    radius = 0.05
)

x_res_a = 10
r_res_a = 7
init_temps_a = np.full((r_res_a, x_res_a), 298.15)
torch_fluxs_a = np.array([[0.0, 0.5e6, 0.0]])
gas_temps_a = np.array([[0.0, 2000.0, 298.15]])

test_solver_a = FiniteDiffSolverAxial(
    system = test_system_a,
    x_res = x_res_a,
    r_res = r_res_a,
    initial_temps = init_temps_a,
    torch_fluxs = torch_fluxs_a,
    gas_temps = gas_temps_a
)

prev_temps = np.full((r_res_a, x_res_a), 298.15, dtype=np.float64)
temps_c01 = prev_temps[:, 0:2].transpose()
emis_data = np.full(r_res_a - 2, 0.21)

print(test_solver_a._calc_top_left_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.5e6))
print(test_solver_a._calc_top_right_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.0))
print(test_solver_a._calc_bottom_left_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.5e6))
print(test_solver_a._calc_bottom_right_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.0))
# print(test_solver_a._calc_left_edge(temps_c01, 50.0, 2000.0, emis_data, 500000.0))

temps_rtop2 = init_temps_a[-2:, :] # Top two rows (printed as bottom)
temps_rtop2[1, 0] = 300
h_data = np.full(x_res_a-2, 50.0, dtype=np.float64)
emis_data = np.full_like(h_data, 0.21)
gas_temps = np.full_like(h_data, 298.15)
edge_temps = test_solver_a._calc_top_edge(temps_rtop2, h_data, emis_data, gas_temps)
print(test_solver_a._calc_top_edge(temps_rtop2, h_data, emis_data, gas_temps))