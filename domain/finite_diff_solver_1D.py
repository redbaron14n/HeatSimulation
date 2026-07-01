# ###################
# Ian Janes
# Prof. Don Lipkin
# Finite Difference Solver class file
# ###################

import numpy as np
from domain.conductive_system_1D import ConductiveSystem1D
from domain.lookup_tables import LookupTables
from numpy.typing import NDArray
from services.data_handling import DataHandler
from services.snapshot_buffer_dataclass import SnapshotBuffer

np.set_printoptions(linewidth=200, threshold=10)

BOLTZ = 5.670374419e-8 # Stefan-Boltzmann constant [W/m^2/K^4]
MAX_TEMP = 6000 # Maximum temperature in the simulation [K]


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


    def _validate_init_temps(self, temps: NDArray[np.float64]): # Call this before running the simulation to catch if x_res was changed without updating initial temperatures

        if len(temps) != self._x_res:
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


    def _check_convergence(self, T_new: NDArray[np.float64], T_old: NDArray[np.float64], conv_tol: float, sim_time: float) -> tuple[bool, float]:

        rms = np.sqrt(((T_new - T_old)**2).mean())
        finished = (rms <= conv_tol) and (sim_time >= self._min_time)
        return finished, rms
    

    def _print_update(self, temps: NDArray[np.float64], tick: int, sim_time: float, rms: float, saved: int, print_every: int):

        if (print_every > 0) and (tick % print_every == 0):
            print(f"Tick {tick:,d}, T+{sim_time:.3f}[s], RMS: {rms:.3e}, Saved {saved:,d} snapshots.\n{temps}")


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
        :return: A tuple of the lookup tables for gas temperature, heat transfer coefs, and emissivity.
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
    

    def _construct_metadata(self) -> dict[str, int | float | NDArray[np.float64]]:

        metadata: dict[str, int | float | NDArray[np.float64]] = {
            "dim": int(1),
            "dx": self._x_step,
            "dt": self._t_step,
            "length": self._system.length,
            "diff": self._system.diffusivity,
            "cond": self._system.conductivity,
            "temp_ambient": self._ambient_temp,
            "emis": self._system.emissivities,
            "htcs": self._htcs,
            "gas_temps": self._gas_temps
        }
        return metadata
    

    def _handle_snapshot_saving(
            self,
            file: DataHandler,
            buffer: SnapshotBuffer,
            new_temps: NDArray[np.float64],
            time: float,
            converged: bool,
            saved: int,
            save_tol: float,
            chunk_size: int
    ) -> int:
        
        rms_last = np.sqrt(((new_temps - buffer.last_saved)**2).mean()) # pyright: ignore[reportPossiblyUnboundVariable]
        if (rms_last >= save_tol) or converged:
            buffer.times.append(time) # pyright: ignore[reportPossiblyUnboundVariable]
            buffer.temps.append(new_temps) # pyright: ignore[reportPossiblyUnboundVariable]
            buffer.last_saved = new_temps
            saved += 1
            buffer.size += 1 # pyright: ignore[reportPossiblyUnboundVariable]
        if ((buffer.size != 0) and (buffer.size % chunk_size == 0)) or converged: # pyright: ignore[reportPossiblyUnboundVariable]
            file.append_snapshots(np.array(buffer.times), np.array(buffer.temps))
        return saved


    ########################################
    # Public Methods
    ########################################


    def create_chop_schedule(self, array_map: NDArray[np.float64], period: float, max_time: float) -> NDArray[np.float64]:

        """
        Given a 2D array of floats, returns a longer repeating version where the first column is increasing each repetition by 'period' until 'max_time' is reached.

        :param array_map: The 2D array of floats mapping gas temperatures or heat transfer coefficients to simulation time.
        :param period: By how much each unit array's time values should increaase each time they repeat.
        :param max_time: Determines how many repetitions should be made. The final row will have first column less than this value.

        :return: The lengthed periodic 2D array.
        """

        if len(array_map.shape) != 2:
            raise ValueError(f"Inputted array must be 2D. Given array has shape {array_map.shape}.")
        if any(array_map[:, 0] < 0.):
            raise ValueError("All entries in the first column of the array must be non-negative.")
        if len(np.unique(array_map[:, 0])) < len(array_map[:, 0]):
            raise ValueError("All entries in the first column of the array must be unique.")
        array_map = array_map[array_map[:, 0].argsort()]
        if period <= array_map[-1, 0]:
            raise ValueError(f"Period must be greater than the greatest entry in the first column, but {period} <= {array_map[-1, 0]}.")
        if max_time <= period:
            raise ValueError(f"The max time must be greater than the given period, but {max_time} <= {period}.")
        
        n_repeats = float(np.floor(max_time / period))
        blocks: list[NDArray[np.float64]] = []
        i = 0.
        while i <= n_repeats:
            block = array_map.copy()
            block[:, 0] += i * period
            blocks.append(block)
            i += 1
        result = np.vstack(blocks)
        return result[result[:, 0] < max_time]


    def run_simulation(
            self,
            filename: str,
            save_evo: bool = True,
            save_final: bool = False,
            load_prev: bool = False,
            conv_tol: float = 1e-6,
            print_every: int = 1000,
            save_tol: float = 1e-3,
            chunk_size: int = 1000
    ):

        """
        Run the finite difference simulation for the given system and solver parameters.

        :param filename: The filename to save the data to.
        :param save_evo: Whether to save periodic snapshots of the temperature evolution over time to the file. Default is True.
        :param save_final: Whether to save the final temperature distribution to the file. Default is False.
        :param load_prev_final: Whether to load the previously saved final temperature distribution to use as this run's initial temperature distribution. Default is False.
        :param conv_tol: The convergence tolerance of the simulation. The simulation will declare steady-state if the sum of the squares of the differences between iterations falls to or below this tolerance. Default is 1e-6.
        :param print_every: The interval at which to print the simulation progress. Default is 1000. 0 means no printing.
        :param save_tol: How large the sum square of differences between latest and last saved temperature distributions must be before it is saved. Default is 1e-3.
        :param chunk_size: How many distributions to calculate before saving.
        """

        file = DataHandler(filename, load=False)
        file.initialize_storage((self._x_res,), self._construct_metadata())
        temps = self._init_temps
        if load_prev:
            temps = file.init_temps
        self._validate_init_temps(temps) # Ensure initial temperatures are valid before starting simulation
        times = np.arange(0, self._max_time + self._t_step, self._t_step, dtype=np.float64)
        gasses, htcs, emis_lookup = self._build_lookup_tables(times)
        saved = 0
        if save_evo:
            buffer = SnapshotBuffer(times=[0.], temps=[temps.copy()], size=1, last_saved=temps.copy())
            saved = 1
        converged = False
        tick = 0
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
            converged, rms = self._check_convergence(T_new, temps, conv_tol, time)
            temps = T_new
            self._print_update(T_new, tick, time, rms, saved, print_every)
            if save_evo:
                saved = self._handle_snapshot_saving(file, buffer, T_new, time, converged, saved, save_tol, chunk_size) # pyright: ignore[reportPossiblyUnboundVariable]
        self._final_temps = temps
        if save_final:
            file.init_temps = temps
        self._simulation_summary(tick, saved, converged)