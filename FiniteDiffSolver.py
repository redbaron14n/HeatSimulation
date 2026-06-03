# ###################
# Ian Janes
# Prof. Don Lipkin
# Finite Difference Solver class file
# ###################

from ConductiveSystem import ConductiveSystem1D
from numpy import abs, float64, linspace, roots
from numpy.typing import NDArray

BOLTZ = 5.670374419e-8 # Stefan-Boltzmann constant [W/m^2/K^4]

class FiniteDiffSolver1D:

    def __init__(
            self,
            system: ConductiveSystem1D,
            ambient_temp: float = 298.15,
            spatial_res: int = 100,
            max_sim_time: float = 100.0,
            diff_num: float = 0.5,
            conv_tol: float = 1e-6
        ):

        """
        :param ConductiveSystem1D system: The conductive system to be solved.
        :param float ambient_temp: The ambient temperature for the simulation [K].
        :param int spatial_res: The number of spatial points to use in the finite difference grid.
        :param float max_sim_time: The maximum simulation time to run the solver for [s].
        :param float diff_num: The diffusion number (alpha * dt / dx^2) to use for numerical stability.
        :param float conv_tol: The tolerance for convergence of the iterative solver.
        """

        self._system = system
        self._ambient_temp = ambient_temp
        self._x_res = spatial_res
        self._max_time = max_sim_time
        self._diff_num = diff_num
        self._conv_tol = conv_tol
        self._update_x_step()


    @property
    def system(self) -> ConductiveSystem1D:

        """
        :return: The conductive system under analysis.
        """

        return self._system
    

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
    def max_simulation_time(self) -> float:

        """
        :return: The maximum simulation time [s].
        """

        return self._max_time
    

    @max_simulation_time.setter
    def max_simulation_time(self, max_time: float):

        if max_time <= 0:
            raise ValueError("Maximum simulation time must be greater than 0.")
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

        self._x_step = self._system.length / (self._x_res - 1)
        self._update_t_step() # Update t_step based on new x_step


    def _update_t_step(self):

        self._t_step = self._diff_num * (self._x_step ** 2) / self._system.diffusivity
        self._update_tick_count() # Update tick count based on new t_step


    def _update_tick_count(self):

        self._tick_count = int(self._max_time / self._t_step) + 1


    def _init_temps(self) -> NDArray[float64]:

        """
        Initialize the temperature array for the finite difference grid based on the system's initial temperatures.

        :return: A 1D array of initial temperatures at each spatial point.
        """

        temps = self._system.temperatures
        return linspace(temps[1], temps[2], self._x_res)
    

    def _init_x_grid(self) -> NDArray[float64]:

        """
        Initialize the spatial grid for the finite difference solver.

        :return: A 1D array of spatial points along the length of the system.
        """

        return linspace(0, self._system.length, self._x_res)
    

    def solve_boundary_temp(self, k: float, epsilon: float, h: float, T_inside: float, T_gas: float) -> float:

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
        roots_T = roots(coeffs)
        #print(roots_T)
        real_roots = roots_T[abs(roots_T.imag) < 1e-10].real
        T_bound: float = real_roots.max()
        #print(T_bound)
        return T_bound


    def run_simulation(self):

        """
        Run the finite difference simulation for the given system and solver parameters.
        """

        temps = self._init_temps()
        x_grid = self._init_x_grid()
        k = self._system.conductivity
        epsilon1, epsilon2 = self._system.emissivities
        h1, h2 = self._system.heat_transfer_coefs
        T_gas1, T_gas2 = self._system.temperatures[0], self._system.temperatures[3]
        converged = False
        tick = 0
        while (not converged) and (tick < self._tick_count):
        #while tick <= 100:
            tick += 1
            T_new = temps.copy()
            T_inside1, T_inside2 = temps[1], temps[-2]
            T_new[0] = self.solve_boundary_temp(k, epsilon1, h1, T_inside1, T_gas1)
            T_new[-1] = self.solve_boundary_temp(k, epsilon2, h2, T_inside2, T_gas2)
            for i in range(1, self._x_res - 1):
                T_new[i] = temps[i] + self._diff_num * (temps[i+1] - 2*temps[i] + temps[i-1])
            sum_square = ((T_new - temps) ** 2).sum()
            if sum_square <= self._conv_tol:
                converged = True
            temps = T_new
            if tick % 1000 == 0:
                print(f"Tick {tick}: [{temps[0]:.3f}, {temps[1]:.3f}, {temps[2]:.3f}, ..., {temps[-2]:.3f}, {temps[-1]:.3f}], Sum Square: {sum_square:.2e}")
        if converged:
            print(f"Simulation converged in {tick} ticks.")
        else:
            print(f"Simulation reached maximum time without convergence.")
        self._final_temps = temps

test_system = ConductiveSystem1D(1.58e-4, 40.0, (0.5, 0.5), (316.227766, 100.0), 0.0035, (2500.0, 350.0, 300.0, 300.0))
test = FiniteDiffSolver1D(test_system, 300.0, 25, 100, 0.5, 1e-13)

test.run_simulation()
print(test._final_temps)