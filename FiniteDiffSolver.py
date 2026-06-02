# ###################
# Ian Janes
# Prof. Don Lipkin
# Finite Difference Solver class file
# ###################

from ConductiveSystem import ConductiveSystem1D

class FiniteDiffSolver1D:

    def __init__(
            self,
            system: ConductiveSystem1D,
            spatial_res: int = 100,
            max_sim_time: float = 100.0,
            diff_num: float = 0.5,
            conv_tol: float = 1e-6
        ):

        """
        :param ConductiveSystem1D system: The conductive system to be solved.
        :param int spatial_res: The number of spatial points to use in the finite difference grid.
        :param float max_sim_time: The maximum simulation time to run the solver for [s].
        :param float diff_num: The diffusion number (alpha * dt / dx^2) to use for numerical stability.
        :param float conv_tol: The tolerance for convergence of the iterative solver.
        """

        self._system = system
        self.spatial_resolution = spatial_res
        self.max_simulation_time = max_sim_time
        self.diffusion_number = diff_num
        self.convergence_tol = conv_tol
        self._set_x_step()
        self._set_t_step()


    @property
    def system(self) -> ConductiveSystem1D:

        """
        :return: The conductive system under analysis.
        """

        return self._system

    
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


    def _set_x_step(self):

        self._x_step = self._system.length / (self._x_res - 1)


    def _set_t_step(self):

        self._t_step = self._diff_num * (self._x_step ** 2) / self._system.diffusivity

test_system = ConductiveSystem1D(1.0, 1.0, (0.5, 0.5), (10.0, 10.0), 1.0, (300.0, 300.0, 300.0, 300.0))
test = FiniteDiffSolver1D(test_system, 100, 100.0, 0.5, 1e-6)