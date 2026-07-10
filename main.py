from domain.conductive_system_1D import ConductiveSystem1D
from domain.finite_diff_solver_1D import FiniteDiffSolver1D
from domain.prefabs import *
from numpy.typing import NDArray
from visualizer import *
import numpy as np


def run_configuration(
        filename: str,
        material: tuple[float, float, NDArray[np.float64]],
        length: float,
        gas_temp_kernel: NDArray[np.float64],
        htcs_kernel: NDArray[np.float64],
        period: float = 0.,
        x_res: int = 25,
        ambient_temp: float = 298.15,
        min_sim_time: float = 5.,
        max_sim_time: float = 50.,
):
    
    diff, cond, emis = material
    system = ConductiveSystem1D(diff, cond, emis, length)
    init_temps = np.full(x_res, ambient_temp, dtype=np.float64)
    solver = FiniteDiffSolver1D(system, init_temps, gas_temp_kernel, htcs_kernel, ambient_temp, x_res, min_sim_time, max_sim_time)
    if period != 0.:
        solver.gas_temperatures = solver.create_chop_schedule(gas_temp_kernel, period, max_sim_time)
        solver.heat_transfer_coefs = solver.create_chop_schedule(htcs_kernel, period, max_sim_time)
    solver.run_simulation(filename)


filename = "copper_chop_after_steady.hdf5"
run_configuration(filename, COPPER, 0.0035, GAS_2200_CONSTANT_ARRAY, HTCS_300_CONSTANT_ARRAY, 0., max_sim_time=1000)
time_evolution_plot(filename)
front_and_back_temp_plot(filename)