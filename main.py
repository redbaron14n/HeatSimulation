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
        init_temps: NDArray[np.float64] | None = None,
        min_sim_time: float = 5.,
        max_sim_time: float = 50.,
        force_overwrite: bool = False
):
    
    diff, cond, emis = material
    system = ConductiveSystem1D(diff, cond, emis, length)
    if init_temps is None:
        init_temps = np.full(x_res, ambient_temp, dtype=np.float64)
    solver = FiniteDiffSolver1D(system, init_temps, gas_temp_kernel, htcs_kernel, ambient_temp, x_res, min_sim_time, max_sim_time)
    if period != 0.:
        solver.gas_temperatures = solver.create_chop_schedule(gas_temp_kernel, period, max_sim_time)
        solver.heat_transfer_coefs = solver.create_chop_schedule(htcs_kernel, period, max_sim_time)
    solver.run_simulation(filename, force_overwrite=force_overwrite)


def get_final_temps(filename: str) -> NDArray[np.float64]:

    steadystate_data = DataHandler(filename)
    steadystate_data.load_data()
    final_temps = steadystate_data.temps[-1]
    steadystate_data.close()
    return final_temps


def print_reports(data: DataHandler):

    data.report_convergence_time()
    data.report_final_avg_temp()
    # data.report_lag_time()


filename = "saving_test.hdf5"
run_configuration(filename, K10L3_TRIAL, 0.003, GAS_2200_CONSTANT_ARRAY, HTCS_100_5_50_2_300_ARRAY, 3.0, max_sim_time=200., force_overwrite=True)
init_temps = get_final_temps(filename)

data = DataHandler(filename)
data.load_data()
print_reports(data)
time_evolution_plot(data)
front_and_back_temp_plot(data)
data.close()