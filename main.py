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
        diff_num: float = 0.1,
        force_overwrite: bool = False
):
    
    diff, cond, emis = material
    system = ConductiveSystem1D(diff, cond, emis, length)
    if init_temps is None:
        init_temps = np.full(x_res, ambient_temp, dtype=np.float64)
    solver = FiniteDiffSolver1D(system, init_temps, gas_temp_kernel, htcs_kernel, period, ambient_temp, x_res, min_sim_time, max_sim_time, diff_num)
    solver.run_simulation(filename, force_overwrite=force_overwrite)


def get_final_temps(filename: str) -> NDArray[np.float64]:

    steadystate_data = DataHandler(filename)
    steadystate_data.load_data()
    final_temps = steadystate_data.temps[-1]
    steadystate_data.close()
    return final_temps


def print_reports(data: DataHandler):

    data.report_convergence_time()
    data.report_lag_time()


filename = "Biot_hh300_hl10_k10_l12_a10.hdf5"
htcs_array = np.array([
    [0., 10., 10.],
    [0.5, 10., 10.],
    [0.52, 300., 10.],
    [2.98, 300., 10.]
], dtype=np.float64)
material = (1e-5, 10., np.array([[400., 0.15], [1000., 0.25]], dtype=np.float64))
run_configuration(filename, material, 0.012, GAS_2200_CONSTANT_ARRAY, htcs_array, 3.0, max_sim_time=300., diff_num=0.1, force_overwrite=False)

data = DataHandler(filename)
data.load_data()
print_reports(data)
time_evolution_plot(data)
front_and_back_temp_plot(data)
data.close()