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
        max_sim_time: float | None = None,
        diff_num: float = 0.1,
        max_t_step: float = 1e-3,
        force_overwrite: bool = False
):
    
    diff, cond, emis = material
    system = ConductiveSystem1D(diff, cond, emis, length)
    if init_temps is None:
        init_temps = np.full(x_res, ambient_temp, dtype=np.float64)
    if max_sim_time is None:
        htc = np.max(htcs_kernel[:, 1])
        max_sim_time = 5. * cond * length / (diff * htc) + 50. # Based on fitting simulated data + tolerance
    solver = FiniteDiffSolver1D(system, init_temps, gas_temp_kernel, htcs_kernel, period, ambient_temp, x_res, min_sim_time, max_sim_time, diff_num, max_t_step_size=max_t_step)
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


filename = "thurs/hh100_a192n7_k142p1_l12.hdf5"
htcs_array = np.array([
    [0., 10., 10.],
    [0.499999, 10., 10.],
    [0.5, 100., 10.],
    [2.999999, 100., 10.]
], dtype=np.float64)
material = (4.91e-5, 57.5, np.array([[400., 0.15], [1000., 0.25]], dtype=np.float64))
run_configuration(filename, material, 0.012, GAS_2200_CONSTANT_ARRAY, htcs_array, period=3.0, diff_num=0.1, max_t_step=1e-4, force_overwrite=False)

data = DataHandler(filename)
data.load_data()
print_reports(data)
time_evolution_plot(data)
front_and_back_temp_plot(data)
data.close()