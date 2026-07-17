from domain.conductive_system_1D import ConductiveSystem1D
from domain.finite_diff_solver_1D import FiniteDiffSolver1D
from domain.prefabs import *
from itertools import product
from numpy.typing import NDArray
from traceback import print_exc
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
        force_overwrite: bool = False,
        print_every: int = int(1e5)
):
    
    diff, cond, emis = material
    system = ConductiveSystem1D(diff, cond, emis, length)
    if init_temps is None:
        init_temps = np.full(x_res, ambient_temp, dtype=np.float64)
    if max_sim_time is None:
        htc = np.max(htcs_kernel[:, 1])
        max_sim_time = 6 * cond * length / (diff * htc) + 200. # Based on fitting simulated data + tolerance
    solver = FiniteDiffSolver1D(system, init_temps, gas_temp_kernel, htcs_kernel, period, ambient_temp, x_res, min_sim_time, max_sim_time, diff_num, max_t_step_size=max_t_step)
    solver.run_simulation(filename, print_every=print_every, force_overwrite=force_overwrite)


def get_final_temps(filename: str) -> NDArray[np.float64]:

    steadystate_data = DataHandler(filename)
    steadystate_data.load_data()
    final_temps = steadystate_data.temps[-1]
    steadystate_data.close()
    return final_temps


def print_reports(data: DataHandler):

    data.report_convergence_time()
    data.report_lag_time()


def analyze_data(filename: str):

    data = DataHandler(filename)
    data.load_data()
    print_reports(data)
    time_evolution_plot(data)
    front_and_back_temp_plot(data)
    data.close()


def run_mini(filename: str, htc: float, diff: float, cond: float, length: float, print_every: int=int(1e6)):

    htcs_array = np.array([
        [0., 10., 10.],
        [0.499999, 10., 10.],
        [0.5, htc, 10.],
        [2.999999, htc, 10.]
    ], dtype=np.float64)
    material = (diff, cond, np.array([[0.1, 0.5]], dtype=np.float64))
    run_configuration(filename, material, length, GAS_2200_CONSTANT_ARRAY, htcs_array, period=3., diff_num=0.1, max_t_step=1e-4, force_overwrite=True, print_every=print_every)


def run_batch(htcs: tuple[float, ...], diff_cond_pairs: tuple[tuple[float, float], ...], lengths: tuple[float, ...]):

    failures: list[tuple[int, float, float, float, float]] = []
    for trial, (htc, diff_cond_pair, length) in enumerate(
        product(htcs, diff_cond_pairs, lengths)
    ):
        # if trial not in [56]:
        #     continue
        filename = f"batch/trial{trial}.hdf5"
        diff, cond = diff_cond_pair
        try:
            run_mini(filename, htc, diff, cond, length)
            data = DataHandler(filename)
            data.load_data()
            print_reports(data)
            data.close()
        except (ValueError, RuntimeError):
            failures.append((trial, htc, diff, cond, length))
            print_exc()
    print(f"{len(failures)} trials failed.\n{failures}")


# filename = "test.hdf5"
# htcs_array = np.array([
#     [0., 10., 10.],
#     [0.499999, 10., 10.],
#     [0.5, 100., 10.],
#     [2.999999, 100., 10.]
# ], dtype=np.float64)
# material = (4.91e-5, 57.5, np.array([[0.1, 0.5]], dtype=np.float64))
# run_configuration(filename, material, 0.003, GAS_2200_CONSTANT_ARRAY, htcs_array, period=3.0, diff_num=0.1, max_t_step=1e-4, force_overwrite=False)

htcs = (100., 300., 600.)
diff_cond_pairs = (
    (1e-6, 2.),
    (1.5e-6, 7.),
    (3e-6, 6.),
    (3e-6, 20.),
    (6e-6, 15.),
    (8e-6, 50.),
    (2e-5, 40.),
    (3e-5, 130.),
    (8e-5, 200.)
)
lengths = (0.003, 0.006, 0.012)
run_batch(htcs, diff_cond_pairs, lengths)