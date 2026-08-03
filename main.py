from domain.conductive_system_1D import ConductiveSystem1D
from domain.finite_diff_solver_1D import FiniteDiffSolver1D
from domain.prefabs import *
from itertools import product
from numpy.typing import NDArray
from pathlib import Path
from traceback import print_exc
from visualizer import *
import numpy as np
import pandas as pd


BATCH_FMTS = {
    "Trial": "{:.0f}",
    "k [W/m/K]": "{:.3g}",
    "alpha [m^2/s]": "{:.3g}",
    "Emissivity": "{:.3g}",
    "h_heat [W/m^2/K]": "{:.3g}",
    "h_nat [W/m^2/K]": "{:.3g}",
    "L [m]": "{:.3g}",
    "Gas Temp [K]": "{:.3f}",
    "Ambient Temp [K]": "{:.3f}",
    "Chop Duration [s]": "{:.3f}",
    "Chop Period [s]": "{:.3f}",
    "Avg Temp [K]": "{:.3f}",
    "Front Min Time [s]": "{:.3f}",
    "Front Min Temp [K]": "{:.3f}",
    "Front Max Time [s]": "{:.3f}",
    "Front Max Temp [K]": "{:.3f}",
    "Rear Temp [K]": "{:.3f}",
    "Rear Min Time [s]": "{:.3f}",
    "Rear Min Temp [K]": "{:.3f}",
    "Rear Max Time [s]": "{:.3f}",
    "Rear Max Temp [K]": "{:.3f}"
}


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
        max_sim_time = 3 * cond * length / (diff * htc) + 400. # Based on fitting simulated data + tolerance
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


def run_mini(filename: str, chop_dur: float, chop_period: float, htc: float, diff: float, cond: float, length: float, print_every: int=int(1e6)):

    htcs_array = np.array([
        [0., 10., 10.],
        [chop_dur-1e-6, 10., 10.],
        [chop_dur, htc, 10.],
        [chop_period-1e-6, htc, 10.]
    ], dtype=np.float64)
    material = (diff, cond, np.array([[0.1, 0.5]], dtype=np.float64))
    run_configuration(filename, material, length, GAS_2200_CONSTANT_ARRAY, htcs_array, period=chop_period, diff_num=0.1, max_t_step=1e-4, force_overwrite=True, print_every=print_every)


def run_batch(
        chop_durations: tuple[float, ...],
        chop_periods: tuple[float, ...],
        htcs: tuple[float, ...],
        diff_cond_pairs: tuple[tuple[float, float], ...],
        lengths: tuple[float, ...],
        start_indx: int=0,
        rerun: list[int] | None = None
    ):

    failures: list[tuple[int, float, float, float, float, float, float]] = []
    for trial, (dur, period, htc, diff_cond_pair, length) in enumerate(
        product(chop_durations, chop_periods, htcs, diff_cond_pairs, lengths)
    ):
        trial += start_indx
        if (rerun is not None) and (trial not in rerun):
            continue
        filename = f"batch/trial{trial}.hdf5"
        diff, cond = diff_cond_pair
        try:
            run_mini(filename, dur, period, htc, diff, cond, length)
            data = DataHandler(filename)
            data.load_data()
            print_reports(data)
            data.close()
        except (ValueError, RuntimeError):
            failures.append((trial, dur, period, htc, diff, cond, length))
            print_exc()
    print(f"{len(failures)} trials failed.\n{failures}")


def extract_data_from_batch(subdirectory: str, files_basename: str, data_filename: str):

    if bool(Path(files_basename).suffix):
        raise ValueError("Enter a file basename without an extension.")
    if Path(data_filename).suffix not in [".csv", ""]:
        raise ValueError(f"Data filename provided {data_filename} has an invalid extension. Provide with no extension or '.csv'.")
    if Path(data_filename).suffix == "":
        data_filename += ".csv"

    l = len(files_basename)
    files = sorted(
        Path(f"data/{subdirectory}").glob(f"{files_basename}*.hdf5"),
        key=lambda p: int(p.stem[l:]) # Sorts by file number
    )

    if not files:
        raise FileNotFoundError(f"No data/{subdirectory}/{files_basename}*.hdf5 files found.")
    
    data_list: list[NDArray[np.float64]] = []
    for path in files:
        entry = np.empty(len(BATCH_FMTS), dtype=np.float64)
        entry[0] = int(path.stem[l:])
        rel_path = path.relative_to("data")
        data = DataHandler(str(rel_path))
        data.load_data()
        data.extract_data(entry)
        data.close()
        data_list.append(entry)
    data_array = np.asarray(data_list)
    df = pd.DataFrame(data_array, columns=list(BATCH_FMTS))
    df_formatted = df.copy()
    for col, fmt in BATCH_FMTS.items():
        df_formatted[col] = df[col].map(fmt.format)
    df_formatted.to_csv(f"data/{subdirectory}/{data_filename}", index=False)


extract_data_from_batch("batch", "trial", "batch_data.csv")