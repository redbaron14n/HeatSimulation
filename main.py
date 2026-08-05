from domain.conductive_system_1D import ConductiveSystem1D
from domain.finite_diff_solver_1D import FiniteDiffSolver1D
from domain.prefabs import *
from domain.steadystate import find_steadystate
from itertools import product
from numpy.typing import NDArray
from pathlib import Path
from time import perf_counter
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


def run_mini(
        filename: str,
        cond: float,
        diff: float,
        emis: float,
        h_heat: float,
        h_nat: float,
        length: float,
        gas: float,
        ambi: float,
        dur: float,
        per: float,
        print_every: int=int(1e6),
        max_t_step: float=1e-4,
        diff_num: float=0.1,
        max_sim_time: float | None = None,
        x_res: int=25
):

    htcs_array = np.array([
        [0., h_nat, h_nat],
        [dur, h_heat, h_nat]
    ], dtype=np.float64)
    material = (diff, cond, np.array([[0.1, emis]], dtype=np.float64))
    gas_temps = np.array([[0., gas, ambi]], dtype=np.float64)
    init_temps = find_steadystate(cond, emis, h_heat, h_nat, length, gas, ambi, dur, per, x_res)
    run_configuration(
        filename=filename,
        material=material,
        length=length,
        gas_temp_kernel=gas_temps,
        htcs_kernel=htcs_array,
        period=per,
        x_res=x_res,
        ambient_temp=ambi,
        init_temps=init_temps,
        max_sim_time=max_sim_time,
        max_t_step=max_t_step,
        diff_num=diff_num,
        print_every=print_every,
        force_overwrite=True
    )


def run_batch(
        subdir: str,
        conds: tuple[float, ...],
        diffs: tuple[float, ...],
        emiss: tuple[float, ...],
        h_heats: tuple[float, ...],
        h_nats: tuple[float, ...],
        lengths: tuple[float, ...],
        gas_temps: tuple[float, ...],
        ambi_temps: tuple[float, ...],
        chop_durs: tuple[float, ...],
        chop_pers: tuple[float, ...],
        start_indx: int=0,
        rerun: list[int] | None = None
):

    start_time = perf_counter()
    completed = 0
    failures: list[tuple[int, float, float, float, float, float, float, float, float, float, float]] = []
    for trial, (emis, h_heat, h_nat, length, gas, ambi, dur, per, cond, diff) in enumerate(
        product(emiss, h_heats, h_nats, lengths, gas_temps, ambi_temps, chop_durs, chop_pers, conds, diffs)
    ):
        trial += start_indx
        if (rerun is not None) and (trial not in rerun):
            continue
        print(f"Trial {trial}, Cond: {cond}, Diff: {diff}, Emis: {emis}, h_heat: {h_heat}, h_nat: {h_nat}, length: {length}, gas: {gas}, ambi: {ambi}, dur: {dur}, per: {per}")
        filename = f"{subdir}/trial{trial}.hdf5"
        try:
            run_mini(filename, cond, diff, emis, h_heat, h_nat, length, gas, ambi, dur, per, max_sim_time=1000., print_every=int(1e6), diff_num=0.1, max_t_step=1e-4, x_res=25)
            data = DataHandler(filename)
            data.load_data()
            print_reports(data)
            data.close()
        except (ValueError, RuntimeError):
            failures.append((trial, cond, diff, emis, h_heat, h_nat, length, gas, ambi, dur, per))
            print_exc()
        completed += 1
        runtime = perf_counter() - start_time
        avg_runtime = runtime / completed
        print(f"Trials Complete: {completed}, Total Runtime: {runtime:.3f}s, Avg. {avg_runtime:.3f}s/trial")
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


conds = 1., 10., 100., 1000.
diffs = 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3
emiss = 0.3,
h_heats = 10., 100., 1000.
h_nats = 5.,
lengths = 0.001, 0.01
gas_temps = 3030.,
ambi_temps = 273.15,
chop_durs = 0.3, 1.
chop_pers = 1.2, 1.5, 2.

# conds = 20.,
# diffs = 3e-5,
# emiss = 0.3,
# h_heats = 200.,
# h_nats = 15.,
# lengths = 0.005,
# gas_temps = 3030.,
# ambi_temps = 273.15,
# chop_durs = 0.3, 1.
# chop_pers = 1.2, 1.5, 2.

run_batch("batch2", conds, diffs, emiss, h_heats, h_nats, lengths, gas_temps, ambi_temps, chop_durs, chop_pers)

# analyze_data("batch2/trial4.hdf5")