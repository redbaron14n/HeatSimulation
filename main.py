from domain.conductive_system_1D import ConductiveSystem1D
from domain.finite_diff_solver_1D import FiniteDiffSolver1D
from pathlib import Path
from visualizer import time_evolution_plot
import numpy as np


diff = 4.50e-4
cond = 20.
emis = np.array([[0.001, 0.40], [1000., 0.33]], dtype=np.float64)
length = 0.01

test_system = ConductiveSystem1D(diff, cond, emis, length)

x_res = 25
initial_temps = np.full(x_res, 298.15)
gas_temps_kernel = np.array([[0., 2500., 298.15], [1., 2500., 298.15], [1.1, 298.15, 298.15], [1.9, 298.15, 298.15]], dtype=np.float64)
htcs_kernel = np.array([[0., 5000., 10.], [1., 5000., 10.], [1.1, 10., 10.], [1.9, 10., 10.]], dtype=np.float64)
ambient_temp = 298.15

test_solver = FiniteDiffSolver1D(test_system, initial_temps, gas_temps_kernel, htcs_kernel, ambient_temp, x_res, min_sim_time=1.0, max_sim_time=50)

gas_temps = test_solver.create_chop_schedule(gas_temps_kernel, 2., 50.)
test_solver.gas_temperatures = gas_temps
htcs = test_solver.create_chop_schedule(htcs_kernel, 2., 50.)
test_solver.heat_transfer_coefs = htcs

test_solver.run_simulation("steel316_survival_test_extreme.hdf5", conv_tol=1e-6, print_every=10000, save_tol=0.01, time_tol=1e-4)

time_evolution_plot(Path("steel316_survival_test_extreme.hdf5"))