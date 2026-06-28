from domain.conductive_system_1D import ConductiveSystem1D
from domain.finite_diff_solver_1D import FiniteDiffSolver1D
import numpy as np


diff = 1.58e-4
cond = 40.
emis = np.array([[0.001, 0.5]], dtype=np.float64)
length = 0.0035

test_system = ConductiveSystem1D(diff, cond, emis, length)

x_res = 100
initial_temps = np.full(100, 298.15)
gas_temps = np.array([[0., 2500., 298.15]], dtype=np.float64)
htcs = np.array([[0., 100., 100.]], dtype=np.float64)
ambient_temp = 298.15

test_solver = FiniteDiffSolver1D(test_system, initial_temps, gas_temps, htcs, ambient_temp, x_res, min_sim_time=1.0)

test_solver.run_simulation("hdf5test.hdf5", conv_tol=1e-6, print_every=10000)