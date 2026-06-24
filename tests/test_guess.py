import numpy as np

from ConductiveSystem import ConductiveSystemAxial
from FiniteDiffSolver import FiniteDiffSolverAxial

emis_a = np.array([[298.15, 0.21], [383.15, 0.33]])
htcs_a = (50.0, 50.0)

test_system_a = ConductiveSystemAxial(
    diff=1.58e-4,
    cond=400.0,
    emis=emis_a,
    htcs=htcs_a,
    length=0.01,
    radius=0.05,
)

x_res_a = 100
r_res_a = 25
init_temps_a = np.full((r_res_a, x_res_a), 298.15)
torch_fluxs_a = np.array([[0.0, 0.5e6, 0.0]])
gas_temps_a = np.array([[0.0, 2000.0, 298.15]])

test_solver_a = FiniteDiffSolverAxial(
    system=test_system_a,
    x_res=x_res_a,
    r_res=r_res_a,
    initial_temps=init_temps_a,
    torch_fluxs=torch_fluxs_a,
    gas_temps=gas_temps_a,
)


def test_finite_diff_solver_axial_initialization():


    assert test_solver_a is not None
    assert test_solver_a.system is test_system_a
    assert test_solver_a.x_resolution == x_res_a
    assert test_solver_a.r_resolution == r_res_a
    assert np.array_equal(test_solver_a.initial_temperatures, init_temps_a)
    assert np.array_equal(test_solver_a.torch_heat_fluxes, torch_fluxs_a)
    assert np.array_equal(test_solver_a.gas_temperatures, gas_temps_a)
    assert test_solver_a.ambient_temperature == 298.15
    assert test_solver_a.min_simulation_time == 0.0
    assert test_solver_a.max_simulation_time == 100.0
    assert test_solver_a.diffusion_number == 0.1
    assert test_solver_a.env_cutoff == 0.0
    assert test_solver_a.convergence_tol == 1e-6