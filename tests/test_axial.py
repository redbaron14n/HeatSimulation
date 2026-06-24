from ConductiveSystem import ConductiveSystemAxial
from FiniteDiffSolver import FiniteDiffSolverAxial
import math as m
import numpy as np

emis_a = np.array([[298.15, 0.21], [383.15, 0.33]])
htcs_a = (50.0, 50.0)

test_system_a = ConductiveSystemAxial(
    diff = 1.58e-4,
    cond = 400.0,
    emis = emis_a,
    htcs = htcs_a,
    dens = 8960.0,
    cphc = 418.0,
    length = 0.01,
    radius = 0.05
)

x_res_a = 10
r_res_a = 7
init_temps_a = np.full((r_res_a, x_res_a), 298.15)
torch_fluxs_a = np.array([[0.0, 0.5e6, 0.0]])
gas_temps_a = np.array([[0.0, 2000.0, 298.15]])

test_solver_a = FiniteDiffSolverAxial(
    system = test_system_a,
    x_res = x_res_a,
    r_res = r_res_a,
    initial_temps = init_temps_a,
    torch_fluxs = torch_fluxs_a,
    gas_temps = gas_temps_a
)


def test_top_left_corner_calculator():

    temp = test_solver_a._calc_top_left_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.5e6)
    assert m.isclose(temp, 299.7723833159169)


def test_top_right_corner_calculator():

    temp = test_solver_a._calc_top_right_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.0)
    assert m.isclose(temp, 298.41316287416487)


def test_bottom_left_corner_calculator():

    temp = test_solver_a._calc_bottom_left_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.5e6)
    assert m.isclose(temp, 299.71923830845077)


def test_bottom_right_corner_calculator():

    temp = test_solver_a._calc_bottom_right_corner(0.21, 50.0, 2000.0, 298.15, 298.15, 0.0)
    assert m.isclose(temp, 298.3950459043153)


def test_left_edge_calculator():

    pass


def test_top_edge_calculator():

    temps_rtop2 = init_temps_a[-2:, :] # Top two rows (printed as bottom)
    temps_rtop2[1, 0] = 300
    h_data = np.full(x_res_a-2, 50.0, dtype=np.float64)
    emis_data = np.full_like(h_data, 0.21)
    gas_temps = np.full_like(h_data, 298.15)
    edge_temps = test_solver_a._calc_top_edge(temps_rtop2, h_data, emis_data, gas_temps)
    expected = np.array([298.25659472, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.16516312]) # These are wrong - need to be generated using array of only 298.15 values, not with updated corners
    np.testing.assert_allclose(edge_temps, expected, rtol=0.0, atol=1e-8)


init_temps_a[-1, 0] = 299.7723833159169
init_temps_a[-1, -1] = 298.41316287416487
init_temps_a[0, 0] = 299.71923830845077
init_temps_a[0, -1] = 298.3950459043153