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


h_data = np.full(x_res_a-2, 50.0, dtype=np.float64)
gas_temps_data = np.full_like(h_data, 2000.0)
side_emis_data = np.full(5, 0.21, dtype=np.float64)


def test_left_edge_calculator():

    temps_c01 = init_temps_a[:, 0:2]
    edge_temps = test_solver_a._calc_left_edge(temps_c01, 50.0, side_emis_data, 2000.0, 0.5e6)
    expected = np.full(5, 298.33728797, dtype=np.float64)
    np.testing.assert_allclose(edge_temps, expected, rtol=0.0, atol=1e-8)


def test_top_edge_calculator():

    temps_rtop2 = init_temps_a[-2:, :] # Top two rows (numpy prints upside-down)
    top_emis_data = np.full_like(h_data, 0.21)
    edge_temps = test_solver_a._calc_top_edge(temps_rtop2, h_data, top_emis_data, gas_temps_data)
    expected = np.full(8, 298.15393445, dtype=np.float64)
    np.testing.assert_allclose(edge_temps, expected, rtol=0.0, atol=1e-8)


def test_right_edge_calculator():

    temps_ctop2 = init_temps_a[:, -2:]
    edge_temps = test_solver_a._calc_right_edge(temps_ctop2, 50.0, side_emis_data, 2000.0, 0.0)
    expected = np.full(5, 298.17723809, dtype=np.float64)
    np.testing.assert_allclose(edge_temps, expected, rtol=0.0, atol=1e-8)


def test_bottom_edge_calculator():

    middle_temps = init_temps_a[:2, :]
    edge_temps = test_solver_a._calc_bottom_edge(middle_temps)
    expected = np.full(8, 298.15, dtype=np.float64)
    np.testing.assert_allclose(edge_temps, expected, rtol=0.0, atol=1e-8)


def test_internal_calculator():

    interior_temps = test_solver_a._calc_internal_temps(init_temps_a)
    expected = np.full((5, 8), 298.15, dtype=np.float64)
    np.testing.assert_allclose(interior_temps, expected, rtol=0.0, atol=1e-8)