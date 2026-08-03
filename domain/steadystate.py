# ###################
# Ian Janes
# Prof. Don Lipkin
# Steadystate Cycle Appoximation file
# ###################

from domain.domain_constants import BOLTZ, H_RAT_C0, H_RAT_C1
from numpy.typing import NDArray
from scipy.optimize import least_squares
from typing import cast
import numpy as np


def _calc_boundary_resids(
        guess: NDArray[np.float64],
        convh: float,
        convn: float,
        rad: float,
        length: float,
        gas: float,
        ambi: float,
        ambi4: float
) -> NDArray[np.float64]:

    slope = (guess[1] - guess[0]) / length
    resid0 = slope - (convh * (guess[0] - gas)) - (rad * (guess[0]**4 - ambi4))
    resid1 = slope + (convn * (guess[1] - ambi)) + (rad * (guess[1]**4 - ambi4))
    return np.asarray([resid0, resid1], dtype=np.float64)


def _calc_eff_htc(
        hheat: float,
        dur: float,
        per: float
) -> float:

    duty1m = 1. - dur/per
    hrat = H_RAT_C0 + H_RAT_C1*duty1m
    return hheat*hrat


def find_steadystate(
        cond: float,
        emis: float,
        hheat: float,
        hnat: float,
        length: float,
        gas_temp: float,
        ambi_temp: float,
        dur: float,
        per: float,
        points: int=25
) -> NDArray[np.float64]:
    
    hheat = _calc_eff_htc(hheat, dur, per)
    convh = hheat / cond
    convn = hnat / cond
    rad = emis * BOLTZ / cond
    ambi4 = ambi_temp**4
    results = least_squares(
        _calc_boundary_resids,
        np.array((gas_temp, ambi_temp)),
        args=(convh, convn, rad, length, gas_temp, ambi_temp, ambi4)
    )
    f_temp, r_temp = cast(tuple[float, float], results.x)
    temps = np.linspace(f_temp, r_temp, points, dtype=np.float64)
    return temps