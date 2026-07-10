# ###################
# Ian Janes
# Prof. Don Lipkin
# Prefabricated settings config file
# ###################

import numpy as np


"""
HVAF Torch HTCS, T=2200K
10in: 1000W/m^2/K
12in: 600
15in: 300

HVOF Torch HTCS, T=3100K
3.2in: 1000W/m^2/K
6in: 250
9in: 100
12in: 50
"""

GAS_3100_CONSTANT_ARRAY = np.array([[0., 3100., 298.15]], dtype=np.float64)

GAS_2200_CONSTANT_ARRAY = np.array([[0., 2200., 298.15]], dtype=np.float64)

HTCS_600_CONSTANT_ARRAY = np.array([[0., 600., 10.]], dtype=np.float64)

HTCS_300_CONSTANT_ARRAY = np.array([[0., 300., 10.]], dtype=np.float64)

HTCS_600_10_5_300_ARRAY = np.array([ # 0.1s chop time, 0.05s transfer time, 3s period
    [0., 10., 10.],
    [0.1, 10., 10.],
    [0.15, 600., 10.],
    [2.95, 600., 10.]
], dtype=np.float64)

HTCS_300_10_5_300_ARRAY = np.array([
    [0., 10., 10.],
    [0.1, 10., 10.],
    [0.15, 300., 10.],
    [2.95, 300., 10.]
], dtype=np.float64)

COPPER = (9.21e-5, 350., np.array([[400., 0.15], [1000., 0.25]], dtype=np.float64)) # Diffusivity, conductivity, emissivities at temp

STEEL316 = (4.5e-4, 20., np.array([[298.15, 0.32], [750., 0.32], [1200., 0.62]], dtype=np.float64))

ZRB2 = (2e-5, 74., np.array([[298.15, 0.34]], dtype=np.float64))