from numpy import float64
from numpy.typing import NDArray
from typing import NamedTuple


class LookupTables(NamedTuple):

    gas_temp_lookup: tuple[NDArray[float64], NDArray[float64]]
    htcs_lookup: tuple[NDArray[float64], NDArray[float64]]
    emis_lookup: NDArray[float64]