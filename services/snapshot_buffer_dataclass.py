from dataclasses import dataclass
from numpy.typing import NDArray
from numpy import float64

@dataclass
class SnapshotBuffer:
    times: list[float]
    temps: list[NDArray[float64]]
    size: int
    last_saved: NDArray[float64]