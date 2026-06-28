# ###################
# Ian Janes
# Prof. Don Lipkin
# Data handling service file
# ###################

from h5py import File
from numpy.typing import NDArray
from pathlib import Path
import numpy as np

FORBIDDEN = "<>:\"/\\|?*"
REQ_DATA = {"times", "temps", "emis", "htcs", "gas_temps"}
REQ_ATTR = {"dim", "dx", "dt", "length", "diff", "cond", "temp_ambient"}
REQ_2D_ATTR = {"dr", "cp", "dens"}


class DataHandler():

    def __init__(self, filename: str, load: bool = True):

        """
        :param filename: The name of the .hdf5 file to save to or load from.
        :param load: Whether the handler should bother loading data from the named .hdf5 file, if it exists. Defaults to True, but may be set to False if only saving or overwriting is intended.
        """

        self.filepath = filename
        self._set_data(load)


    ########################################
    # Getters and Setters
    ########################################


    @property
    def filepath(self) -> Path:

        """
        :return: The filepath of the data.
        """

        return self._filepath
    

    @filepath.setter
    def filepath(self, filename: str):

        if any(c in filename for c in FORBIDDEN):
            raise ValueError(f"Filename {filename} contains at least one forbidden character {FORBIDDEN} .")
        path = Path(f"Data/{filename}")
        if path.suffix == "":
            path = Path(f"Data/{filename}.hdf5")
        elif len(path.suffixes) > 1:
            raise ValueError(f"Filename {filename} contains multiple extensions.")
        elif path.suffix != ".hdf5":
            raise ValueError(f"File extension must be '.hdf5'.")
        self._filepath = path


    def _set_data(self, load: bool):

        if not(self._filepath.exists() and load):
            return
        print(f"Loading temperature data from {self._filepath}...")
        with File(str(self._filepath), "r") as f:
            self._times = np.array(f["times"], dtype=np.float64)
            self._temps = np.array(f["temps"], dtype=np.float64)
            self._emis = np.array(f["emis"], dtype=np.float64)
            self._htcs = np.array(f["htcs"], dtype=np.float64)
            self._gas_temps = np.array(f["gas_temps"], dtype=np.float64)
            self._dim = int(f.attrs["dim"]) # Don't know how to make this go away atm
            self._dx = float(f.attrs["dx"])
            self._dt = float(f.attrs["dt"])
            self._length = float(f.attrs["length"])
            self._diff = float(f.attrs["diff"])
            self._cond = float(f.attrs["cond"])
            self._temp_ambient = float(f.attrs["temp_ambient"])
            if self._dim == 2:
                self._dr = float(f.attrs["dr"])
                self._cphc = float(f.attrs["cphc"])
                self._dens = float(f.attrs["dens"])
        print("Successfully loaded data.")


    ########################################
    # Public Methods
    ########################################

    
    def save_1d_data(
            self,
            times: NDArray[np.float64],
            temps: NDArray[np.float64],
            emis: NDArray[np.float64],
            htcs: NDArray[np.float64],
            gas_temps: NDArray[np.float64],
            dim: int,
            dx: float,
            dt: float,
            length: float,
            diff: float,
            cond: float,
            temp_ambient: float
    ):    
        
        with File(str(self._filepath), "w") as f:
            f.create_dataset("times", data=times)
            f.create_dataset("temps", data=temps, compression="gzip")
            f.create_dataset("emis", data=emis)
            f.create_dataset("htcs", data=htcs)
            f.create_dataset("gas_temps", data=gas_temps)
            f.attrs["dim"] = dim
            f.attrs["dx"] = dx
            f.attrs["dt"] = dt
            f.attrs["length"] = length
            f.attrs["diff"] = diff
            f.attrs["cond"] = cond
            f.attrs["temp_ambient"] = temp_ambient


    def save_2d_data(
            self,
            times: NDArray[np.float64],
            temps: NDArray[np.float64],
            emis: NDArray[np.float64],
            htcs: NDArray[np.float64],
            gas_temps: NDArray[np.float64],
            dim: int,
            dx: float,
            dr: float,
            dt: float,
            length: float,
            diff: float,
            cond: float,
            temp_ambient: float,
            cphc: float,
            dens: float,
    ):    
        
        with File(str(self._filepath), "w") as f:
            f.create_dataset("times", data=times)
            f.create_dataset("temps", data=temps, compression="gzip")
            f.create_dataset("emis", data=emis)
            f.create_dataset("htcs", data=htcs)
            f.create_dataset("gas_temps", data=gas_temps)
            f.attrs["dim"] = dim
            f.attrs["dx"] = dx
            f.attrs["dr"] = dr
            f.attrs["dt"] = dt
            f.attrs["length"] = length
            f.attrs["diff"] = diff
            f.attrs["cond"] = cond
            f.attrs["temp_ambient"] = temp_ambient
            f.attrs["cphc"] = cphc
            f.attrs["dens"] = dens