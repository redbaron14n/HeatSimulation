# ###################
# Ian Janes
# Prof. Don Lipkin
# Data handling service file
# ###################

from h5py import Dataset, File
from numpy.typing import NDArray
from pathlib import Path
from typing import cast
import numpy as np

FORBIDDEN = "<>:\"/\\|?*"
REQ_DATA = {"times", "temps"}
REQ_ATTR = {"dim", "dx", "dt", "length", "emis", "htcs", "gas_temps", "diff", "cond", "temp_ambient"}
REQ_2D_ATTR = {"dr", "cp", "dens"}


class DataHandler():

    def __init__(self, filename: str, load: bool = True):

        """
        :param filename: The name of the .hdf5 file to save to or load from.
        :param load: Whether the handler should bother loading data from the named .hdf5 file, if it exists. Defaults to True, but may be set to False if only saving or overwriting is intended.
        """

        self.filepath = filename

        self._loaded_times: NDArray[np.float64] | None = None
        self._loaded_temps: NDArray[np.float64] | None = None

        self._attrs: dict[str, int | float | NDArray[np.float64]] = {}

        self._file: File | None = None
        self._snapshot_count: int = 0

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


    ########################################
    # Private Methods
    ########################################


    def _set_data(self, load: bool):

        if not(self._filepath.exists() and load):
            return
        print(f"Loading temperature data from {self._filepath}...")
        with File(str(self._filepath), "r") as f:
            self._loaded_times = np.array(f["times"], dtype=np.float64)
            self._loaded_temps = np.array(f["temps"], dtype=np.float64)
            self._attrs = {
                name: cast(int | float | NDArray[np.float64], value)
                  for name, value in f.attrs.items()
            }
        print("Successfully loaded data.")


    ########################################
    # Public Methods
    ########################################

    
    def save(self, times: NDArray[np.float64], temps: NDArray[np.float64], attrs: dict[str, int | float | NDArray[np.float64]]):

        with File(str(self._filepath), "w") as f:
            f.create_dataset("times", data=times)
            f.create_dataset("temps", data=temps, compression="gzip")
            for name, val in attrs.items():
                f.attrs[name] = val


    def initialize_storage(self, resolution: tuple[int, ...], metadata: dict[str, int | float | NDArray[np.float64]]):

        """
        Preps for storage of simulation data.

        :param resolution: A tuple of integers corresponding to the spatial resolution of the temperature data.
        :param metadata: A dictionary mapping simulation inputs to attribute names.
        """

        self._file = File(str(self._filepath), "w")
        self._times = self._file.create_dataset("times", shape=(0,), maxshape=(None,), dtype=np.float64)
        self._temps = self._file.create_dataset(
            "temps",
            shape=(0,)+resolution,
            maxshape=(None,)+resolution,
            dtype=np.float64,
            compression="gzip"
        )
        for name, val in metadata.items():
            self._file.attrs[name] = val
        self._snapshot_count: int = 0


    def append_snapshots(self, times: NDArray[np.float64], temps: NDArray[np.float64]):

        """
        Appends a number of snapshots to the stored data.

        :param times: A 1D array of floats.
        :param temps: A multidimensional array of floats where the first axis has the same length as 'times'.
        """

        if self._file is None:
            raise RuntimeError("Storage has not been initialized.")
        n_new = times.shape[0]
        if temps.shape[0] != n_new:
            raise ValueError("Times and temperatures contain differing numbers of snapshots.")
        old_count = self._snapshot_count
        new_count = old_count + n_new

        times_ds = cast(Dataset, self._file["times"])
        times_ds.resize(new_count, 0)
        times_ds[old_count:new_count] = times

        temps_ds = cast(Dataset, self._file["temps"])
        temps_ds.resize(new_count, 0)
        temps_ds[old_count:new_count] = temps


    def close(self):

        if self._file is not None:
            self._file.close()
            self._file = None


    @property
    def times(self) -> NDArray[np.float64]:

        if self._loaded_times is None:
            raise ValueError("No data has been loaded.")
        return self._loaded_times
    

    @property
    def temps(self) -> NDArray[np.float64]:

        if self._loaded_temps is None:
            raise ValueError("No data has been loaded.")
        return self._loaded_temps