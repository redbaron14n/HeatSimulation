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

DIRECTORY = "Data/"
FORBIDDEN = "<>:\"|?*"


class DataHandler():

    def __init__(self, filename: str):

        """
        :param filename: The name of the .hdf5 file to save or load to.
        """

        self.filepath = filename

        self._times: NDArray[np.float64] | None = None
        self._temps: NDArray[np.float64] | None = None
        self._init_temps: NDArray[np.float64] | None = None
        self._length: float | None = None

        self._file: File | None = None


    ########################################
    # Getters and Setters
    ########################################


    @property
    def filepath(self) -> Path:

        """
        :return: The filepath of the data as a Path object.
        """

        return self._filepath
    

    @filepath.setter
    def filepath(self, filename: str):

        if any(c in filename for c in FORBIDDEN):
            raise ValueError(f"Filename {filename} contains at least one forbidden character {FORBIDDEN} .")
        path = Path(f"{DIRECTORY}{filename}")
        if path.suffix == "":
            path = Path(f"{DIRECTORY}{filename}.hdf5")
        elif len(path.suffixes) > 1:
            raise ValueError(f"Filename {filename} contains multiple extensions.")
        elif path.suffix != ".hdf5":
            raise ValueError(f"File extension must be '.hdf5'.")
        self._filepath = path


    @property
    def times(self) -> NDArray[np.float64]:

        if self._times is None:
            raise ValueError("No data has been loaded.")
        return self._times
    

    @property
    def temps(self) -> NDArray[np.float64]:

        if self._temps is None:
            raise ValueError("No data has been loaded.")
        return self._temps
    

    @property
    def init_temps(self) -> NDArray[np.float64]:

        if self._init_temps is None:
            raise ValueError("No data has been loaded.")
        return self._init_temps
    

    @init_temps.setter
    def init_temps(self, temps: NDArray[np.float64]):

        if self._file is None:
            raise RuntimeError("File is not open.")
        self._init_temps = temps
        self._file.attrs["init_temps"] = temps
    

    @property
    def length(self) -> float:

        if self._length is None:
            raise ValueError("No data has been loaded.")
        return self._length


    ########################################
    # Public Methods
    ########################################


    def initialize_storage(self, resolution: tuple[int, ...], metadata: dict[str, int | float | NDArray[np.float64]]):

        """
        Preps for storage of simulation data.

        :param resolution: A tuple of integers corresponding to the spatial resolution of the temperature data.
        :param metadata: A dictionary mapping simulation inputs to attribute names.
        """

        self._file = File(str(self._filepath), "w")
        self._file.create_dataset(
            name="times",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64
        )
        self._file.create_dataset(
            name="temps",
            shape=(0,)+resolution,
            maxshape=(None,)+resolution,
            dtype=np.float64,
            compression="gzip"
        )
        for name, val in metadata.items():
            self._file.attrs[name] = val
        self._snapshot_count: int = 0


    def load_data(self):

        """
        Loads the stored data from the file into the object.
        """

        print(f"Loading temperature data from {self._filepath}...")
        with File(str(self._filepath), "r") as f:
            self._times = np.array(f["times"], dtype=np.float64)
            self._temps = np.array(f["temps"], dtype=np.float64)
            self._dim = cast(int, f.attrs["dim"])
            self._dx = cast(float, f.attrs["dx"])
            self._dt = cast(float, f.attrs["dt"])
            self._length = cast(float, f.attrs["length"])
            self._emis = np.array(f.attrs["emis"], dtype=np.float64)
            self._htcs = np.array(f.attrs["htcs"], dtype=np.float64)
            self._gas_temps = np.array(f.attrs["gas_temps"], dtype=np.float64)
            self._diff = cast(float, f.attrs["diff"])
            self._cond = cast(float, f.attrs["cond"])
            self._temp_ambient = cast(float, f.attrs["temp_ambient"])
            self._init_temps = np.array(f.attrs["init_temps"], dtype=np.float64)
        print("Successfully loaded data.")


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

        self._snapshot_count = new_count


    def close(self):

        if self._file is not None:
            self._file.close()
            self._file = None