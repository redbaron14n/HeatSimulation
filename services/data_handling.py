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
CHOP_CONV_TOL = 1e-3


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
        self._minima: list[NDArray[np.float64]] | None = None
        self._maxima: list[NDArray[np.float64]] | None = None
        self._chops: NDArray[np.float64] | None = None

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
    

    @property
    def minima(self) -> list[NDArray[np.float64]]:

        if self._minima is None:
            raise ValueError("No data has been loaded.")
        return self._minima
    

    @property
    def maxima(self) -> list[NDArray[np.float64]]:

        if self._maxima is None:
            raise ValueError("No data has been loaded.")
        return self._maxima


    ########################################
    # Private Methods
    ########################################


    def _confirm_overwrite(self):

        invalid = True
        while invalid:
            invalid = False
            response = input(f"Pre-existing file {str(self._filepath)} detected. Confirm overwrite (y/n)? ")
            if response.lower() == "n":
                raise RuntimeError("Simulation aborted by user.")
            elif response.lower() not in ["y", "n"]:
                invalid = True
                print("Unknown input. Try again.")


    def _set_minmax(self):

        if (self._times is None) or (self._temps is None):
            raise ValueError("Data has not been loaded.")
        self._minima = []
        self._min_indices: list[NDArray[np.int64]] = []
        self._maxima = []
        self._max_indices: list[NDArray[np.int64]] = []
        for i in range(self._temps.shape[1]):
            temps = self._temps[:, i]

            min_mask = (temps[1:-1] < temps[:-2]) & (temps[1:-1] < temps[2:])
            min_indices = (np.where(min_mask)[0]+1).astype(np.int64)
            self._min_indices.append(min_indices)
            min_temps = temps[min_indices]
            min_times = self._times[min_indices]
            self._minima.append(np.column_stack((min_times, min_temps)))

            max_mask = (temps[1:-1] > temps[:-2]) & (temps[1:-1] > temps[2:])
            max_indices = (np.where(max_mask)[0]+1).astype(np.int64)
            self._max_indices.append(max_indices)
            max_temps = temps[max_indices]
            max_times = self._times[max_indices]
            self._maxima.append(np.column_stack((max_times, max_temps)))


    def _set_chops(self):

        if (self._maxima is None) or (self._minima is None) or (self._temps is None):
            raise ValueError("No data has been loaded.")
        num_chop = min(self._maxima[0].shape[0]+1, self._maxima[-1].shape[0], self._minima[0].shape[0], self._minima[-1].shape[0])
        self._chops = np.empty((num_chop, 4, 2), dtype=np.float64)
        indx = 0
        while indx < num_chop:
            if indx == 0:
                self._chops[0, 0] = np.array([0., self._temps[0, 0]])
            else:
                self._chops[indx, 0] = self._maxima[0][indx-1]
            self._chops[indx, 1] = self._maxima[-1][indx]
            self._chops[indx, 2] = self._minima[0][indx]
            self._chops[indx, 3] = self._minima[-1][indx]
            indx += 1


    def _find_chop_steady_index(self) -> int:

        if self._chops is None:
            raise ValueError("No data has been loaded.")
        if self._chops.shape[0] == 1:
            raise ValueError("Only one chop cycle is detected. No information on steady-state.")
        for i in range(1, self._chops.shape[0]):
            if (
                abs(self._chops[i, 0, 1] - self._chops[i-1, 0, 1]) <= CHOP_CONV_TOL and
                abs(self._chops[i, 1, 1] - self._chops[i-1, 1, 1]) <= CHOP_CONV_TOL and
                abs(self._chops[i, 2, 1] - self._chops[i-1, 2, 1]) <= CHOP_CONV_TOL and
                abs(self._chops[i, 3, 1] - self._chops[i-1, 3, 1]) <= CHOP_CONV_TOL
            ):
                return i
        raise RuntimeError("Chop simulation did not reach steady-state. Either run simulation longer or lower chop convergence tolerance constant.")


    ########################################
    # Public Methods
    ########################################


    def initialize_storage(self, resolution: tuple[int, ...], metadata: dict[str, int | float | NDArray[np.float64]]):

        """
        Preps for storage of simulation data.

        :param resolution: A tuple of integers corresponding to the spatial resolution of the temperature data.
        :param metadata: A dictionary mapping simulation inputs to attribute names.
        """

        if self._filepath.exists():
            self._confirm_overwrite()
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
            self._set_minmax()
            self._set_chops()
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


    def report_convergence_time(self, chop: bool=False):

        """
        Reports the simulated time taken for the system to reach steady-state.

        :param chop: Whether the simulation was running with chopped heat flow rather than steady. Default is False.
        """

        if self._times is None:
            raise ValueError("Data has not been loaded.")
        if not chop:
            print(f"Simulation converged in {self._times[-1]:.3f}s.")
            return
        raise NotImplementedError("Haven't implemented convergence time for chopper simulations.")
    

    def report_final_avg_temp(self):

        """
        Reports the final average temperature of the system. Only makes sense for systems that reached steady-state.
        """

        if self._temps is None:
            raise ValueError("Data has not been loaded.")
        avg_temp: float = np.average(self._temps[-1])
        print(f"Final average temperature of the system was {avg_temp:.3f}K.")


    def report_lag_time(self):

        if self._chops is None:
            raise ValueError("Data has not been loaded.")
        index = self._find_chop_steady_index()
        f_max_time, f_max_temp = self._chops[index, 0, 0], self._chops[index, 0, 1]
        r_max_time, r_max_temp = self._chops[index, 1, 0], self._chops[index, 1, 1]
        f_min_time, f_min_temp = self._chops[index, 2, 0], self._chops[index, 2, 1]
        r_min_time, r_min_temp = self._chops[index, 3, 0], self._chops[index, 3, 1]
        print(f"The front temperature fell {(f_max_temp-f_min_temp):.3f}K from {f_max_temp:.3f}K at t={f_max_time:.3f}s to {f_min_temp:.3f}K at t={f_min_time:.3f}s over {(f_min_time-f_max_time):.3f}s.")
        print(f"The rear temperature fell {(r_max_temp-r_min_temp):.3f}K from {r_max_temp:.3f}K at t={r_max_time:.3f}s to {r_min_temp:.3f}K at t={r_min_time:.3f}s over {(r_min_time-r_max_time):.3f}s.")
        print(f"The max temperatures lagged by {(r_max_time-f_max_time):.3f}s and the min temperatures by {(r_min_time-f_min_time):.3f}s.")


    def close(self):

        if self._file is not None:
            self._file.close()
            self._file = None