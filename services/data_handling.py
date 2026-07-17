# ###################
# Ian Janes
# Prof. Don Lipkin
# Data handling service file
# ###################

from h5py import Dataset, File
from numpy.typing import NDArray
from pathlib import Path
from typing import cast, Tuple
import numpy as np

DIRECTORY = "Data/"
FORBIDDEN = "<>:\"|?*"
CHOP_CONV_WINDOW = 5
CHOP_CONV_TOL = 1e-2
CYCLE_WINDOW = (-0.1, 0.9)


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
        self._extrema: NDArray[np.float64] | None

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
    def extrema(self) -> NDArray[np.float64]:

        if self._extrema is None:
            raise RuntimeError("Data has not been loaded.")
        return self._extrema


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


    def _set_extrema(self):

        if (self._times is None) or (self._temps is None):
            raise ValueError("Data has not been loaded.")
        extrema: list[NDArray[np.float64]] = []
        max_time: float = max(self._times)
        cycles = int(max_time // self._period)
        for i in range(cycles):
            tmin = max(0., (i+CYCLE_WINDOW[0])*self._period)
            imin = np.searchsorted(self._times, tmin, side="left")
            tmax = min(max_time, (i+CYCLE_WINDOW[1])*self._period)
            imax = np.searchsorted(self._times, tmax, side="right")
            temps = self._temps[imin:imax]

            n_points = temps.shape[1]
            cycle_extrema = np.empty((n_points, 4), dtype=np.float64)
            for j in range(n_points):
                col = temps[:, j]

                # Minimum
                min_mask = ((col[1:-1] < col[:-2]) & (col[1:-1] < col[2:]))
                local = np.where(min_mask)[0] + 1
                if len(local):
                    min_indx = local[0]
                else:
                    min_indx = np.argmin(col)
                
                # Maximum
                max_mask = ((col[1:-1] > col[:-2]) & (col[1:-1] > col[2:]))
                local = np.where(max_mask)[0] + 1
                if len(local):
                    max_indx = local[0]
                else:
                    max_indx = np.argmax(col)
                
                cycle_extrema[j] = (
                    self._times[imin + min_indx],
                    col[min_indx],
                    self._times[imin + max_indx],
                    col[max_indx]
                )
            extrema.append(cycle_extrema)
        self._extrema = np.array(extrema)


    def _find_chop_steady_index(self) -> Tuple[int, int]:

        if (self._extrema is None) or (self._times is None):
            raise RuntimeError("No data has been loaded.")
        n: int = self._extrema.shape[0]
        if n < CHOP_CONV_WINDOW + 1:
            raise ValueError(f"At least {CHOP_CONV_WINDOW+1} cycles are needed for convergence testing. Current data contains only {n}.")
        for i in range(CHOP_CONV_WINDOW, n):
            prev = self._extrema[i-CHOP_CONV_WINDOW:i, :, [1, 3]].mean(axis=1)
            curr = self._extrema[i-CHOP_CONV_WINDOW+1:i+1, :, [1, 3]].mean(axis=1)
            if np.all(np.abs(curr - prev) <= CHOP_CONV_TOL):
                true_i = cast(int, np.searchsorted(self._times, self._extrema[i, 0, 0], side="right"))
                return i, true_i
        raise RuntimeError("Chop simulation did not reach steady-state. Either run simulation longer or lower chop convergence tolerance constant.")


    ########################################
    # Public Methods
    ########################################


    def initialize_storage(self, resolution: tuple[int, ...], metadata: dict[str, int | float | NDArray[np.float64]], push_overwrite: bool=False):

        """
        Preps for storage of simulation data.

        :param resolution: A tuple of integers corresponding to the spatial resolution of the temperature data.
        :param metadata: A dictionary mapping simulation inputs to attribute names.
        :param push_overwrite: Automatically force overwrite for batch solving. Default is False.
        """

        if self._filepath.exists() and not push_overwrite:
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
            self._period = cast(float, f.attrs["period"])
            self._set_extrema()
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
    

    def report_final_avg_temp(self, steady_index: int=-1):

        """
        Reports the final average temperature of the system. Only makes sense for systems that reached steady-state.

        :param steady_index:
        """

        if self._temps is None:
            raise ValueError("Data has not been loaded.")
        avg_temp: float = np.average(self._temps[steady_index:])
        print(f"Final average temperature of the system was {avg_temp:.3f}K.")


    def report_lag_time(self):

        if self._extrema is None:
            raise ValueError("Data has not been loaded.")
        index, abs_index = self._find_chop_steady_index()
        self.report_final_avg_temp(abs_index)
        f_min_time, f_min_temp, f_max_time, f_max_temp = tuple(self._extrema[index, 0])
        r_min_time, r_min_temp, r_max_time, r_max_temp = tuple(self._extrema[index, -1])
        print(f"The front temperature fell {(f_max_temp-f_min_temp):.3f}K from {f_max_temp:.3f}K at t={f_max_time:.3f}s to {f_min_temp:.3f}K at t={f_min_time:.3f}s over {(f_min_time-f_max_time):.3f}s.")
        print(f"The rear temperature fell {(r_max_temp-r_min_temp):.3f}K from {r_max_temp:.3f}K at t={r_max_time:.3f}s to {r_min_temp:.3f}K at t={r_min_time:.3f}s over {(r_min_time-r_max_time):.3f}s.")
        print(f"The max temperatures lagged by {(r_max_time-f_max_time):.3f}s and the min temperatures by {(r_min_time-f_min_time):.3f}s.")


    def extract_data(self, entry_row: NDArray[np.float64]):

        """
        Fills a numpy array row with particular data from the simulation.

        :param entry_row: The 1D numpy array reference to fill out.
        """

        if len(entry_row.shape) != 1:
            raise ValueError(f"Inputted array must one-dimensional, but passed array has shape {entry_row.shape}.")
        if (self._extrema is None) or (self._temps is None):
            raise RuntimeError("Data has not been loaded.")
        
        entry_row[1] = self._cond
        entry_row[2] = self._diff
        entry_row[3] = np.max(self._htcs[:, 1])
        entry_row[4] = self._length
        entry_row[5] = np.max(self._gas_temps[:, 1])
        entry_row[6] = self._period
        entry_row[7] = self._htcs[2, 0] # Chop duration

        cycle_index, abs_index = self._find_chop_steady_index()
        entry_row[8] = np.average(self._temps[abs_index:])
        entry_row[9], entry_row[10], entry_row[11], entry_row[12] = tuple(self._extrema[cycle_index, 0])
        entry_row[13], entry_row[14], entry_row[15], entry_row[16] = tuple(self._extrema[cycle_index, -1])


    def close(self):

        if self._file is not None:
            self._file.close()
            self._file = None