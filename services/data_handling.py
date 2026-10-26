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
CHOP_CONV_TOL = 5e-3
CYCLE_WINDOW = (-0.1, 0.9)
TEMP_THRESHOLD = 1. # Minimum difference between temperatures to be considered viable


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


    # def _set_extrema(self):

    #     if (self._times is None) or (self._temps is None):
    #         raise ValueError("Data has not been loaded.")
    #     extrema: list[NDArray[np.float64]] = []
    #     max_time: float = max(self._times)
    #     cycles = int(max_time // self._period)
    #     max_time_index: int = self._temps.shape[0] - 1
    #     for i in range(cycles):

    #         # tmin = max(0., (i+CYCLE_WINDOW[0])*self._period)
    #         tmin = i*self._period
    #         imin = max(np.searchsorted(self._times, tmin, side="left") - 5, 0)
    #         # tmax = min(max_time, (i+CYCLE_WINDOW[1])*self._period)
    #         tmax = (i+1)*self._period
    #         imax = min(np.searchsorted(self._times, tmax, side="right") + 5, max_time_index)
    #         temps = self._temps[imin:imax]

    #         n_points = temps.shape[1] # x_res
    #         cycle_extrema = np.empty((n_points, 6), dtype=np.float64)
    #         for j in range(n_points):
    #             cycle = temps[:, j] # temp evolution within cycle for given depth

    #             if (i >= 75) and (j == n_points-1):
    #                 pass

    #             # Minimum
    #             min_mask = ((cycle[1:-1] < cycle[:-2]) & (cycle[1:-1] < cycle[2:]))
    #             local = np.where(min_mask)[0] + 1 # finds index of first local minimum
    #             if len(local):
    #                 if (i == 75) and (j == n_points-1):
    #                     index_value_array = np.column_stack((local, cycle[local]))
    #                     print(index_value_array)
    #                     print(local[cycle[local].argmin()])
    #                 min_indx = local[cycle[local].argmin()]
    #             else:
    #                 min_indx = np.argmin(cycle)
                
    #             # Maximum
    #             max_mask = ((cycle[1:-1] > cycle[:-2]) & (cycle[1:-1] > cycle[2:]))
    #             local = np.where(max_mask)[0] + 1
    #             if len(local):
    #                 max_indx = local[cycle[local].argmax()]
    #             else:
    #                 max_indx = np.argmax(cycle)

    #             # Inflection
    #             lo = min(min_indx, max_indx)
    #             hi = max(min_indx, max_indx)
    #             if hi == lo:
    #                 inf_time = np.nan
    #                 inf_temp = np.nan
    #             else:
    #                 focus = cycle[lo:hi+1]
    #                 dt = np.diff(self._times[imin+lo:imin+hi+1])
    #                 dTdt = np.diff(focus) / dt
    #                 rel_indx = np.argmax(np.abs(dTdt))
    #                 inf_indx = lo + rel_indx + 1
    #                 inf_time = self._times[imin + inf_indx]
    #                 inf_temp = cycle[inf_indx]
                
    #             cycle_extrema[j] = (
    #                 self._times[imin + min_indx],
    #                 cycle[min_indx],
    #                 inf_time,
    #                 inf_temp,
    #                 self._times[imin + max_indx],
    #                 cycle[max_indx]
    #             )
    #         extrema.append(cycle_extrema)
    #     self._extrema = np.array(extrema)


    def _set_extrema(self):

        if (self._times is None) or (self._temps is None):
            raise ValueError("Data has not been loaded")

        extrema: list[NDArray[np.float64]] = []
        max_time = max(self._times)
        cycles = int(max_time // self._period)
        max_time_index = self._times.shape[0] - 1
        n_points = self._temps.shape[1]
        for i in range(1, cycles):

            cycle_tmin = i*self._period
            cycle_imin = max(np.searchsorted(self._times, cycle_tmin, side="left") - 5, 0)
            cycle_tmax = (i+0.9)*self._period
            cycle_imax = min(np.searchsorted(self._times, cycle_tmax, side="right") + 5, max_time_index)

            cycle_temps = self._temps[cycle_imin:cycle_imax] # 2D array
            cycle_extrema = np.empty((n_points, 8), dtype=np.float64)

            for x in range(n_points): # Identify maximum at each x-value

                cycle = cycle_temps[:, x] # 1D array

                # Maximum
                max_mask = ((cycle[1:-1] > cycle[:-2]) & (cycle[1:-1] > cycle[2:]))
                local_max = np.where(max_mask)[0] + 1 # +1 to account for slicing in previous step
                if len(local_max):
                    max_index = local_max[cycle[local_max].argmax()]
                else:
                    max_index = np.argmax(cycle)
                local_max_time: float = self._times[cycle_imin + max_index]
                local_max_temp: float = cycle[max_index]

                # Refocus - relevant local minimum and inflection point will be between this local max and the next
                subcycle_tmin = local_max_time
                subcycle_imin = max(np.searchsorted(self._times, subcycle_tmin, side="left") - 5, 0)
                subcycle_tmax = local_max_time + self._period # This extends beyond current cycle, thus why self._times is called again
                subcycle_imax = min(np.searchsorted(self._times, subcycle_tmax, side="right") + 5, max_time_index)

                subcycle = self._temps[subcycle_imin:subcycle_imax, x]

                # Minimum
                min_mask = ((subcycle[1:-1] < subcycle[:-2]) & (subcycle[1:-1] < subcycle[2:]))
                local_min = np.where(min_mask)[0] + 1
                if len(local_min):
                    min_index = local_min[subcycle[local_min].argmin()]
                else:
                    min_index = np.argmin(subcycle)
                abs_min_index = subcycle_imin + min_index
                local_min_time: float = self._times[abs_min_index]
                local_min_temp: float = subcycle[min_index]

                # Inflection
                subcycle = subcycle[:min_index] # Prevents catching rising inflection
                if local_max_temp < local_min_temp + TEMP_THRESHOLD:
                    inf_time = inf_temp = -1.
                else:
                    dt = np.diff(self._times[subcycle_imin:abs_min_index])
                    dTdt = np.diff(subcycle) / dt
                    inf_index = np.argmax(np.absolute(dTdt))
                    inf_time: float = self._times[subcycle_imin + inf_index]
                    inf_temp = subcycle[inf_index]

                # Half-max
                if local_max_temp < local_min_temp + TEMP_THRESHOLD:
                    halfway_time = halfway_temp = -1.
                else:
                    halfway_temp = (local_max_temp + local_min_temp) / 2
                    mask = (subcycle <= halfway_temp)
                    halfway_index = np.argmax(mask)
                    halfway_time: float = self._times[subcycle_imin + halfway_index]
                
                cycle_extrema[x, 0] = local_min_time
                cycle_extrema[x, 1] = local_min_temp
                cycle_extrema[x, 2] = inf_time
                cycle_extrema[x, 3] = inf_temp
                cycle_extrema[x, 4] = halfway_time
                cycle_extrema[x, 5] = halfway_temp
                cycle_extrema[x, 6] = local_max_time
                cycle_extrema[x, 7] = local_max_temp

            extrema.append(cycle_extrema)
        self._extrema = np.array(extrema)


    def _find_chop_steady_index(self) -> Tuple[int, int]:

        if (self._times is None) or (self._temps is None):
            raise RuntimeError("No data has been loaded.")

        max_time = self._times[-1]
        n_cycles = int(max_time // self._period)

        if n_cycles < 2:
            raise RuntimeError("At least two complete cycles are required to determine steadystate.")

        for cycle in range(1, n_cycles):
            t0_prev = (cycle - 1) * self._period
            t1_prev = cycle * self._period
            t0_curr = cycle * self._period
            t1_curr = (cycle + 1) * self._period

            i0_prev = np.searchsorted(self._times, t0_prev, side="left")
            i1_prev = np.searchsorted(self._times, t1_prev, side="left")
            i0_curr = np.searchsorted(self._times, t0_curr, side="left")
            i1_curr = np.searchsorted(self._times, t1_curr, side="left")

            prev_times = self._times[i0_prev:i1_prev] - t0_prev
            prev_temps = self._temps[i0_prev:i1_prev]
            curr_times = self._times[i0_curr:i1_curr] - t0_curr
            curr_temps = self._temps[i0_curr:i1_curr]

            phase_grid = prev_times if len(prev_times) >= len(curr_times) else curr_times

            prev_interp = np.empty((len(phase_grid), self._temps.shape[1]))
            curr_interp = np.empty_like(prev_interp)

            for j in range(self._temps.shape[1]):
                prev_interp[:, j] = np.interp(phase_grid, prev_times, prev_temps[:, j])
                curr_interp[:, j] = np.interp(phase_grid, curr_times, curr_temps[:, j])

            rms = np.sqrt(np.mean((curr_interp - prev_interp)**2))
            if rms < CHOP_CONV_TOL:
                true_index = int(i0_curr)
                return cycle, true_index

        raise RuntimeError("Chop simulation did not reach steady-state.")


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
        f_min_time, f_min_temp, f_max_time, f_max_temp = tuple(self._extrema[index, 0, [0, 1, 6, 7]])
        r_min_time, r_min_temp, r_inf_time, r_half_time, r_max_time, r_max_temp = tuple(self._extrema[index, -1, [0, 1, 2, 4, 6, 7]])
        print(f"The front temperature fell {(f_max_temp-f_min_temp):.3f}K from {f_max_temp:.3f}K at t={f_max_time:.3f}s to {f_min_temp:.3f}K at t={f_min_time:.3f}s over {(f_min_time-f_max_time):.3f}s.")
        print(f"The rear temperature fell {(r_max_temp-r_min_temp):.3f}K from {r_max_temp:.3f}K at t={r_max_time:.3f}s to {r_min_temp:.3f}K at t={r_min_time:.3f}s over {(r_min_time-r_max_time):.3f}s.")
        print(f"The max temperatures lagged by {(r_max_time-f_max_time):.3f}s and the min temperatures by {(r_min_time-f_min_time):.3f}s.")
        if r_inf_time == -1.:
            print(f"No measurable inflection point.")
        else:
            print(f"The inflection point lagged by {(r_inf_time - f_max_time):.3f}s.")
        if r_half_time == -1.:
            print(f"No measurable halfway point.")
        else:
            print(f"The halfway point lagged by {(r_half_time - f_max_time):.3f}s.")


    def extract_data(self, entry_row: NDArray[np.float64]) -> bool:

        """
        Fills a numpy array row with particular data from the simulation.

        :param entry_row: The 1D numpy array reference to fill out.
        """

        if len(entry_row.shape) != 1:
            raise ValueError(f"Inputted array must one-dimensional, but passed array has shape {entry_row.shape}.")
        if (self._extrema is None) or (self._temps is None) or (self._times is None):
            raise RuntimeError("Data has not been loaded.")
        
        entry_row[2] = self._cond
        entry_row[3] = self._diff
        entry_row[4] = self._emis[0, 1]
        entry_row[5] = np.max(self._htcs[:, 1])
        entry_row[6] = self._htcs[0, 2]
        entry_row[7] = self._length
        entry_row[8] = np.max(self._gas_temps[:, 1])
        entry_row[9] = self._temp_ambient
        entry_row[10] = self._htcs[1, 0] # Chop duration
        entry_row[11] = self._period

        try:
            cycle_index, abs_index = self._find_chop_steady_index()
            entry_row[1] = 1
        except RuntimeError:
            cycle_index = abs_index = -1
            entry_row[1] = 0

        entry_row[12] = np.average(self._temps[abs_index:])

        try:
            entry_row[13:17] = tuple(self._extrema[cycle_index, 0, [0, 1, 6, 7]])

            f_max_time = self._extrema[cycle_index, 0, 4]
            f_max_time_i = cast(int, np.searchsorted(self._times, f_max_time))
            cor_r_temp = self._temps[f_max_time_i, -1]
            entry_row[17] = cor_r_temp # Temp at rear surface when front is at max

            entry_row[18:] = tuple(self._extrema[cycle_index, -1, [0, 1, 2, 4, 6, 7]])
        except IndexError:
            entry_row[13:] = -1
            return True
        return False


    def close(self):

        if self._file is not None:
            self._file.close()
            self._file = None