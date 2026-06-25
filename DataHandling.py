# ###################
# Ian Janes
# Prof. Don Lipkin
# Data Handling service file
# ###################

from numpy.typing import NDArray
from pathlib import Path
import numpy as np


def save_procedure(raw_temp_map: NDArray[np.float64], final_temps: NDArray[np.float64]):

    """
    Perform the save procedure for the simulation data and final temperatures.

    :param raw_temp_map: The raw temperature data from the simulation.
    :param final_temps: The final temperature distribution.
    """

    ask_save_data(raw_temp_map)
    ask_save_final_temps(final_temps)


def ask_save_data(raw_temp_map: NDArray[np.float64]):

    invalid = True
    while invalid:
        response = input("Do you want to save the simulation data to a file? (y/n): ").lower()
        if response == "y":
            filename = input("Enter the filename to save the simulation data (e.g., simulation_data.csv): ")
            save_data(raw_temp_map, filename)
            invalid = False
        elif response == "n":
            invalid = False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def ask_save_final_temps(final_temps: NDArray[np.float64]):

    invalid = True
    while invalid:
        response = input("Do you want to save the final temperature distribution to a file? (y/n): ").lower()
        if response == 'y':
            filename = input("Enter the filename to save the temperatures (e.g., final_temps.csv): ")
            save_data(final_temps, filename)
            invalid = False
        elif response == 'n':
            invalid = False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def save_data(data: NDArray[np.float64], filename: str):

    """
    Save the temperature data to a CSV file.

    :param NDArray[float64] data: The temperature data to save.
    :param str filename: The name of the file to save the data to.
    """

    path = Path(f"Data/{filename}")
    if not data.any():
        raise ValueError("No data to save.")
    print(f"Saving data to {path}...")
    np.savetxt(path, data, delimiter=',', fmt='%.6g', comments='')
    print("Data saved successfully.")


def load_init_temps(filename: Path) -> NDArray[np.float64]:

    """
    Load an initial temperature distribution from a .csv file.

    :param Path filename: The path to the file to load the initial temperatures from.
    :return: The initial temperature distribution.
    """

    path = Path(f"Data/{filename}")
    print(f"Loading initial temperatures from {path}...")
    init_temps = np.loadtxt(path, delimiter=',')
    print("Initial temperatures loaded successfully.")
    return init_temps


def load_points(filename: Path, points: list[int]) -> NDArray[np.float64]:

    """
    Loads the temperature over time data from specific positions from a .csv file.

    :param Path filename: The path to the file to load the temperature data from.
    :param list[int] points: The positions to load the temperature data from.
    :return: The temperature data from the specified positions.
    """

    path = Path(f"Data/{filename}")
    print(f"Loading temperature data from {path} at positions {points}...")
    data: NDArray[np.float64] = np.loadtxt(path, delimiter=',')
    columns = [0] + points
    temperature_data = data[:, columns]
    print("Temperature data loaded successfully.")
    return temperature_data


def load_temp_data(csv_path: Path):

    path = Path(f"Data/{csv_path}")
    print(f"Loading temperature data from {path}...")
    data = np.loadtxt(path, delimiter=',')
    times = data[:, 0]
    temps = data[:, 1:]
    print(f"Successfully loaded data.")
    return times, temps


def save_minimums(temperature_data: NDArray[np.float64], length: float, filename: str):

    temps = temperature_data[:, 1:]
    min_indices = np.argmin(temps, axis=0)
    positions = np.linspace(0.0, length, temps.shape[1])
    min_times = temperature_data[min_indices, 0]
    offset = np.min(min_times)
    offset_times = min_times - offset
    min_temps = temps[min_indices, np.arange(temps.shape[1])]
    minimums = np.column_stack((positions, min_times, offset_times, min_temps))
    save_data(minimums, filename)

# test_data = load_points(Path("copper_chop.csv"), [1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
# print(find_minimums(test_data))
save_minimums(np.loadtxt("data/copper_chop.csv", delimiter=','), 0.0035, "copper_minimums.csv")