# ###################
# Ian Janes
# Prof. Don Lipkin
# Data Handling service file
# ###################

from numpy.typing import NDArray
from pathlib import Path
import numpy as np


def save_procedure(raw_temp_map, final_temps):

    """
    Perform the save procedure for the simulation data and final temperatures.

    :param NDArray[np.float64] raw_temp_map: The raw temperature data from the simulation.
    :param NDArray[np.float64] final_temps: The final temperature distribution.
    """

    ask_save_data(raw_temp_map)
    ask_save_final_temps(final_temps)


def ask_save_data(raw_temp_map):

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


def ask_save_final_temps(final_temps):

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


def load_init_temps(filename: str) -> NDArray[np.float64]:

    """
    Load an initial temperature disitribution from a .csv file.

    :param str filename: The name of the file to load the initial temperatures from.
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


def find_minimums(temperature_data: NDArray[np.float64]) -> NDArray[np.float64]:

    temps = temperature_data[:, 1:]
    min_indices = np.argmin(temps, axis=0)
    min_times = temperature_data[min_indices, 0]
    min_times -= min_times[0]
    min_temps = temps[min_indices, np.arange(temps.shape[1])]
    minimums = np.vstack((min_times, min_temps))
    return minimums

# test_data = load_points(Path("copper_chop.csv"), [1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
# print(find_minimums(test_data))