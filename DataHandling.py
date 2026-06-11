# ###################
# Ian Janes
# Prof. Don Lipkin
# Data Handling service file
# ###################

from numpy.typing import NDArray
import numpy as np


def save_procedure(raw_temp_map, final_temps):

    """
    Perform the save procedure for the simulation data and final temperatures.

    :param NDArray[np.float64] raw_temp_map: The raw temperature data from the simulation.
    :param NDArray[np.float64] final_temps: The final temperature distribution.
    """

    response = input("Do you want to save the simulation data to a file? (y/n): ").lower()
    if response == "y":
        filename = input("Enter the filename to save the simulation data (e.g., simulation_data.csv): ")
        save_data(raw_temp_map, filename)
    response = input("Do you want to save the final temperature distribution to a file? (y/n): ").lower()
    if response == 'y':
        filename = input("Enter the filename to save the temperatures (e.g., final_temps.csv): ")
        save_data(final_temps, filename)


def save_data(data: NDArray[np.float64], filename: str):

    """
    Save the temperature data to a CSV file.

    :param NDArray[float64] data: The temperature data to save.
    :param str filename: The name of the file to save the data to.
    """

    if not data.any():
        raise ValueError("No data to save.")
    print(f"Saving data to {filename}...")
    np.savetxt(filename, data, delimiter=',', fmt='%.6g', comments='')
    print("Data saved successfully.")


def load_init_temps(filename: str) -> NDArray[np.float64]:

    """
    Load an initial temperature disitribution from a .csv file.

    :param str filename: The name of the file to load the initial temperatures from.
    :return: The initial temperature distribution.
    """

    print(f"Loading initial temperatures from {filename}...")
    init_temps = np.loadtxt(filename, delimiter=',')
    print("Initial temperatures loaded successfully.")
    return init_temps