from pathlib import Path

import matplotlib.pyplot as plt
from DataHandling import load_points, load_temp_data
from matplotlib.widgets import Slider
import numpy as np


COLORS = ["#500000", "#F15946", "#F9C22E", "#53B3CB", "#E4E50", "#500000", "#F15946", "#F9C22E", "#53B3CB", "#3E4E50"]


def time_evolution_plot(csv_path: Path):

    times, temps = load_temp_data(csv_path)
    x_positions = np.linspace(0.0, 1.0, temps.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    [profile_line] = ax.plot(x_positions, temps[0])
    ax.set_xlabel('Position')
    ax.set_ylabel('Temperature')
    ax.set_ylim(np.min(temps)/1.1, np.max(temps)*1.1)
    ax.set_title(f'Temperature Profile at t = {times[0]:.6g}')
    ax.grid(True)

    slider_ax = fig.add_axes((0.2, 0.02, 0.6, 0.05))
    time_slider = Slider(
        ax = slider_ax,
        label = "Time Index",
        valmin = 0,
        valmax = len(times) - 1,
        valinit = 0,
        valstep = 1
    )

    def update(val: float):
        
        index = int(time_slider.val)
        profile_line.set_ydata(temps[index])
        ax.set_title(f"Temperature Profile at t = {times[index]:.6g}")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def positions_over_time_plot(csv_path: Path):

    data = load_points(csv_path, [1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
    times = data[:, 0]
    temps = data[:, 1:]
    fig, ax = plt.subplots(figsize=(10, 6))
    min_times = []
    min_temps = []
    ax.set_title("Temperature Evolution at Various Positions")
    for i in range(temps.shape[1]):
        curve = temps[:, i]
        min_indx = np.argmin(curve)
        min_times.append(times[min_indx])
        min_temps.append(curve[min_indx])
        ax.plot(times, curve, label=f"Position {10*i + 1}", color=f"{COLORS[i]}")
    ax.scatter(min_times, min_temps, color="#CE1483", s=20, zorder=5, marker="x", label='Minimum Temperatures')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    temp_diff = np.max(temps) - np.min(temps)
    ax.set_xlim(0.0, 25.0)
    ax.set_ylim(np.min(temps) - temp_diff * 0.1, np.max(temps) + temp_diff * 0.1)
    ax.grid(True)

    plt.show()


def minimum_curve_plot(csv_path: Path):

    positions, min_data = load_temp_data(csv_path)
    min_times = min_data[:, 1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Lag Time [s]")
    ax.set_title("Lag Time of Temperature Drop vs Position")
    ax.grid(True)
    ax.plot(positions, min_times, c="#500000")
    plt.show()



# positions_over_time_plot(Path("copper_chop.csv"))
# time_evolution_plot(Path("copper_chop.csv"))
# minimum_curve_plot(Path("copper_minimums.csv"))