from matplotlib.widgets import Slider
from services.data_handling import DataHandler
from typing import Any, cast
import matplotlib.pyplot as _plt
import numpy as np

plt = cast(Any, _plt)

DIRECTORY = "Data/"


def time_evolution_plot(data: DataHandler):

    times = data.times
    temps = data.temps
    length = data.length

    x_positions = np.linspace(0., length, temps.shape[1])
    fig, ax = plt.subplots(figsize = (10, 6))
    [profile_line] = ax.plot(x_positions, temps[0])
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Temperature [K]")
    min_temp, max_temp = np.min(temps), np.max(temps)
    temp_diff = max_temp - min_temp
    ax.set_ylim(min_temp - temp_diff/10, max_temp+temp_diff/10)
    ax.set_title(f"Temperature Profile at t = {times[0]:.6g}s")
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
        ax.set_title(f"Temperature Profile at t = {times[index]:.6g}s")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def front_and_back_temp_plot(data: DataHandler):

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Temperature Evolution at Front and Rear of Sample")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")

    ax.plot(data.times, data.temps[:, 0], label="Front")
    ax.scatter(data.extrema[:, 0, 0], data.extrema[:, 0, 1], color="Blue", s=20, zorder=5, marker="x")
    ax.scatter(data.extrema[:, 0, 4], data.extrema[:, 0, 5], color="Purple", s=20, zorder=5, marker="x")

    ax.plot(data.times, data.temps[:, -1], label="Rear")
    ax.scatter(data.extrema[:, -1, 0], data.extrema[:, -1, 1], color="Red", s=20, zorder=5, marker="x")
    ax.scatter(data.extrema[:, -1, 2], data.extrema[:, -1, 3], color="Pink", s=20, zorder=5, marker="x")
    ax.scatter(data.extrema[:, -1, 4], data.extrema[:, -1, 5], color="Yellow", s=20, zorder=5, marker="x")

    temps = data.temps[:, np.array([0, -1])]
    min_temp = np.min(temps)
    max_temp = np.max(temps)
    temp_diff = max_temp - min_temp
    ax.set_ylim(min_temp - temp_diff/10, max_temp + temp_diff/10)
    ax.grid(True)
    plt.show()


def position_plot(data: DataHandler):

    times = data.times
    temps = data.temps
    dx = data.length / (temps.shape[1] - 1)

    
    t_points = np.linspace(0., np.max(times), len(times))
    fig, ax = plt.subplots(figsize = (10, 6))
    [profile_line] = ax.plot(t_points, temps[:, 0])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")

    min_temp, max_temp = np.min(temps), np.max(temps)
    temp_diff = max_temp - min_temp
    ax.set_ylim(min_temp - temp_diff/10, max_temp + temp_diff/10)
    ax.set_title(f"Temperature evolution at x = 0mm")

    slider_ax = fig.add_axes((0.2, 0.02, 0.6, 0.05))
    pos_slider = Slider(
        ax = slider_ax,
        label = "Position Index",
        valmin = 0,
        valmax = temps.shape[1] - 1,
        valinit = 0,
        valstep = 1
    )

    def update(val: float):

        index = int(pos_slider.val)
        profile_line.set_ydata(temps[:, index])
        ax.set_title(f"Temperature Profile at x = {(1000*dx*index):.2f}mm")
        fig.canvas.draw_idle()

    pos_slider.on_changed(update)
    plt.show()