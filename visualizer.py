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

    times = data.times
    temps = data.temps[:, np.array([0, -1])]

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Temperature Evolution at Front and Rear of Sample")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")

    front = temps[:, 0]
    rear = temps[:, 1]

    front_min_mask = (front[1:-1] < front[:-2]) & (front[1:-1] < front[2:])
    front_min_indices = np.where(front_min_mask)[0]+1
    front_min_temps = front[front_min_indices]
    front_min_times = times[front_min_indices]
    ax.scatter(front_min_times, front_min_temps, color="Blue", s=20, zorder=5, marker="x")

    rear_min_mask = (rear[1:-1] < rear[:-2]) & (rear[1:-1] < rear[2:])
    rear_min_indices = np.where(rear_min_mask)[0]+1
    rear_min_temps = rear[rear_min_indices]
    rear_min_times = times[rear_min_indices]
    ax.scatter(rear_min_times, rear_min_temps, color="Red", s=20, zorder=5, marker="x")

    ax.plot(times, front, label="Front")
    ax.plot(times, rear, label="Rear")

    min_temp = np.min(temps)
    max_temp = np.max(temps)
    temp_diff = max_temp - min_temp
    ax.set_ylim(min_temp - temp_diff/10, max_temp + temp_diff/10)
    ax.grid(True)
    plt.show()