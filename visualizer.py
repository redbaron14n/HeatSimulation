from matplotlib.widgets import Slider
from pathlib import Path
from services.data_handling import DataHandler
from typing import Any, cast
import matplotlib.pyplot as _plt
import numpy as np

plt = cast(Any, _plt)


def time_evolution_plot(file_path: Path):

    handler = DataHandler(str(file_path), load=True)
    times = handler.times
    temps = handler.temps
    length = handler.length

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