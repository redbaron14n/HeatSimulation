from pathlib import Path

import matplotlib.pyplot as plt
from DataHandling import load_init_temps, load_points, load_temp_data
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.widgets import Slider
import numpy as np


COLORS = ["#500000", "#F15946", "#F9C22E", "#53B3CB", "#E4E50", "#500000", "#F15946", "#F9C22E", "#53B3CB", "#3E4E50"]


def time_evolution_plot(csv_path: Path, subject_length: float = 1.0):

    times, temps = load_temp_data(csv_path)
    x_positions = np.linspace(0.0, subject_length, temps.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    [profile_line] = ax.plot(x_positions, temps[0])
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Temperature [K]')
    min_temp, max_temp = np.min(temps), np.max(temps)
    temp_diff = max_temp - min_temp
    ax.set_ylim(min_temp - temp_diff * 0.1, max_temp + temp_diff * 0.1)
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


def save_evo_video(csv_path: Path, output_path: Path, subject_length: float = 1.0, fps: int = 30, vid_length: float = 10.0, max_sim_time: float = 100.0):

    output_path = Path(f"Data/{output_path}")
    times, temps = load_temp_data(csv_path)
    x_positions = np.linspace(0.0, subject_length, temps.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    [profile_line] = ax.plot(x_positions, temps[0])
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Temperature [K]')
    min_temp, max_temp = np.min(temps), np.max(temps)
    temp_diff = max_temp - min_temp
    ax.set_ylim(min_temp - temp_diff * 0.1, max_temp + temp_diff * 0.1)
    ax.set_title(f'Temperature Profile at t = {times[0]:.6g}')
    ax.grid(True)

    def set_frame(index: int):

        profile_line.set_ydata(temps[index])
        ax.set_title(f"Temperature Profile at t = {times[index]:.6g}")
        return profile_line,

    end_index = np.searchsorted(times, max_sim_time, side="right")
    target_frames = int(fps * vid_length)
    step = max(1, end_index // target_frames)
    frame_indices = range(0, end_index, step)
    anim = FuncAnimation(fig, set_frame, frames=frame_indices, interval=100, blit=False)
    print(f"Saving video to {output_path}...")
    anim.save(output_path, writer=FFMpegWriter(fps=fps))
    print("Video saved successfully.")
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


def linear_regression_plot(csv_path: Path, subject_length: float = 0.100):

    temps = np.asarray(load_init_temps(csv_path)).ravel()
    x_positions = np.linspace(0.0, subject_length, temps.shape[0])
    slope, intercept = np.polyfit(x_positions, temps, 1)
    regression_line = slope * x_positions + intercept
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title("Linear Regression of Steady-state Temperature Distribution")
    ax.grid(True)
    ax.plot(x_positions, temps, label="Data")
    ax.plot(x_positions, regression_line, linestyle="--", color="#CE1483", label=f"Fit: y = {slope:.4g}x + {intercept:.4g}")
    ax.legend()
    plt.show()


# positions_over_time_plot(Path("copper_chop.csv"))
# time_evolution_plot(Path("copper_heating.csv"))
# minimum_curve_plot(Path("copper_minimums.csv"))
# save_evo_video(Path("copper_heating.csv"), Path("copper_heating.mp4"), subject_length=0.100, fps=30, vid_length=10.0, max_sim_time=2000.0)
linear_regression_plot(Path("copper_steadystate.csv"), subject_length=0.100)