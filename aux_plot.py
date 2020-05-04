import os

import altair as alt
import numpy as onp
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d import Axes3D


# Aux functions
def disable_axes(ax):
    ax.axis("off")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_scatter_and_quiver(ax, step, paths, grid, drifts, diffusions, color=None, labels=None):
    if color is None:
        color = 'b'
    ax.quiver(grid[step, :, 0], grid[step, :, 1], drifts[step, :, 0], drifts[step, :, 1], diffusions[step], alpha=0.2)
    if labels is None:
        ax.scatter(paths[:, step, 0], paths[:, step, 1], c=color, marker="1", s=30, alpha=0.5)
    else:
        ax.scatter(paths[:, step, 0], paths[:, step, 1], c=labels, marker="1", s=30, alpha=0.5)


# Actual functions
def plot_ard_weights(config, weights):
    image_name = os.path.join(config["results_dir"], f"kernel_weights")

    data = {"x": list(range(int(onp.prod(config["state_size"])))),
            "y": weights,
            "color": [0.5] * onp.prod(config["state_size"])}

    sns_bar = sns.barplot(x="x", y="y", hue="color", data=data)
    sns_bar.figure.savefig(f"{image_name}.png")
    sns_bar.figure.clear()


def plot_1d_timeseries_data(config, data, true_data, data_dims, time_steps, time_steps_test):
    image_name = os.path.join(config["results_dir"], f"time_series_data")
    f, axarr = plt.subplots(data_dims)

    t_true = onp.arange(0, (time_steps + 1) * true_data.shape[1])
    t_data = onp.arange(0, (time_steps_test + 1) * true_data.shape[1])
    true_data = true_data.reshape((-1, 8))
    data_interpolation = data[:time_steps + 1].reshape((-1, 8))
    data = data.reshape((-1, 8))
    for i in range(data_dims):
        axarr[i].plot(t_true, data_interpolation[:, i], color="orange")
        axarr[i].plot(t_true, true_data[:, i])

    f.savefig(f"{image_name}.png", bbox_inches="tight", pad_inches=0, dpi=1000)
    plt.close(f)


def plot_1d_paths(config, paths_mean, grid, drifts, diffusions, labels=None):
    image_name = os.path.join(config["results_dir"], f"paths1d")
    num_steps = config["num_steps_test"] + 1

    color_dict = {0: 'b', 1: 'r', 2: 'g', 3: 'c', 4: 'y', 5: 'k', 6: 'm', 7: 'tab:orange'}

    f, axarr = plt.subplots(1)
    t_array = onp.linspace(0, 1, num_steps)
    axarr.quiver(grid[:, 0], grid[:, 1], 0, drifts, diffusions, alpha=0.2)

    for i in range(paths_mean.shape[0]):
        if labels is None:
            axarr.plot(t_array, paths_mean[i], '--')
        else:
            axarr.plot(t_array, paths_mean[i], '--', color=color_dict[labels[i]])

    f.savefig(f"{image_name}.png", bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close(f)


def plot_2d_paths(config, path_true, paths_mean, labels=None):
    image_name = os.path.join(config["results_dir"], f"paths2d")

    color_dict = {0: 'b', 1: 'r', 2: 'g', 3: 'c', 4: 'y', 5: 'k', 6: 'm', 7: 'tab:orange'}

    f, axarr = plt.subplots(1)

    for i in range(0, paths_mean.shape[0]):
        if labels is None:
            axarr.plot(paths_mean[i, :, 0], paths_mean[i, :, 1], linewidth=1, alpha=0.6)
        else:
            axarr.plot(paths_mean[i, :, 0], paths_mean[i, :, 1], linewidth=1, alpha=0.6,
                       color=color_dict[labels[i]])

    num_classes = len(onp.unique(labels))
    for i in range(num_classes):
        if labels is None:
            axarr.plot(path_true[:, i * (config["num_data_points_test"] // num_classes), 0],
                       path_true[:, i * (config["num_data_points_test"] // num_classes), 1], '--', linewidth=1.5)
        else:
            axarr.plot(path_true[:, i * (config["num_data_points_test"] // num_classes), 0],
                       path_true[:, i * (config["num_data_points_test"] // num_classes), 1], '--', linewidth=1.5,
                       color=color_dict[labels[i * (config["num_data_points_test"] // num_classes)]])

    f.savefig(f"{image_name}.png", bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close(f)

def plot_paths(config, paths, grid, drifts, diffusions, labels=None, paths_original=None, step_size=10):
    image_name = os.path.join(config["results_dir"], f"paths")
    num_plots = (config["num_steps_test"] + 1) // step_size

    if paths_original is None:
        f, axarr = plt.subplots(1)
    else:
        f, axarr = plt.subplots(2)

    plt.subplots_adjust(wspace=0, hspace=0)

    mean_trajectory_1 = None
    mean_trajectory_2 = None
    mean_trajectory_3 = None
    if config["dataset"] == "missile_to_air":
        if config["num_trajectories"] >= 1:
            mean_trajectory_1 = onp.mean(paths_original[labels == 0], axis=0)
        if config["num_trajectories"] >= 2:
            mean_trajectory_2 = onp.mean(paths_original[labels == 1], axis=0)
        if config["num_trajectories"] == 3:
            mean_trajectory_3 = onp.mean(paths_original[labels == 2], axis=0)

    for i, step in enumerate(range(0, config["num_steps_test"] + 1, step_size)):

        if paths_original is None:
            axarr.clear()
            plot_scatter_and_quiver(axarr, step, paths, grid, drifts, diffusions, 'r', labels)
            disable_axes(axarr)
        else:
            axarr[0].clear()
            axarr[1].clear()
            if config["dataset"] == "missile_to_air":
                if config["num_trajectories"] >= 1:
                    axarr[0].plot(mean_trajectory_1[:, 0], mean_trajectory_1[:, 1], linewidth=1.5, color='purple')
                    axarr[1].plot(mean_trajectory_1[:, 0], mean_trajectory_1[:, 1], linewidth=1.5, color='purple')
                if config["num_trajectories"] >= 2:
                    axarr[0].plot(mean_trajectory_2[:, 0], mean_trajectory_2[:, 1], linewidth=1.5, color='g')
                    axarr[1].plot(mean_trajectory_2[:, 0], mean_trajectory_2[:, 1], linewidth=1.5, color='g')
                if config["num_trajectories"] == 3:
                    axarr[0].plot(mean_trajectory_3[:, 0], mean_trajectory_3[:, 1], linewidth=1.5, color='yellow')
                    axarr[1].plot(mean_trajectory_3[:, 0], mean_trajectory_3[:, 1], linewidth=1.5, color='yellow')

            plot_scatter_and_quiver(axarr[1], step, paths, grid, drifts, diffusions, 'r', labels)
            disable_axes(axarr[1])
            if step < config["num_steps"]:
                axarr[0].scatter(paths_original[:, step, 0], paths_original[:, step, 1],
                                 c=labels, marker="1", s=30, alpha=0.5)
            else:
                axarr[0].scatter(paths_original[:, config["num_steps"], 0],
                                 paths_original[:, config["num_steps"], 1],
                                 c=labels, marker="1", s=30, alpha=0.5)
            if "y_max" in config:
                axarr[0].set_xlim([config["x_min"], config["x_max"]])
                axarr[0].set_ylim([config["y_min"], config["y_max"]])
            disable_axes(axarr[0])

        f.savefig(f"{image_name}_{i}.png", bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(f)


def make_paths_gif(config, paths, grid, drifts, diffusions, labels=None, paths_original=None):
    gif_name = os.path.join(config["results_dir"], f"paths_gif")

    if paths_original is None:
        f, axarr = plt.subplots(1)
    else:
        f, axarr = plt.subplots(2)
    plt.subplots_adjust(wspace=0, hspace=0)

    mean_trajectory_1 = None
    mean_trajectory_2 = None
    mean_trajectory_3 = None
    if config["dataset"] == "missile_to_air":
        if config["num_trajectories"] >= 1:
            mean_trajectory_1 = onp.mean(paths_original[labels == 0], axis=0)
        if config["num_trajectories"] >= 2:
            mean_trajectory_2 = onp.mean(paths_original[labels == 1], axis=0)
        if config["num_trajectories"] == 3:
            mean_trajectory_3 = onp.mean(paths_original[labels == 2], axis=0)

    def update(i):
        if paths_original is None:
            axarr.clear()
            plot_scatter_and_quiver(axarr, i, paths, grid, drifts, diffusions, 'r', labels)
            disable_axes(axarr)
        else:
            axarr[0].clear()
            axarr[1].clear()
            if config["dataset"] == "missile_to_air":
                if config["num_trajectories"] >= 1:
                    axarr[0].plot(mean_trajectory_1[:, 0], mean_trajectory_1[:, 1], linewidth=1.5, color='purple')
                    axarr[1].plot(mean_trajectory_1[:, 0], mean_trajectory_1[:, 1], linewidth=1.5, color='purple')
                if config["num_trajectories"] >= 2:
                    axarr[0].plot(mean_trajectory_2[:, 0], mean_trajectory_2[:, 1], linewidth=1.5, color='g')
                    axarr[1].plot(mean_trajectory_2[:, 0], mean_trajectory_2[:, 1], linewidth=1.5, color='g')
                if config["num_trajectories"] == 3:
                    axarr[0].plot(mean_trajectory_3[:, 0], mean_trajectory_3[:, 1], linewidth=1.5, color='yellow')
                    axarr[1].plot(mean_trajectory_3[:, 0], mean_trajectory_3[:, 1], linewidth=1.5, color='yellow')
            plot_scatter_and_quiver(axarr[1], i, paths, grid, drifts, diffusions, 'r', labels)
            if labels is None:
                if i < config["num_steps"]:
                    axarr[0].scatter(paths_original[:, i, 0], paths_original[:, i, 1], marker="1", s=30, alpha=0.5)
                else:
                    axarr[0].scatter(paths_original[:, config["num_steps"], 0],
                                     paths_original[:, config["num_steps"], 1], marker="1", s=30, alpha=0.5)
            else:
                if i < config["num_steps"]:
                    axarr[0].scatter(paths_original[:, i, 0], paths_original[:, i, 1],
                                     c=labels, marker="1", s=30, alpha=0.5)
                else:
                    axarr[0].scatter(paths_original[:, config["num_steps"], 0],
                                     paths_original[:, config["num_steps"], 1],
                                     c=labels, marker="1", s=30, alpha=0.5)
            if "y_max" in config:
                axarr[0].set_xlim([config["x_min"], config["x_max"]])
                axarr[0].set_ylim([config["y_min"], config["y_max"]])
            disable_axes(axarr[0])
            disable_axes(axarr[1])
        return axarr

    anim = FuncAnimation(f, update, frames=config["num_steps_test"] + 1, interval=500)
    anim.save(f"{gif_name}.gif", dpi=160, writer="imagemagick")
