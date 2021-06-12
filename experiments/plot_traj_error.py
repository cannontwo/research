import sys
import os
import glob
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(no_learning_folder, learning_folder):
    no_learning_runs = []
    for d in sorted(os.listdir(no_learning_folder)):
        run_dict = {}
        run_dict["planned"] = []
        run_dict["executed"] = []

        for p in sorted(glob.glob(no_learning_folder + "/" + d + "/planned_traj_*.csv")):
            run_dict["planned"].append(pd.read_csv(p))

        for e in sorted(glob.glob(no_learning_folder + "/" + d + "/executed_traj_*.csv")):
            run_dict["executed"].append(pd.read_csv(e))

        run_dict["distances"] = pd.read_csv(no_learning_folder + "/" + d + "/distances_0.csv")

        no_learning_runs.append(run_dict)

    learning_runs = []
    for d in sorted(os.listdir(learning_folder)):
        run_dict = {}
        run_dict["planned"] = []
        run_dict["executed"] = []

        for p in sorted(glob.glob(learning_folder + "/" + d + "/planned_traj_*.csv")):
            run_dict["planned"].append(pd.read_csv(p))

        for e in sorted(glob.glob(learning_folder + "/" + d + "/executed_traj_*.csv")):
            run_dict["executed"].append(pd.read_csv(e))

        run_dict["distances"] = pd.read_csv(learning_folder + "/" + d + "/distances_0.csv")

        learning_runs.append(run_dict)

    return (no_learning_runs, learning_runs)

def plot_distances(all_learning_df, all_no_learning_df, block=True):
    plt.figure()
    sns.lineplot(x='timestep', y='distance',
            data=all_no_learning_df.reset_index(), label="Just Replanning")
    sns.lineplot(x='timestep', y='distance',
            data=all_learning_df.reset_index(), label="PARL+Planning")

    plt.title("Distance to Goal Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Distance to Goal")
    plt.legend();
    plt.show(block=block)

def compute_norm(x):
    return math.sqrt(math.pow(x[0], 2) + math.pow(x[1], 2) + math.pow(x[2], 2))

def make_traj_error_norm_df(run_dict):
    planned_df = pd.concat(run_dict["planned"])
    exec_df = pd.concat(run_dict["executed"])

    diff_df = planned_df[["statex", "statey", "stateth", "statev", "statedth"]] - exec_df[["statex", "statey", "stateth", "statev", "statedth"]]
    norm_df = diff_df.apply(compute_norm, axis=1)

    return norm_df

def plot_cumulative_error(no_learning_runs, learning_runs, block=True):
    plt.figure()
    all_no_learning_error_norms = pd.concat([make_traj_error_norm_df(r).cumsum() for r in no_learning_runs if len(r["planned"]) > 0])
    all_learning_error_norms = pd.concat([make_traj_error_norm_df(r).cumsum() for r in learning_runs if len(r["planned"]) > 0])
    sns.lineplot(data=all_no_learning_error_norms, label="Just Replanning")
    sns.lineplot(data=all_learning_error_norms, label="PARL+Planning")

    plt.title("Cumulative Trajectory Tracking Error")
    plt.ylabel("Cumulative Error")
    plt.xlabel("Timestep")
    plt.legend()

    plt.show(block=block)

def run():
    if len(sys.argv) != 3:
        print("Incorrect number of arguments entered. Pass <no_learning_folder> <learning_folder>")
        return

    no_learning_runs, learning_runs = read_data(sys.argv[1], sys.argv[2])

    # TODO Combine all plots
    all_learning_dist_df = pd.concat([r["distances"] for r in learning_runs])
    all_no_learning_dist_df = pd.concat([r["distances"] for r in no_learning_runs])

    plot_distances(all_learning_dist_df, all_no_learning_dist_df, block=False)
    plot_cumulative_error(no_learning_runs, learning_runs)

if __name__ == "__main__":
    run()
