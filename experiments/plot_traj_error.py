import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

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

def run():
    if len(sys.argv) != 3:
        print("Incorrect number of arguments passed. Pass <no_learning_folder> <learning_folder>")
        return

    no_learning_runs, learning_runs = read_data(sys.argv[1], sys.argv[2])

    print(no_learning_runs)
    print(learning_runs)

    # TODO Combine all plots

    learning_runs[0]["distances"].plot(x="timestep", y="distance")
    plt.title("PARL+Planning Goal Distance Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Distance to Goal")
    plt.ylim(bottom=0.0)
    plt.show(block=False)

    no_learning_runs[0]["distances"].plot(x="timestep", y="distance")
    plt.title("Just Replanning Goal Distance Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Distance to Goal")
    plt.ylim(bottom=0.0)
    plt.show()


if __name__ == "__main__":
    run()
