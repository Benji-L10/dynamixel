# Miguel Floran, Ben Ly, Electrical Engineering Department, Cal Poly
# Lab Assignment 1: Joint Space Control Interface, Data Logging, and Visualization

"""
1. Use the provided Robot API to command coordinated joint-space motions with the built-in
time profile.
2. Acquire joint positions continuously with precise timestamps and compute sampling-interval
statistics.
3. Generate clear, unit-labeled time-series plots for all four joints and a histogram of ∆t.
4. Persist run data to pickle files and reproduce the same plots by loading from pickle.
5. Compare results across two trajectory times (10 s and 2 s) and briefly interpret differences.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from classes.Robot import Robot
from utils import print_readings, save_to_pickle

# Get name of current file
filename = os.path.basename(__file__)


np.set_printoptions(precision=3, suppress=True)


def main():
    robot = Robot()

    # Enable torque and set time-based profile
    robot.write_motor_state(True)

    # Zero the robot
    traj_time_h = 3.0  # sec
    joints_h = [0, 0, 0, 0]  # deg
    print(f"Homing to {joints_h} deg ...")
    robot.write_time(traj_time_h)
    robot.write_joints(joints_h)
    time.sleep(traj_time_h)

    # Commanded motion of all joints
    traj_time_s = 10.0
    robot.write_time(traj_time_s)
    poll_dt = 0.5  # sec
    target_deg = [45, -30, 30, 75]  # deg

    t_list, q_list = [], []

    # Sweep joints
    print(f"\nMoving all joints to {target_deg} deg over {traj_time_s:.1f}s ...")
    t0 = time.perf_counter()
    robot.write_joints(target_deg)
    while True:
        t_now = time.perf_counter() - t0
        readings = robot.get_joints_readings()
        t_list.append(t_now)
        q_list.append(readings)
        print_readings(readings)
        if t_now >= traj_time_s:
            break
        time.sleep(poll_dt)

    # Convert to NumPy arrays: times -> (N,), joints -> (N, 4)
    timestamps_s = np.array(t_list, dtype=float)
    joint_deg = np.array(q_list, dtype=float)

    # Timing histogram of sampling intervals
    if timestamps_s.size >= 2:
        dt = np.diff(timestamps_s)  # (N-1,)
        dt_mean = float(np.mean(dt))
        dt_median = float(np.median(dt))
        dt_min = float(np.min(dt))
        dt_max = float(np.max(dt))
        dt_std = float(np.std(dt))

        print("\nSampling interval statistics (dt in seconds):")
        print(f"  mean   : {dt_mean:.6f} s")
        print(f"  median : {dt_median:.6f} s")
        print(f"  min    : {dt_min:.6f} s")
        print(f"  max    : {dt_max:.6f} s")
        print(f"  std    : {dt_std:.6f} s")

        plt.figure(figsize=(6, 4))
        plt.hist(dt, bins="auto", edgecolor="black")
        plt.xlabel("Sampling interval dt (s)")
        plt.ylabel("Count")
        plt.title("Histogram of Sampling Intervals")
        plt.tight_layout()
        plt.savefig("histogram_sampling_intervals.png", dpi=160)

    # Joint angles plot block
    angles_deg = joint_deg[:, 0, :]  # (N, 4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    titles = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, ax in enumerate(axes.ravel()):
        ax.plot(timestamps_s, angles_deg[:, idx], color=colors[idx])
        ax.set_title(titles[idx])
        ax.set_ylabel("Angle (deg)")
        ax.grid(True)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig("joint_angles_over_time.png", dpi=150)  # save explicitly
    plt.show()
    plt.close(fig)

    # Save data to a pickle file
    save_to_pickle(
        {
            "timestamps_s": timestamps_s,
            "joint_deg": joint_deg,
            "target_deg": target_deg,
            "traj_time_s": traj_time_s,
        },
        f"{filename}_robot_data.pkl",
    )

    # Shutdown
    robot.close()


if __name__ == "__main__":
    main()
