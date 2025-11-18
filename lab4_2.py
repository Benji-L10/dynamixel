# Ben Ly, Electrical Engineering Department, Cal Poly
#Lab Assignment 4: Trajectory Generation in Joint Space for Robot Manipulators
# 2. Trajectory planning in joint space using cubic polynomials

"""
Write a Python script named lab4_2.py to execute joint-space cubic trajectory planning on the
physical robot. You will convert the task-space waypoints from Lab 3 into joint angles using IK,
generate smooth cubic trajectories in joint space, and command the robot to follow the planned
path.
For this part, use the same three task-space waypoints from Lab 3 shown in Table 1 below (repro-
duced for convenience). Create a complete triangular motion sequence: Waypoint 1 →2 →3 →
4(=1).
"""
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from classes.Robot import Robot
from classes.TrajPlanner import TrajPlanner

def collect_data():
    """
    Collects data for the robot's movement and saves it to a pickle file.
    (Joint-space cubic trajectory; IK only at the task-space waypoints)
    """
    traj_time = 5          # Trajectory time per segment (seconds)
    points_num = 998       # Number of INTERIOR waypoints per segment (~5 ms)
    robot = Robot()

    # Define task-space setpoints (triangle + return)
    ee_poses = np.array([
        [25,  -100, 150, -60],
        [150,   80, 300,   0],
        [250, -115,  75, -45],
        [25,  -100, 150, -60],  # Return to start
    ], dtype=float)

    print("Computing IK for waypoints.")
    try:
        joint_angles = np.array([
            robot.get_ik(ee_poses[0, :]),
            robot.get_ik(ee_poses[1, :]),
            robot.get_ik(ee_poses[2, :]),
            robot.get_ik(ee_poses[3, :]),
        ], dtype=float)
        print("IK solutions (deg):")
        for i, angles in enumerate(joint_angles):
            print(f"  Waypoint {i+1}: {np.array2string(angles, precision=2, floatmode='fixed')}")
    except ValueError as e:
        raise ValueError(f"End-Effector Pose Unreachable: {e}")

    # Create trajectory in joint space (cubic, v=0 at boundaries)
    print("\nGenerating cubic trajectory (joint space).")
    tj = TrajPlanner(joint_angles)
    trajectories = tj.get_cubic_traj(traj_time, points_num)   # (N,5) = [t, q1..q4]

    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Total trajectory time: {trajectories[-1, 0]:.2f} seconds")

    # Time step and derived command rate
    time_step = trajectories[1, 0] - trajectories[0, 0]
    print(f"Time step between points: {time_step*1000:.2f} ms")
    print(f"Command frequency: {1/time_step:.1f} Hz")

    # Pre-allocate data (over-allocate for safety during continuous sampling)
    total_points = len(trajectories)
    max_samples = total_points * 5  # assume we may sample 5× while waiting to send each point
    data_time = np.zeros(max_samples, dtype=float)
    data_ee_poses = np.zeros((max_samples, 4), dtype=float)
    data_q = np.zeros((max_samples, 4), dtype=float)
    count = 0

    # Initialize robot
    print("\nInitializing robot.")
    robot.write_motor_state(True)

    # Zero the robot
    traj_time_h = 3.0  # sec
    joints_h = [0, 0, 0, 0]  # deg
    print(f"Homing to {joints_h} deg ...")
    robot.write_time(traj_time_h)
    robot.write_joints(joints_h)
    time.sleep(traj_time_h)

    # Move to starting position
    print("Moving to start position.")
    robot.write_time(traj_time)               # coarse profile to move to first point
    robot.write_joints(trajectories[0, 1:])
    time.sleep(traj_time)                     # wait for completion

    # Execute trajectory by streaming commands at planned step size
    print("\nExecuting trajectory.")
    robot.write_time(time_step)               # set built-in profile time ≈ step
    start_time = time.perf_counter()

    for i in range(1, len(trajectories)):
        # Calculate when this command should be sent
        target_time = start_time + trajectories[i, 0]

        # Wait until it's time to send this command; sample while waiting
        while time.perf_counter() < target_time:
            current_time = time.perf_counter() - start_time

            if count < max_samples:
                q_now = robot.get_joints_readings()[0, :]
                data_q[count, :] = q_now
                data_time[count] = current_time
                data_ee_poses[count, :] = robot.get_ee_pos(q_now)[0:4]  # [x,y,z,pitch]
                count += 1

            # Small sleep to prevent CPU overload
            time.sleep(0.001)  # 1 ms

        # Send the command at the scheduled time
        robot.write_joints(trajectories[i, 1:])
    
    # Shutdown
    robot.close()

    total_time = time.perf_counter() - start_time
    print(f"\nTrajectory complete!")
    print(f"Planned time: {trajectories[-1, 0]:.2f}s")
    print(f"Actual time: {total_time:.2f}s")
    print(f"Total samples collected: {count}")
    if total_time > 0:
        print(f"Average sample rate: {count/total_time:.1f} Hz")

    # Trim unused space
    data_time = data_time[:count]
    data_ee_poses = data_ee_poses[:count, :]
    data_q = data_q[:count, :]

    # SAVE DATA (filled TODO)
    out = dict(
        ee_waypoints=ee_poses,
        joint_waypoints=joint_angles,
        traj_joint=trajectories,     # planned joint trajectory (time + q1..q4)
        t=data_time,                 # measured timestamps
        q_meas=data_q,               # measured joint angles (deg)
        ee_meas=data_ee_poses,       # measured EE pose [x,y,z,pitch]
    )
    with open("lab4_part2.pkl", "wb") as f:
        pickle.dump(out, f)
    print("Saved run to lab4_part2.pkl")


def plot_data():
    """
    Loads data from a pickle file and plots it.
    Plots:
      (a) Joint angles vs time (measured + planned overlay)
      (b) 3D end-effector trajectory with waypoint markers
    """
    fname = "lab4_part2.pkl"
    with open(fname, "rb") as f:
        D = pickle.load(f)

    # Unpack
    t = np.asarray(D["t"], float)                                    # (N,)
    q_meas = np.asarray(D["q_meas"], float)                          # (N,4)
    ee_meas = np.asarray(D["ee_meas"], float)                        # (N,4) [x,y,z,pitch]
    traj = np.asarray(D["traj_joint"], float)                        # (M,5) [t,q1..q4]
    ee_wps = np.asarray(D["ee_waypoints"], float)                    # (4,4)

    # (a) Joint angles vs time
    plt.figure(figsize=(10, 6))
    labels = ["q1", "q2", "q3", "q4"]
    for j in range(4):
        plt.plot(t, q_meas[:, j], label=f"{labels[j]} (meas)")
        plt.plot(traj[:, 0], traj[:, j+1], "--", alpha=0.8, label=f"{labels[j]} (plan)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Part 2 — Joint-space cubic: measured vs planned")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

    # (b) 3D end-effector trajectory
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_meas[:, 0], ee_meas[:, 1], ee_meas[:, 2], lw=1.5, label="Measured")
    ax.scatter(ee_wps[:, 0], ee_wps[:, 1], ee_wps[:, 2], s=60, label="Waypoints")
    for i, (x, y, z) in enumerate(ee_wps[:3, :3], 1):
        ax.text(x, y, z, f" WP{i}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Part 2 — 3D End-Effector Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # (c) Task-space pose, velocity, and acceleration (three stacked subplots)
    def diff_by_time(tvec, Y):
        """Numerical time derivative using np.gradient with nonuniform t support."""
        # np.gradient handles per-sample spacing if we pass tvec
        return np.gradient(Y, tvec, axis=0)

    # Pose (x,y,z,alpha) from measured EE
    X = ee_meas[:, 0:4]              # columns: x(mm), y(mm), z(mm), alpha(deg)

    # Velocities and accelerations
    V = diff_by_time(t, X)           # (N,4)  units: [mm/s, mm/s, mm/s, deg/s]
    A = diff_by_time(t, V)           # (N,4)  units: [mm/s^2, ..., deg/s^2]

    labels = ["x (mm)", "y (mm)", "z (mm)", "alpha (deg)"]
    vlabels = ["x_dot (mm/s)", "y_dot (mm/s)", "z_dot (mm/s)", "alpha_dot (deg/s)"]
    alabels = ["x_dot_dot (mm/s^2)", "y_dot_dot (mm/s^2)", "z_dot_dot (mm/s^2)", "alpha_dot_dot (deg/s^2)"]
    linestyles = ["-", "-", "-", "-"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Top: pose
    for i in range(4):
        axs[0].plot(t, X[:, i], linestyles[i], label=labels[i])
    axs[0].set_ylabel("Pose")
    axs[0].set_title("Task-space pose, velocity, and acceleration vs time")
    axs[0].grid(True)
    axs[0].legend(ncol=2)

    # Middle: velocity
    for i in range(4):
        axs[1].plot(t, V[:, i], linestyles[i], label=vlabels[i])
    axs[1].set_ylabel("Velocity")
    axs[1].grid(True)
    axs[1].legend(ncol=2)

    # Bottom: acceleration
    for i in range(4):
        axs[2].plot(t, A[:, i], linestyles[i], label=alabels[i])
    axs[2].set_ylabel("Acceleration")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)
    axs[2].legend(ncol=2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Collect data
    collect_data()
    # Plot data
    plot_data()