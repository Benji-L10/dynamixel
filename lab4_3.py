# Ben Ly, Electrical Engineering Department, Cal Poly
#Lab Assignment 4: Trajectory Generation in Joint Space for Robot Manipulators
# 3. Trajectory planning in task space using quintic polynomials

"""
Extend the TrajPlanner class to implement quintic polynomial trajectory generation. Quintic
polynomials provide fifth-order trajectories that allow explicit control over position, velocity, and
acceleration at boundary points, to enable smoother motion profiles.
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
    (Task-space quintic trajectory; IK computed at every sample)
    """
    traj_time = 5.0        # seconds per segment
    points_num = 998       # interior samples per segment (~5 ms)
    robot = Robot()

    # Task-space triangle (+ return) [x(mm), y(mm), z(mm), alpha(deg)]
    ee_poses = np.array([
        [25,  -100, 150, -60],
        [150,   80, 300,   0],
        [250, -115,  75, -45],
        [25,  -100, 150, -60],  # back to start
    ], dtype=float)

    # Plan QUINTIC in task space (rest-to-rest: v=a=0 at boundaries)
    print("Generating task-space QUINTIC trajectory.")
    planner = TrajPlanner(ee_poses)
    traj_task = planner.get_quintic_traj(traj_time, points_num)   # (N,5) = [t, x y z alpha]

    # Convert every sample to joints via IK
    print("Running IK for each sample...")
    q_cmd = np.zeros((traj_task.shape[0], 4), dtype=float)
    for i in range(traj_task.shape[0]):
        q_cmd[i, :] = robot.get_ik(traj_task[i, 1:5])  # deg

    # Combined joint trajectory array (for plotting): [t, q1..q4]
    traj_joint = np.column_stack([traj_task[:, 0], q_cmd])

    # Derived timing
    time_step = traj_task[1, 0] - traj_task[0, 0]
    print(f"Trajectory shape: {traj_task.shape}  delta_t≈{time_step*1000:.2f} ms  (≈{1/time_step:.1f} Hz)")

    # Pre-allocate logs (over-allocate; we trim later)
    total_points = len(traj_joint)
    max_samples = total_points * 5
    data_time = np.zeros(max_samples, dtype=float)
    data_ee_poses = np.zeros((max_samples, 4), dtype=float)  # [x,y,z,pitch]
    data_q = np.zeros((max_samples, 4), dtype=float)
    count = 0

    print("\nInitializing robot.")
    robot.write_motor_state(True)

    # Zero the robot
    traj_time_h = 3.0  # sec
    joints_h = [0, 0, 0, 0]  # deg
    print(f"Homing to {joints_h} deg ...")
    robot.write_time(traj_time_h)
    robot.write_joints(joints_h)
    time.sleep(traj_time_h)
    
    # Move to starting joint pose (coarse profile)
    print("Moving to start position.")
    robot.write_time(traj_time)
    robot.write_joints(traj_joint[0, 1:])
    time.sleep(traj_time)

    # Execute trajectory by streaming commands at planned step size
    print("\nExecuting IK-driven joint commands from task-space quintic.")
    robot.write_time(time_step)   # DXL time-based profile per update
    start_time = time.perf_counter()

    for i in range(1, len(traj_joint)):
        target_time = start_time + traj_joint[i, 0]

        # sample while waiting for scheduled send time
        while time.perf_counter() < target_time:
            current_time = time.perf_counter() - start_time

            if count < max_samples:
                q_now = robot.get_joints_readings()[0, :]
                data_q[count, :] = q_now
                data_time[count] = current_time
                data_ee_poses[count, :] = robot.get_ee_pos(q_now)[0:4]   # [x,y,z,pitch]
                count += 1

            time.sleep(0.001)

        # send the joint command for this sample
        robot.write_joints(traj_joint[i, 1:])

    # Shutdown
    robot.close()

    total_time = time.perf_counter() - start_time
    print(f"\nTrajectory complete!")
    print(f"Planned time: {traj_joint[-1, 0]:.2f}s,  Actual time: {total_time:.2f}s")
    print(f"Total samples collected: {count}")
    if total_time > 0:
        print(f"Average sample rate: {count/total_time:.1f} Hz")

    # Trim logs
    data_time = data_time[:count]
    data_ee_poses = data_ee_poses[:count, :]
    data_q = data_q[:count, :]

    # Save
    out = dict(
        ee_waypoints=ee_poses,      # original task-space waypoints
        traj_task=traj_task,        # planned task-space quintic [t,x,y,z,alpha]
        traj_joint=traj_joint,      # IK-converted joint commands [t,q1..q4]
        t=data_time,                # measured timestamps
        q_meas=data_q,              # measured joint angles (deg)
        ee_meas=data_ee_poses,      # measured EE [x,y,z,pitch]
    )
    with open("lab4_part3.pkl", "wb") as f:
        pickle.dump(out, f)
    print("Saved run to lab4_part3.pkl")

def plot_data():
    """
    Loads data from 'lab4_part3.pkl' and plots:
      (d) Joint angles vs time (measured + IK-from-task overlay)
      (b) 3D end-effector trajectory with waypoint markers (measured)
      (c) Task-space pose, velocity, acceleration (from measurements)
    """
    fname = "lab4_part3.pkl"
    with open(fname, "rb") as f:
        D = pickle.load(f)

    # Unpack
    t = np.asarray(D["t"], float)                   # (N,)
    q_meas = np.asarray(D["q_meas"], float)         # (N,4)
    ee_meas = np.asarray(D["ee_meas"], float)       # (N,4) [x,y,z,pitch]
    traj_joint = np.asarray(D["traj_joint"], float) # (M,5) [t,q1..q4]
    traj_task  = np.asarray(D["traj_task"], float)  # (M,5) [t,x,y,z,alpha]
    ee_wps = np.asarray(D["ee_waypoints"], float)   # (4,4)

    # (d) Joints: measured vs planned (IK from task-space)
    plt.figure(figsize=(10, 6))
    labels = ["q1", "q2", "q3", "q4"]
    for j in range(4):
        plt.plot(t, q_meas[:, j], label=f"{labels[j]} (meas)")
        plt.plot(traj_joint[:, 0], traj_joint[:, j+1], "--", alpha=0.8, label=f"{labels[j]} (plan IK)")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Part 3 — Joint angles: measured vs IK from task-space quintic")
    plt.grid(True); plt.legend(ncol=2); plt.tight_layout(); plt.show()

    # (b) 3D end-effector trajectory (measured) with waypoint markers
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_meas[:, 0], ee_meas[:, 1], ee_meas[:, 2], lw=1.5, label="Measured")
    ax.scatter(ee_wps[:, 0], ee_wps[:, 1], ee_wps[:, 2], s=60, label="Waypoints")
    for i, (x, y, z) in enumerate(ee_wps[:3, :3], 1):
        ax.text(x, y, z, f" WP{i}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Part 3 — 3D End-Effector Trajectory")
    ax.legend(); plt.tight_layout(); plt.show()

    # (c) Task-space pose, velocity, acceleration (from measured EE)
    def diff_by_time(tvec, Y):
        return np.gradient(Y, tvec, axis=0)

    X = ee_meas[:, 0:4]     # [x,y,z,pitch] (pitch as alpha)
    V = diff_by_time(t, X)  # [mm/s, mm/s, mm/s, deg/s]
    A = diff_by_time(t, V)  # [mm/s^2, ..., deg/s^2]

    labels = ["x (mm)", "y (mm)", "z (mm)", "alpha (deg)"]
    vlabels = ["x_dot (mm/s)", "y_dot (mm/s)", "z_dot (mm/s)", "alpha_dot (deg/s)"]
    alabels = ["x_dot_dot (mm/s^2)", "y_dot_dot (mm/s^2)", "z_dot_dot (mm/s^2)", "alpha_dot_dot (deg/s^2)"]
    styles = ["-", "-", "-", "-"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i in range(4):
        axs[0].plot(t, X[:, i], styles[i], label=labels[i])
    axs[0].set_ylabel("Pose"); axs[0].set_title("Task-space pose, velocity, acceleration")
    axs[0].grid(True); axs[0].legend(ncol=2)

    for i in range(4):
        axs[1].plot(t, V[:, i], styles[i], label=vlabels[i])
    axs[1].set_ylabel("Velocity"); axs[1].grid(True); axs[1].legend(ncol=2)

    for i in range(4):
        axs[2].plot(t, A[:, i], styles[i], label=alabels[i])
    axs[2].set_ylabel("Acceleration"); axs[2].set_xlabel("Time (s)")
    axs[2].grid(True); axs[2].legend(ncol=2)

    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # Collect data
    collect_data()
    # Plot data
    plot_data()