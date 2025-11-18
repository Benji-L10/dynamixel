# Miguel Floran, Ben Ly, Electrical Engineering Department, Cal Poly
# Lab Assignment 2: Forward Kinematics of Robot Manipulators
# 4. Motion Planning in Joint Space (Waypoint-based 2D motion planning)

"""
1. Formulate the forward kinematics of the robot arm using DH convention.
2. Develop Python scripts and methods to implement forward kinematics equations.
3. Use the joint configuration received from the robot and the forward kinematics solution to
calculate the pose of the end-effctor.
4. Implement point-to-point motion between setpoints in the joint space.
5. Visualize and characterize task-space motion trajectories.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from classes.Robot import Robot, JointPose
from utils import save_to_pickle


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

    # Waypoints forming a triangular motion in joint space (deg)
    waypoints_deg: JointPose = np.array(
        [
            [0, -45, 60, 50],  # 1
            [0, 10, 50, -45],  # 2
            [0, 10, 0, -80],  # 3
            [0, -45, 60, 50],  # 4 (return to 1)
        ],
        np.float64,
    )

    # Commanded motion of all joints
    traj_time_s = 3.0  # seconds per segment
    robot.write_time(traj_time_s)

    # Data collection
    t_list, q_list, xyz_list, waypoints_s = [], [], [], []
    # Global start timestamp
    t_start = time.perf_counter()
    print("\nStarting waypoint motion sequence...")

    # Command each segment
    for i, q_end in enumerate(waypoints_deg):
        print(f"\nSegment {i}: {q_end} over {traj_time_s:.1f}s")
        # Sweep joints
        robot.write_joints(q_end)
        # Timestamp for the start of the segment
        t_segment = time.perf_counter()
        while ((t_now := time.perf_counter()) - t_segment) < traj_time_s:
            t_list.append(t_now - t_start)
            q_list.append(robot.get_joints_readings()[0, :].copy().tolist())
            xyz_list.append(robot.get_current_fk()[:3, 3].copy().tolist())
        # Record the waypoint time at the end of each segment
        waypoints_s.append(t_now - t_start)

    # Shutdown
    robot.close()

    # Convert to arrays
    timestamps_s = np.array(t_list, dtype=np.float64) # (N,)
    joint_data = np.array(q_list, dtype=np.float64) # (N, 4)
    xyz_data = np.array(xyz_list, dtype=np.float64) # (N, 3)
    waypoints_s = np.array(waypoints_s, dtype=np.float64) # (N,)

    # Save data to a pickle file
    save_to_pickle(
        {
            "timestamps_s": timestamps_s, # (N,)
            "joint_deg": joint_data,
            "xyz_mm": xyz_data,
            "waypoints_deg": waypoints_deg,
            "waypoints_s": waypoints_s,
        },
        f"robot_data.pkl",
    )


if __name__ == "__main__":
    main()
