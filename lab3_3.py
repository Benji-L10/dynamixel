# Miguel Floran, Ben Ly, Electrical Engineering Department, Cal Poly
# Lab Assignment 3: Inverse Kinematics of Robot Manipulators
# 3. Interface and validate your IK solution with the robot

"""
Write a Python script named lab3_3.py to validate your inverse kinematics implementation with
the physical robot. The script will command the robot through task-space waypoints forming a tri-
angular path, similar to Lab 2 Part 4 but using IK to plan motion in task space rather than joint space.
Move the robot through the waypoint sequence in Table 2 (start and end at Waypoint 1). Set the
trajectory time to 5 sec between waypoints. For each waypoint:
    Call get_ik() with the desired task-space pose (x;y;z;alpha) to compute joint angles
    Issue a single write_joints() command with the computed angles
    Record data continuously while the motion executes using time.perf_counter() times-
tamps
"""

import time
import numpy as np
from classes.Robot import Robot, JointPose
from utils import save_to_pickle

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

    # Waypoints forming a triangular motion in task space
    waypoints_taskspace: JointPose = np.array(
        [
            [25, -100, 150, -60],
            [150, 80, 300, 0],
            [250, -115, 75, -45],
            [25, -100, 150, -60],
        ],
        dtype=np.float64,
    )

    waypoints_jointspace = np.array(
        [robot.get_ik(waypoint) for waypoint in waypoints_taskspace],
        dtype=np.float64,
    )

    waypoints_taskspace_calc = np.array(
        [robot.get_ee_pos((waypoint)) for waypoint in waypoints_jointspace],
        dtype=np.float64,
    )

    # Commanded motion of all joints
    traj_time_s = 3.0  # seconds per segment
    robot.write_time(traj_time_s)

    # Move to the first waypoint before starting the timed sequence
    print(f"\nMoving to first waypoint: {waypoints_jointspace[0]} ...")
    robot.write_joints(waypoints_jointspace[0])
    time.sleep(traj_time_s)


    # Data collection
    t_list, q_list, waypoints_s = [], [], []
    # Global start timestamp
    t_start = time.perf_counter()
    print("\nStarting waypoint motion sequence...")

    # Command each segment while skipping the first waypoint (already there)
    for i, q_end in enumerate(waypoints_jointspace[1:]):
        print(f"\nSegment {i}: {q_end} over {traj_time_s:.1f}s")
        # Sweep joints
        robot.write_joints(q_end)
        # Timestamp for the start of the segment
        t_segment = time.perf_counter()
        while ((t_now := time.perf_counter()) - t_segment) < traj_time_s:
            t_list.append(t_now - t_start)
            q_list.append(robot.get_joints_readings())
        # Record the waypoint time at the end of each segment
        waypoints_s.append(t_now - t_start)

    # Shutdown
    robot.close()

    # Convert to arrays
    timestamps_s = np.array(t_list, dtype=np.float64)  # (N,)
    joint_deg = np.array(q_list, dtype=np.float64)  # (N, 4)
    waypoints_s = np.array(waypoints_s, dtype=np.float64)  # (N,)

    # Save data to a pickle file
    save_to_pickle(
        {
            "timestamps_s": timestamps_s,  # (N,)
            "joint_deg": joint_deg,
            "waypoints_taskspace": waypoints_taskspace,
            "waypoints_jointspace": waypoints_jointspace,
            "waypoints_taskspace_calc": waypoints_taskspace_calc,
            "waypoints_s": waypoints_s,
        },
        f"robot_data.pkl",
    )


if __name__ == "__main__":
    main()
