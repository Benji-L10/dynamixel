# Miguel Floran, Ben Ly, Electrical Engineering Department, Cal Poly
# Lab Assignment 2: Forward Kinematics of Robot Manipulators
# 2. Implement the forward kinematics in Python

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
from classes.Robot import Robot

np.set_printoptions(precision=3, suppress=True)


def main():
    # Home traj time
    traj_time_h = 3.0  # sec

    robot = Robot()

    # Enable torque and set time-based profile
    robot.write_motor_state(True)

    # Home
    print("Homing to [0, 0, 0, 0] deg ...")
    robot.write_time(traj_time_h)
    robot.write_joints([0, 0, 0, 0])
    time.sleep(traj_time_h)

    # Commanded motion of all joints
    traj_time_s = 5.0
    robot.write_time(traj_time_s)
    poll_dt = 0.5  # sec
    # target_deg = np.deg2rad([0, 0, 0, 0])  # deg
    # target_deg = [15, -45, -60, 90]   # deg
    target_deg = [-90, 15, 30, -45]  # deg
    print(f"\nMoving all joints to {target_deg} deg over {traj_time_s:.1f}s ...")

    times = []
    joints = []

    # print(robot.get_fk([0, 0, 0, 0]).__repr__())
    print(robot.get_fk(np.deg2rad(target_deg)).__repr__())
    print(robot.get_ee_pos(np.deg2rad(target_deg)).__repr__())

    # Sweep joints
    print(f"\nMoving all joints to {target_deg} deg over {traj_time_s:.1f}s ...")
    t0 = time.perf_counter()
    robot.write_joints(target_deg)
    while True:
        t_now = time.perf_counter() - t0
        readings = robot.get_joints_readings()
        times.append(t_now)
        joints.append(readings)
        # print_readings(readings)
        if t_now >= traj_time_s:
            break
    time.sleep(poll_dt)

    # Shutdown
    robot.close()


if __name__ == "__main__":
    main()
