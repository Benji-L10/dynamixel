# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Starter script for OpenManipulator-X Robot for EE 471

import time

from classes.Robot import Robot
from utils import print_readings

"""
This script demonstrates basic operation of the OpenManipulator-X via the Robot class.
It initializes the robot, sets a time-based move profile, moves the base joint
through a few waypoints while printing live joint readings, and toggles the gripper.
"""


def main():
    traj_time = 3.0  # sec
    poll_dt = 0.5  # sec
    base_waypoints = [-45, 45, 0]  # deg

    robot = Robot()

    # Enable torque and set time-based profile
    robot.write_motor_state(True)
    robot.write_time(traj_time)

    # Home
    print("Homing to [0, 0, 0, 0] deg ...")
    robot.write_joints([0, 0, 0, 0])
    time.sleep(traj_time)

    # Sweep base joint
    for wp in base_waypoints:
        print(f"\nMoving base to {wp} deg over {traj_time:.1f}s ...")
        robot.write_joints([wp, 0, 0, 0])
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < traj_time:
            print_readings(robot.get_joints_readings())
            time.sleep(poll_dt)

    # Toggle gripper 2 times
    for i in range(2):
        is_open = robot.read_gripper_open()
        print(f"[{i+1}/2] is_open (before) = {is_open}")
        robot.write_gripper(not is_open)
        time.sleep(1.0)
        is_open = robot.read_gripper_open()
        print(f"[{i+1}/2] is_open (after)  = {is_open}")

    print("\nGripper toggled 2 times. Demo complete.")

    # Shutdown
    robot.close()


if __name__ == "__main__":
    main()
