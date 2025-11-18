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
from classes.Robot import Robot, JointPose

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
            [274, 0, 204, 0],
            [16, 4, 336, 15],
            [0, -270, 106, 0],
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

    # Data collection
    t_list, q_list, waypoints_s = [], [], []
    # Global start timestamp
    t_start = time.perf_counter()
    print("\nStarting waypoint motion sequence...")

    # Command each segment
    for i in range(len(waypoints_taskspace)):
        print(f"Task Space Waypoint {i}: {waypoints_taskspace[i]}")
        print(f"Joint Space Waypoint {i}: {waypoints_jointspace[i]}")
        print(f"Calculated Task Space Waypoint {i}: {waypoints_taskspace_calc[i]}")


    # Shutdown
    robot.close()


if __name__ == "__main__":
    main()
