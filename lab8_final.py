# lab8_final.py
# Lab 8 (Option 2: PBVS) – main script

import time
import numpy as np
import cv2

from classes.Robot import Robot
from classes.Realsense import Realsense
from lab8_helper import (
    BALL_RADIUS_MM, DT,
    APPROACH_OFFSET_MM, GRASP_Z_MM, LIFT_Z_MM,
    detect_balls, get_ball_pose, PBVSController, move_trajectory
)

# ---------- Scenario config (lab-specific) ----------
# Workspace bounds (mm) to ignore outliers
X_MIN, X_MAX = 50, 230
Y_MIN, Y_MAX = -150, 150

# Home posture and bins: [x, y, z, pitch]
HOME_POSITION = [100, 0, 220, -15]
BINS = {
    'red':    [  0, -220, 150, -40],
    'orange': [120, -220, 150, -40],
    'blue':   [  0,  220, 150, -45],
    'yellow': [120,  220, 150, -45],
}

TRAJECTORY_TIME = 2.0  # s (for non-visual transport segments)

def place_ball(robot, color):
    """Open-loop transport to bin; PBVS not required once grasped."""
    print(f"Placing {color} ball")
    move_trajectory(robot, BINS[color], TRAJECTORY_TIME, num_pts=100)
    robot.write_gripper(1)  # release
    time.sleep(1.0)

def go_home(robot):
    move_trajectory(robot, HOME_POSITION, TRAJECTORY_TIME, num_pts=100)

def main():
    print("=" * 60)
    print("Lab 8 (Option 2): Position-Based Visual Servoing")
    print("=" * 60)

    # Devices
    robot = Robot()
    camera = Realsense()
    intrinsics = camera.get_intrinsics()

    # Calibration (camera → robot)
    T_cam_to_robot = np.load('camera_robot_transform.npy')

    # Start in position mode, go home, open gripper
    robot.write_mode("position")
    robot.write_time(2.0)
    q_home = robot.get_ik(HOME_POSITION)
    robot.write_joints(q_home)
    time.sleep(2.5)
    robot.write_gripper(1)
    time.sleep(0.5)

    pbvs = PBVSController(dt=DT)

    print("\nReady. Press Ctrl+C to stop.\n")

    try:
        iteration = 0
        while True:
            print(f"\n{'='*60}\nIteration {iteration}")

            # 1) Frame
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                print("No camera frame")
                time.sleep(0.2)
                iteration += 1
                continue

            # 2) Detect balls
            detections = detect_balls(color_frame)
            if not detections:
                print("No balls detected")
                time.sleep(0.5)
                iteration += 1
                continue

            # 3) Choose first in-workspace target and convert to robot frame
            target = None
            for color, (cx, cy), rpx in detections:
                # 4 equator points for PnP: left, right, bottom, top
                corners = np.array([
                    [cx - rpx, cy],
                    [cx + rpx, cy],
                    [cx,       cy + rpx],
                    [cx,       cy - rpx],
                ], dtype=np.float32)

                _, tvec = get_ball_pose(corners, intrinsics, BALL_RADIUS_MM)  # camera (mm)
                cam_pos = tvec.reshape(3)
                pos_h = np.hstack([cam_pos, 1.0]).reshape(4, 1)
                rob_h = T_cam_to_robot @ pos_h
                robot_pos = rob_h[:3, 0].astype(float)  # mm

                if X_MIN <= robot_pos[0] <= X_MAX and Y_MIN <= robot_pos[1] <= Y_MAX:
                    target = (color, (cx, cy, rpx), robot_pos)
                    print(f"Target {color} at robot {robot_pos}")
                    break

            if target is None:
                print("No valid targets in workspace")
                time.sleep(0.5)
                iteration += 1
                continue

            target_color, (cx0, cy0, r0), target_pos_robot = target

            # Switch to velocity mode for PBVS phases
            robot.write_mode("velocity")
            time.sleep(0.1)

            # ---- PBVS Phase A: APPROACH (keep visibility) ----
            print("PBVS: approach")
            t_start = time.time()
            while True:
                # Refresh target pose by re-detecting same color
                frame, _ = camera.get_frames()
                det = detect_balls(frame)
                if det:
                    same = [d for d in det if d[0] == target_color]
                    if same:
                        # closest to original centroid
                        cx, cy, rpx = min(((d[1][0], d[1][1], d[2]) for d in same),
                                          key=lambda P: (P[0]-cx0)**2 + (P[1]-cy0)**2)
                        corners = np.array([[cx - rpx, cy], [cx + rpx, cy],
                                            [cx, cy + rpx], [cx, cy - rpx]], dtype=np.float32)
                        _, tvec = get_ball_pose(corners, intrinsics, BALL_RADIUS_MM)
                        cam_pos = tvec.reshape(3)
                        pos_h = np.hstack([cam_pos, 1.0]).reshape(4, 1)
                        rob_h = T_cam_to_robot @ pos_h
                        target_pos_robot = rob_h[:3, 0].astype(float)

                p_des = target_pos_robot + APPROACH_OFFSET_MM
                e, qdot_deg = pbvs.step(robot, p_des)
                robot.write_velocities(qdot_deg.tolist())

                if np.linalg.norm(e) < 5.0:           # mm
                    break
                if time.time() - t_start > 6.0:       # timeout
                    break
                time.sleep(DT)

            # ---- PBVS Phase B: DESCEND to GRASP_Z ----
            print("PBVS: descend")
            t_start = time.time()
            while True:
                p_des = np.array([target_pos_robot[0], target_pos_robot[1], GRASP_Z_MM])
                e, qdot_deg = pbvs.step(robot, p_des)
                robot.write_velocities(qdot_deg.tolist())

                if np.linalg.norm(e) < 3.0:
                    break
                if time.time() - t_start > 4.0:
                    break
                time.sleep(DT)

            # Grasp
            robot.write_velocities([0, 0, 0, 0])
            robot.write_gripper(0)  # close
            time.sleep(1.0)

            # ---- PBVS Phase C: LIFT ----
            print("PBVS: lift")
            t_start = time.time()
            while True:
                p_des = np.array([target_pos_robot[0], target_pos_robot[1], LIFT_Z_MM])
                e, qdot_deg = pbvs.step(robot, p_des)
                robot.write_velocities(qdot_deg.tolist())

                if np.linalg.norm(e) < 4.0:
                    break
                if time.time() - t_start > 4.0:
                    break
                time.sleep(DT)

            # Stop PBVS and switch back to position mode for transport
            robot.write_velocities([0, 0, 0, 0])
            robot.write_mode("position")
            time.sleep(0.2)

            # Place + home (open-loop trajectories are fine here)
            place_ball(robot, target_color)
            go_home(robot)

            iteration += 1
            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        try:
            camera.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
