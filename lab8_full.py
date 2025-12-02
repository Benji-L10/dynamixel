"""
Lab 8 (Final Project): Vision-Guided Robotic Pick-and-Place Sorting System
Option 2: Position-Based Visual Servoing (PBVS)

Team: [Your Team Name]
Members: [Team Member Names]

Pipeline:
  Detect balls (color + image circle) →
  PnP for 3D center in camera frame →
  Transform to robot base frame →
  PBVS (PID in task space) for approach / grasp / lift →
  Position trajectories for transport to bins and home.

Requires:
  - classes/Robot.py (OpenManipulator-X interface)
  - classes/Realsense.py (RealSense RGB stream + intrinsics)
  - classes/TrajPlanner.py (optional; used here only for simple position segments via IK)
  - camera_robot_transform.npy (Lab 6.2 result)
"""

import numpy as np
import cv2
import time

from classes.Robot import Robot
from classes.Realsense import Realsense
from classes.TrajPlanner import TrajPlanner

# =============================================================================
# CONFIGURATION
# =============================================================================

# Physical ball radius in millimeters (measure your set)
BALL_RADIUS = 15.0

# Control loop timing
DT = 0.05  # 20 Hz

# PID gains for PBVS (start with Lab 7 tuned values)
Kp = np.diag([1.0, 1.0, 1.0])
Ki = np.diag([0.10, 0.10, 0.10])
Kd = np.diag([0.10, 0.10, 0.10])

# Joint safety (deg/s)
VEL_CLAMP = 60.0
VEL_DEADBAND = 0.5

# Workspace bounds (mm, robot frame) to ignore outliers
X_MIN, X_MAX = 50, 230
Y_MIN, Y_MAX = -150, 150

# Home posture: [x, y, z, pitch] (mm, mm, mm, deg)
HOME_POSITION = [100, 0, 220, -15]

# Bins: [x, y, z, pitch] (mm, mm, mm, deg)
BINS = {
    'red':    [  0, -220, 150, -40],
    'orange': [120, -220, 150, -40],
    'blue':   [  0,  220, 150, -45],
    'yellow': [120,  220, 150, -45],
}

# Approach/Grasp/Lift targets relative to ball (mm)
APPROACH_OFFSET = np.array([0.0, 0.0, 80.0])  # above ball to keep visibility
GRASP_Z        = 39.0                         # table height + margin
LIFT_Z         = 100.0

# Trajectory settings for non-visual transport segments
TRAJECTORY_TIME = 2.0
NUM_POINTS = 100


# =============================================================================
# VISION HELPERS
# =============================================================================

def get_ball_pose(corners_px: np.ndarray, intrinsics, radius_mm: float):
    """
    Estimate ball center using PnP on 4 "equator" points of the image circle.
    corners_px: 4x2 pixel points in order [left, right, bottom, top].
    intrinsics: RealSense intrinsics object (fx, fy, ppx, ppy, coeffs)
    radius_mm:  sphere radius (mm)
    Returns:
        rot_matrix (3x3), tvec (3x1) in camera frame (mm)
    """
    # 3D object points on sphere equator (match corners order)
    object_points = np.array(
        [[-radius_mm,   0.0, 0.0],   # left
         [ radius_mm,   0.0, 0.0],   # right
         [ 0.0,       -radius_mm, 0.0],   # bottom
         [ 0.0,        radius_mm, 0.0]],  # top
        dtype=np.float32
    )

    image_points = np.asarray(corners_px, dtype=np.float32).reshape(-1, 2)

    K = np.array([[intrinsics.fx, 0.0,            intrinsics.ppx],
                  [0.0,           intrinsics.fy,  intrinsics.ppy],
                  [0.0,           0.0,            1.0]], dtype=np.float32)

    try:
        dist = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)
    except Exception:
        dist = np.zeros((5, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed for sphere pose")

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec  # tvec is (3,1) mm


def detect_balls(image_bgr):
    """
    Detect colored balls via HSV segmentation + contour circularity.
    Returns: list[(color_str, (cx, cy), r_px)]  OR  None if none found.
    Also draws overlays to a "Detection" window for operator feedback.
    """
    blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    ranges = {
        "red":    [(np.array([  0, 120, 80]), np.array([ 10, 255, 255])),
                   (np.array([170, 120, 80]), np.array([180, 255, 255]))],
        "orange": [(np.array([ 10, 120, 80]), np.array([ 22, 255, 255]))],
        "yellow": [(np.array([ 22,  80, 80]), np.array([ 35, 255, 255]))],
        "blue":   [(np.array([ 90,  80, 80]), np.array([130, 255, 255]))],
    }

    kernel = np.ones((5, 5), np.uint8)
    results = []

    for cname, spans in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in spans:
            mask |= cv2.inRange(hsv, lo, hi)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 200:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * np.pi * area / (perim ** 2)
            if circ < 0.70:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            r  = int(np.sqrt(area / np.pi))

            # quick sanity using local HSV stats (avoid dull/gray blobs)
            mask_circle = np.zeros_like(mask)
            cv2.circle(mask_circle, (cx, cy), int(r * 0.8), 255, -1)
            h = hsv[:, :, 0][mask_circle == 255]
            s = hsv[:, :, 1][mask_circle == 255]
            v = hsv[:, :, 2][mask_circle == 255]
            if h.size == 0 or np.mean(s) < 50 or np.mean(v) < 50:
                continue

            results.append((cname, (cx, cy), r))

            # overlays
            cv2.circle(image_bgr, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(image_bgr, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(image_bgr, cname, (cx - 20, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Detection", image_bgr)
    cv2.waitKey(1)

    return results if results else None


# =============================================================================
# UTILS: SIMPLE POSITION TRAJECTORY (USED ONLY FOR TRANSPORT/HOME)
# =============================================================================

def move_trajectory(robot: Robot, target_pos, traj_time=TRAJECTORY_TIME, num_pts=NUM_POINTS):
    """
    Simple linear interpolation in task space executed via IK.
    Used for non-visual transport segments (to bins and home).
    """
    readings = robot.get_joints_readings()
    q_now = readings[0, :]                # deg
    p_now = robot.get_ee_pos(q_now)[:4]   # [x,y,z,pitch]

    target_pos = np.asarray(target_pos, dtype=float)
    p_now = np.asarray(p_now, dtype=float)

    waypoints = np.column_stack([
        np.linspace(p_now[i], target_pos[i], num_pts) for i in range(4)
    ])  # (num_pts, 4)

    dt = traj_time / max(num_pts - 1, 1)
    robot.write_time(dt)

    t0 = time.time()
    for i in range(num_pts):
        q_cmd = robot.get_ik(waypoints[i].tolist())
        t_target = t0 + i * dt
        while True:
            now = time.time()
            if now >= t_target: break
            time.sleep(max(0.0, t_target - now))
        robot.write_joints(q_cmd)


def place_ball(robot: Robot, color: str):
    """Open-loop transport to bin; PBVS is not required once grasped."""
    print(f"Placing {color} ball")
    move_trajectory(robot, BINS[color], TRAJECTORY_TIME)
    robot.write_gripper(1)  # release
    time.sleep(1.0)


def go_home(robot: Robot):
    move_trajectory(robot, HOME_POSITION, TRAJECTORY_TIME)


# =============================================================================
# PBVS CORE
# =============================================================================

class PBVSController:
    """
    PID in task space: v = Kp e + Ki int(e) + Kd de/dt
    Velocity allocation: qdot = J_v^# v
    """
    def __init__(self, dt=DT, Kp=Kp, Ki=Ki, Kd=Kd, vel_clamp=VEL_CLAMP, deadband=VEL_DEADBAND):
        self.dt = dt
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.vel_clamp = vel_clamp
        self.deadband = deadband

        self.e_int = np.zeros(3)
        self.e_prev = np.zeros(3)

    def step(self, robot: Robot, p_des_mm: np.ndarray):
        # current joints/pose
        readings = robot.get_joints_readings()
        q_deg = readings[0, :]
        p_cur = robot.get_ee_pos(q_deg)[:3]   # mm

        # PID
        e = (p_des_mm - p_cur).astype(float)  # mm
        self.e_int += e * self.dt
        e_dot = (e - self.e_prev) / self.dt
        self.e_prev = e.copy()

        v = (self.Kp @ e) + (self.Ki @ self.e_int) + (self.Kd @ e_dot)  # mm/s

        # Jacobian + allocation
        J = robot.get_jacobian(q_deg)[:3, :]   # 3x4
        Jpinv = np.linalg.pinv(J, rcond=1e-3)
        qdot_rad = (Jpinv @ v.reshape(3, 1)).reshape(4)

        # safety shaping
        qdot_deg = np.degrees(qdot_rad)
        qdot_deg = np.clip(qdot_deg, -self.vel_clamp, self.vel_clamp)
        qdot_deg[np.abs(qdot_deg) < self.deadband] = 0.0

        return e, qdot_deg


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Lab 8 (Option 2): Position-Based Visual Servoing")
    print("=" * 60)

    # --- Initialize devices ---
    robot = Robot()
    camera = Realsense()
    intrinsics = camera.get_intrinsics()

    # --- Calibration: camera → robot transform ---
    T_cam_to_robot = np.load('camera_robot_transform.npy')

    # --- Move to home in position mode ---
    robot.write_mode("position")
    robot.write_time(2.0)
    q_home = robot.get_ik(HOME_POSITION)
    robot.write_joints(q_home)
    time.sleep(2.5)

    # --- Open gripper to start ---
    robot.write_gripper(1)
    time.sleep(0.5)

    print("\nReady. Press Ctrl+C to stop.\n")

    pbvs = PBVSController(dt=DT, Kp=Kp, Ki=Ki, Kd=Kd)

    try:
        iteration = 0
        while True:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")

            # 1) Grab a frame
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

            # 3) Convert first in-workspace ball to robot frame
            target = None
            for color, (cx, cy), rpx in detections:
                # build 4 corners for PnP (left, right, bottom, top)
                corners = np.array([
                    [cx - rpx, cy],
                    [cx + rpx, cy],
                    [cx,       cy + rpx],
                    [cx,       cy - rpx]
                ], dtype=np.float32)

                _, tvec = get_ball_pose(corners, intrinsics, BALL_RADIUS)
                cam_pos = tvec.reshape(3)                         # mm
                pos_h = np.hstack([cam_pos, 1.0]).reshape(4, 1)
                rob_h = T_cam_to_robot @ pos_h
                robot_pos = rob_h[:3, 0].astype(float)            # mm

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

            # 4) Switch to velocity mode for PBVS phases
            robot.write_mode("velocity")
            time.sleep(0.1)

            # ---- PBVS Phase A: APPROACH (keep tag visible) ----
            print("PBVS: approach")
            t_start = time.time()
            while True:
                # Re-detect target color to refresh target_pos_robot
                frame, _ = camera.get_frames()
                det = detect_balls(frame)
                if det:
                    same = [d for d in det if d[0] == target_color]
                    # Pick closest in pixel space to original centroid
                    if same:
                        cx, cy, rpx = min(((d[1][0], d[1][1], d[2]) for d in same),
                                          key=lambda P: (P[0]-cx0)**2 + (P[1]-cy0)**2)
                        corners = np.array([[cx - rpx, cy], [cx + rpx, cy],
                                            [cx, cy + rpx], [cx, cy - rpx]], dtype=np.float32)
                        _, tvec = get_ball_pose(corners, intrinsics, BALL_RADIUS)
                        cam_pos = tvec.reshape(3)
                        pos_h = np.hstack([cam_pos, 1.0]).reshape(4, 1)
                        rob_h = T_cam_to_robot @ pos_h
                        target_pos_robot = rob_h[:3, 0].astype(float)

                p_des = target_pos_robot + APPROACH_OFFSET
                e, qdot_deg = pbvs.step(robot, p_des)
                robot.write_velocities(qdot_deg.tolist())

                if np.linalg.norm(e) < 5.0:  # mm
                    break
                if time.time() - t_start > 6.0:
                    break
                time.sleep(DT)

            # ---- PBVS Phase B: DESCEND to GRASP_Z ----
            print("PBVS: descend")
            t_start = time.time()
            while True:
                p_des = np.array([target_pos_robot[0], target_pos_robot[1], GRASP_Z])
                e, qdot_deg = pbvs.step(robot, p_des)
                robot.write_velocities(qdot_deg.tolist())

                if np.linalg.norm(e) < 3.0:
                    break
                if time.time() - t_start > 4.0:
                    break
                time.sleep(DT)

            # Grasp
            robot.write_velocities([0, 0, 0, 0])
            robot.write_gripper(0)   # close
            time.sleep(1.0)

            # ---- PBVS Phase C: LIFT to LIFT_Z ----
            print("PBVS: lift")
            t_start = time.time()
            while True:
                p_des = np.array([target_pos_robot[0], target_pos_robot[1], LIFT_Z])
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

            # 5) Transport to bin and release
            place_ball(robot, target_color)

            # 6) Return home
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
        print("Done!")


if __name__ == "__main__":
    main()
