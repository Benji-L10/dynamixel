"""
Lab 8 (Final Project): Vision-Guided Robotic Pick-and-Place Sorting System
Team: RoboSquad
Members: Bryan Lew, Ben Ly

This script implements a complete robotic sorting system that:
1. Detects colored balls using computer vision
2. Localizes them in 3D space using camera-robot calibration
3. Plans smooth trajectories to pick up each ball
4. Sorts them into color-coded bins

System Architecture:
    Detection → Localization → Motion Planning → Execution → Repeat
"""

import numpy as np
import cv2
import time

from classes.Robot import Robot
from classes.Realsense import Realsense
from classes.TrajPlanner import TrajPlanner

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters
BALL_RADIUS = 15  # Physical radius of balls in millimeters

# Motion control parameters
TRAJECTORY_TIME = 2.5  # Time for each trajectory segment in seconds
NUM_POINTS = 100       # Number of waypoints in each trajectory

# Workspace safety bounds (millimeters, in robot frame)
# TODO: Adjust these based on your setup to prevent collisions
X_MIN, X_MAX = 50, 230   # Forward/backward limits
Y_MIN, Y_MAX = -150, 150 # Left/right limits

# Home position: [x, y, z, pitch] in mm and degrees
# This position should give the camera a clear view of the workspace
HOME_POSITION = [100, 0, 220, -15]

# Sorting bin locations: [x, y, z, pitch] in mm and degrees
# TODO: Adjust these positions based on your physical bin locations
BINS = {
    'red': [0, -220, 150, -40],
    'orange': [120, -220, 150, -40],
    'blue': [0, 220, 150, -45],
    'yellow': [120, 220, 150, -45]
}

# ============================================================================
# COMPUTER VISION: BALL DETECTION AND POSE ESTIMATION
# ============================================================================

def get_ball_pose(corners: np.ndarray, intrinsics: any, radius: float) -> tuple:
    """
    Estimate the 3D pose of a detected sphere using the Perspective-n-Point (PnP) algorithm.
    Args:
        corners: 4x2 array of boundary points (pixels). Expected order: [left, right, bottom, top].
        intrinsics: Camera intrinsics (fx, fy, ppx, ppy, [optional dist coeffs])
        radius: Physical sphere radius in mm
    Returns:
        (rot_matrix, tvec) with tvec in mm (camera frame)
    """
    # --- 3D object points on the sphere "equator" (match order of 'corners') ---
    # left, right, bottom, top
    object_points = np.array(
        [[-radius, 0.0, 0.0],
         [ radius, 0.0, 0.0],
         [ 0.0, -radius, 0.0],
         [ 0.0,  radius, 0.0]],
        dtype=np.float32
    )

    # --- 2D image points (pixels) ---
    image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)

    # --- Camera matrix (fx, fy, cx, cy) ---
    K = np.array([[intrinsics.fx, 0.0,           intrinsics.ppx],
                  [0.0,           intrinsics.fy, intrinsics.ppy],
                  [0.0,           0.0,           1.0]], dtype=np.float32)

    # Distortion coefficients (RealSense intrinsics often near zero for our use)
    try:
        dist = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)
    except Exception:
        dist = np.zeros((5, 1), dtype=np.float32)

    # --- Solve PnP ---
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed for sphere pose")

    rot_matrix, _ = cv2.Rodrigues(rvec)
    return rot_matrix, tvec


def detect_balls(image):
    """
    Detect colored balls in the input image using computer vision.
    Returns:
        list[(color, (cx, cy), radius)] or None
    """
    # --- Preprocess ---
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- Color ranges (tunable) ---
    ranges = {
        "red":    [(np.array([0,   120, 80]), np.array([10, 255, 255])),
                   (np.array([170, 120, 80]), np.array([180, 255, 255]))],
        "orange": [(np.array([10,  120, 80]), np.array([22, 255, 255]))],
        "yellow": [(np.array([22,   80, 80]), np.array([35, 255, 255]))],
        "blue":   [(np.array([90,   80, 80]), np.array([130,255, 255]))],
    }

    circles_all = []   # aggregate circle candidates as (cx,cy,r,color_mask_name)

    # Segment by color, then find circular blobs by contours (robust vs Hough)
    kernel = np.ones((5, 5), np.uint8)
    for cname, cranges in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in cranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # ignore tiny specks
                continue
            perim = cv2.arcLength(cnt, True)
            if perim <= 0:
                continue
            circ = 4*np.pi*area/(perim**2)
            if circ < 0.70:  # keep reasonably circular
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            r  = int(np.sqrt(area/np.pi))
            circles_all.append((cx, cy, r, cname))

    if not circles_all:
        cv2.imshow('Detection', image)
        cv2.waitKey(1)
        return None

    # Convert to Hough-like array shape and then classify via masked hue mean
    circles = np.array([[c[:3] for c in circles_all]], dtype=np.int32)  # (1, N, 3)

    result = []
    for (cx, cy, r), (_, _, _, cname) in zip(circles[0], circles_all):
        # Build a circular mask to compute mean hue inside the circle
        mask_circle = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(mask_circle, (int(cx), int(cy)), int(r*0.8), 255, -1)
        h_vals = hsv[:, :, 0][mask_circle == 255]
        s_vals = hsv[:, :, 1][mask_circle == 255]
        v_vals = hsv[:, :, 2][mask_circle == 255]
        if h_vals.size == 0:
            continue

        h_mean = np.mean(h_vals)
        s_mean = np.mean(s_vals)
        v_mean = np.mean(v_vals)

        # Basic sanity (avoid gray/dark)
        if s_mean < 50 or v_mean < 50:
            continue

        # Classification using hue + the initial color label from segmentation
        color = None
        if cname == "red"   and (h_mean < 15 or h_mean > 165): color = "red"
        if cname == "orange" and 10 <= h_mean <= 26:            color = "orange"
        if cname == "yellow" and 22 <= h_mean <= 40:            color = "yellow"
        if cname == "blue"   and 90 <= h_mean <= 135:           color = "blue"
        if color is None:
            continue

        result.append((color, (int(cx), int(cy)), int(r)))

        # Draw overlays
        cv2.circle(image, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.circle(image, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(image, color, (int(cx-20), int(cy-r-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow('Detection', image)
    cv2.waitKey(1)
    return result if result else None


# ============================================================================
# MOTION CONTROL: TRAJECTORY PLANNING AND EXECUTION
# ============================================================================

def move_trajectory(robot, target_pos, traj_time=TRAJECTORY_TIME):
    """
    Smooth task-space trajectory (linear blend in task space) executed via IK.
    """
    # --- Current joint & EE pose ---
    readings = robot.get_joints_readings()
    q_now = readings[0, :]                # deg
    p_now = robot.get_ee_pos(q_now)[:4]   # [x,y,z,pitch]

    # --- Build task-space waypoints (NUM_POINTS) ---
    target_pos = np.asarray(target_pos, dtype=float)
    p_now = np.asarray(p_now, dtype=float)
    waypoints = np.column_stack([
        np.linspace(p_now[i], target_pos[i], NUM_POINTS) for i in range(4)
    ])  # shape (NUM_POINTS, 4)

    # --- Convert waypoints to joint space via IK ---
    q_traj = []
    for p in waypoints:
        q = robot.get_ik(p.tolist())      # deg (4,)
        q_traj.append(q)
    q_traj = np.array(q_traj)

    # --- Timing and execution ---
    dt = traj_time / max(NUM_POINTS - 1, 1)
    robot.write_time(dt)   # set controller time per segment if applicable

    t0 = time.time()
    for i, qcmd in enumerate(q_traj):
        t_target = t0 + i * dt
        # Wait to maintain uniform rate
        while True:
            now = time.time()
            if now >= t_target: break
            time.sleep(max(0.0, t_target - now))
        robot.write_joints(qcmd)


# ============================================================================
# PICK AND PLACE OPERATIONS
# ============================================================================

def pick_ball(robot, ball_pos):
    """
    Execute a pick operation to grasp a ball.
    """
    print(f"Picking ball at {ball_pos}")
    robot.write_gripper(1)
    time.sleep(0.5)

    approach = [ball_pos[0], ball_pos[1], 100, -80]
    move_trajectory(robot, approach, TRAJECTORY_TIME)

    grasp = [ball_pos[0], ball_pos[1], 39, -80]   # tune to your table height
    move_trajectory(robot, grasp, TRAJECTORY_TIME * 0.5)

    robot.write_gripper(0)
    time.sleep(1)

    lift = [ball_pos[0], ball_pos[1], 100, -80]
    move_trajectory(robot, lift, TRAJECTORY_TIME * 0.5)


def place_ball(robot, color):
    """
    Place ball in the appropriate color-coded bin.
    """
    print(f"Placing {color} ball")
    bin_pos = BINS[color]
    move_trajectory(robot, bin_pos, TRAJECTORY_TIME)
    robot.write_gripper(1)
    time.sleep(1)


def go_home(robot):
    move_trajectory(robot, HOME_POSITION, TRAJECTORY_TIME)


# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================

def main():
    """
    Main control loop for the robotic sorting system.
    """
    print("="*60)
    print("Lab 8: Robotic Sorting System")
    print("="*60)

    # --- Initialize robot & camera ---
    robot = Robot()
    camera = Realsense()
    intrinsics = camera.get_intrinsics()

    # --- Load camera→robot transform ---
    T_cam_to_robot = np.load('camera_robot_transform.npy')

    # --- Position mode & initial posture ---
    robot.write_mode("position")
    robot.write_time(2.0)
    q_home = robot.get_ik(HOME_POSITION)
    robot.write_joints(q_home)
    time.sleep(2.5)
    robot.write_gripper(1)

    print(f"\nReady! Using TRAJECTORY control")
    print("Press Ctrl+C to stop\n")

    try:
        iteration = 0
        while True:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")

            # --- capture frame ---
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                print("No camera frame")
                time.sleep(0.2)
                continue

            # --- detect balls ---
            spheres = detect_balls(color_frame)
            if spheres is None:
                print("No balls detected")
                time.sleep(1)
                iteration += 1
                continue

            print(f"Detected {len(spheres)} ball(s)")

            # --- convert to robot frame ---
            robot_spheres = []
            for color, (cx, cy), radius in spheres:
                # build boundary points (px) in order: left, right, bottom, top
                corners = np.array([
                    [cx - radius, cy],
                    [cx + radius, cy],
                    [cx, cy + radius],
                    [cx, cy - radius]
                ], dtype=np.float32)

                # PnP → camera frame (mm)
                _, tvec = get_ball_pose(corners, intrinsics, BALL_RADIUS)
                cam_pos = tvec.reshape(3)

                # camera → robot (homogeneous)
                pos_h = np.hstack([cam_pos, 1.0]).reshape(4, 1)
                rob_h = T_cam_to_robot @ pos_h
                robot_pos = rob_h[:3, 0].astype(float)

                # workspace check
                if not (X_MIN <= robot_pos[0] <= X_MAX and
                        Y_MIN <= robot_pos[1] <= Y_MAX):
                    print(f"  Skipping {color} ball outside workspace: {robot_pos}")
                    continue

                robot_spheres.append((color, robot_pos))
                print(f"  {color}: {robot_pos}")

            if not robot_spheres:
                print("No balls in workspace")
                time.sleep(1)
                iteration += 1
                continue

            # --- pick & place the first valid ball ---
            color, pos = robot_spheres[0]
            print(f"\nSorting {color} ball at {pos}")
            pick_ball(robot, pos)
            place_ball(robot, color)
            go_home(robot)

            iteration += 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    except Exception as e:
        print(f"\nError occurred: {e}")
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


# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()