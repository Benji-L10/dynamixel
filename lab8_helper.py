"""
Lab 8 (Final Project): Vision-Guided Robotic Pick-and-Place Sorting System
Team: [Your Team Name]
Members: [Team Member Names]

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
    """
    # ---------- OBJECT POINTS (equator points of the sphere) ----------
    object_points = np.array(
        [
            [-radius, 0.0, 0.0],  # left
            [ radius, 0.0, 0.0],  # right
            [ 0.0,   radius, 0.0],# top
            [ 0.0,  -radius, 0.0] # bottom
        ],
        dtype=np.float32
    )

    # ---------- IMAGE POINTS ----------
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    # ---------- CAMERA INTRINSICS ----------
    K = np.array(
        [
            [intrinsics.fx, 0.0,           intrinsics.ppx],
            [0.0,           intrinsics.fy, intrinsics.ppy],
            [0.0,           0.0,           1.0]
        ],
        dtype=np.float32
    )
    try:
        dist = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)
    except Exception:
        dist = np.zeros((5, 1), dtype=np.float32)

    # ---------- SOLVE PnP ----------
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")

    rot_matrix, _ = cv2.Rodrigues(rvec)
    return rot_matrix, tvec


def detect_balls(image):
    """
    Detect colored balls in the input image using the SAME method as prelab8.py:
      - HSV color threshold per color range
      - Morphological OPEN/CLOSE cleanup
      - Contour filtering by area & circularity
      - Centroid & radius from area

    Args:
        image: BGR color image from camera

    Returns:
        list of (color_str, (cx, cy), radius) OR None if none found
    """
    if image is None:
        return None

    annotated = image.copy()

    # Slight blur to reduce noise (same as prelab8)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # HSV conversion (same as prelab8)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- Use the SAME HSV ranges as prelab8.py ---
    color_ranges = {
        "red": [
            (np.array([0,   120, 80]), np.array([7, 255, 255])),   # lower red
            (np.array([170, 120, 80]), np.array([180, 255, 255]))   # upper red
        ],
        "orange": [
            (np.array([7,  120, 80]), np.array([22, 255, 255]))
        ],
        "yellow": [
            (np.array([22,  80, 80]), np.array([35, 255, 255]))
        ],
        "blue": [
            (np.array([90,  80, 80]), np.array([130, 255, 255]))
        ],
    }

    results = []

    # For each color → threshold → morph → find circular contours (prelab8 style)
    for color_name, ranges in color_ranges.items():
        # Combine masks for multi-range colors (red)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)

        # Morphology cleanup (same: OPEN then CLOSE)
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # Find external contours
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # ignore tiny blobs (same threshold as prelab8)
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)

            # Keep reasonably circular blobs (same threshold as prelab8)
            if circularity < 0.7:
                continue

            # Centroid from image moments
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Approximate radius from area
            radius = int(np.sqrt(area / np.pi))

            # Save detection
            results.append((color_name, (cx, cy), radius))

            # Draw same viz as prelab8
            cv2.circle(annotated, (cx, cy), radius, (0, 255, 0), 2)  # outline
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)      # centroid
            cv2.putText(
                annotated, color_name, (cx - 20, cy - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

    # Show for debugging (optional; comment out if you don’t want a window)
    cv2.imshow('Detection', annotated)
    cv2.waitKey(1)

    return results if results else None


# ============================================================================
# MOTION CONTROL: TRAJECTORY PLANNING AND EXECUTION
# ============================================================================

def move_trajectory(robot, target_pos, traj_time=TRAJECTORY_TIME):
    """
    Move robot to target position using smooth quintic trajectory.
    """
    # Get current joint positions and EE pose
    readings = robot.get_joints_readings()
    q_now = readings[0, :]
    p_now = robot.get_ee_pos(q_now)[:4]

    # Plan task-space quintic trajectory (use TrajPlanner API consistent with Lab 4)
    waypoints = np.vstack([p_now, np.asarray(target_pos, dtype=float)])
    planner = TrajPlanner(waypoints)                       # __init__(setpoints)
    p_traj = planner.get_quintic_traj(traj_time, NUM_POINTS)  # returns [t, x, y, z, pitch]

    # Convert to joint space (drop time column if present)
    p_traj_xyzp = p_traj[:, 1:5] if p_traj.shape[1] >= 5 else p_traj
    q_traj = np.array([robot.get_ik(p.tolist()) for p in p_traj_xyzp])

    # Use planner timing if available, else compute uniform dt
    if p_traj.shape[1] >= 5:
        dt = float(p_traj[1, 0] - p_traj[0, 0])
    else:
        dt = traj_time / max(NUM_POINTS - 1, 1)
    robot.write_time(dt)

    # Execute with precise timing (Lab 4 method)
    t0 = time.time()
    for i, qcmd in enumerate(q_traj):
        t_target = t0 + i * dt
        while True:
            now = time.time()
            if now >= t_target:
                break
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

    # Open gripper
    robot.write_gripper(1)
    time.sleep(0.5)

    # Approach
    approach = [(ball_pos[0] + 20), ball_pos[1], 100, -80]
    move_trajectory(robot, approach, TRAJECTORY_TIME)

    # Grasp
    grasp = [(ball_pos[0] + 20), ball_pos[1], 30, -80]
    move_trajectory(robot, grasp, TRAJECTORY_TIME * 0.5)
    robot.write_gripper(0)
    time.sleep(1)

    # Lift
    lift = [(ball_pos[0] + 20), ball_pos[1], 100, -80]
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
    """
    Return robot to home position for next detection cycle.
    """
    move_trajectory(robot, HOME_POSITION, TRAJECTORY_TIME)
