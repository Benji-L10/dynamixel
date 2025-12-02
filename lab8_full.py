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
    
    The PnP algorithm finds the position and orientation of an object by matching:
    - Known 3D points on the object (object_points)
    - Corresponding 2D points in the image (image_points)
    
    For a sphere, we create artificial "corner" points on the sphere's visible boundary
    to establish these correspondences.
    
    Args:
        corners: A 4x2 array of circle boundary points [top, bottom, left, right]
        intrinsics: Camera intrinsic parameters from RealSense
        radius: The sphere's physical radius in millimeters
    
    Returns:
        tuple: (rotation_matrix, translation_vector)
            - rotation_matrix: 3x3 matrix (not used for spheres, but returned by PnP)
            - translation_vector: 3x1 vector giving sphere center in camera frame (mm)
    
    Raises:
        RuntimeError: If PnP algorithm fails to find a solution
    """
    # ==========================================================================
    # TODO: Define 3D object points on the sphere
    # ==========================================================================
    # Hint: Create 4 points on the sphere's "equator" as viewed from camera
    # These should correspond to the top, bottom, left, and right boundary points
    # Example structure:
    #   Point 1: [-radius, 0, 0]  (left edge of sphere)
    #   Point 2: [+radius, 0, 0]  (right edge of sphere)
    #   Point 3: [0, +radius, 0]  (top edge of sphere)
    #   Point 4: [0, -radius, 0]  (bottom edge of sphere)
    # YOUR CODE HERE
    object_points = np.array(
        [
            [-radius, 0.0, 0.0],  # left
            [ radius, 0.0, 0.0],  # right
            [ 0.0,   radius, 0.0],# top
            [ 0.0,  -radius, 0.0] # bottom
        ],
        dtype=np.float32
    )
    
    
    # ==========================================================================
    # TODO: Prepare image points (the detected corner points in pixels)
    # ==========================================================================
    # Hint: Reshape corners array to (4, 2) and ensure float32 type
    # YOUR CODE HERE    
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    

    # ==========================================================================
    # TODO: Construct camera intrinsic matrix
    # ==========================================================================
    # The camera matrix format is:
    #   [[fx,  0, cx],
    #    [ 0, fy, cy],
    #    [ 0,  0,  1]]
    # where fx, fy are focal lengths and cx, cy are principal point coordinates
    # Hint: Access intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    # YOUR CODE HERE
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
    
    
    # ==========================================================================
    # TODO: Solve PnP to get rotation and translation
    # ==========================================================================
    # Hint: Use cv2.solvePnP()
    # Returns: (success, rvec, tvec) where rvec is rotation vector, tvec is translation
    # YOUR CODE HERE
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    
    # Convert rotation vector to rotation matrix (required return format)
    rot_matrix, _ = cv2.Rodrigues(rvec)
    
    return rot_matrix, tvec


def detect_balls(image):
    """
    Detect colored balls in the input image using computer vision.
    
    Pipeline:
        1. Convert image to HSV color space for robust color detection
        2. Detect circular shapes using Hough Circle Transform
        3. For each detected circle:
           - Extract the color by analyzing hue values
           - Classify as red, orange, yellow, or blue
           - Record position and radius
    
    Args:
        image: BGR color image from camera
    
    Returns:
        list: List of tuples (color, (cx, cy), radius) for each detected ball
              Returns None if no balls detected
    """
    # ==========================================================================
    # TODO: DETECT CIRCULAR OBJECTS IN IMAGE
    # ==========================================================================
    # Implement your circle detection pipeline here. Consider approaches such as:
    #   - Hough Circle Transform (cv2.HoughCircles)
    #   - Contour detection (cv2.findContours) with circularity filtering
    #   - Color-based segmentation followed by shape analysis
    #   - Edge detection + morphological operations
    #
    # Your implementation should:
    #   - Detect circles reliably under varying lighting conditions
    #   - Handle multiple balls of different colors
    #   - Return circle parameters (center x, y and radius) for each detection
    #
    # Recommended: Store results in a variable called 'circles' with format:
    #              array of shape (1, N, 3) where each circle is [x, y, radius]
    #              (This matches cv2.HoughCircles output format)
    #
    # Explore different preprocessing techniques and parameters!
    # YOUR CODE HERE
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=120,
        param2=30,
        minRadius=10,
        maxRadius=120
    )
    
    # If no circles detected, show image and return None
    if circles is None or len(circles[0]) == 0:
        cv2.imshow('Detection', image)
        cv2.waitKey(1)
        return None
    
    # Convert circle parameters to integers
    circles = np.uint16(np.around(circles))
    
    result = []  # List to store (color, center, radius) tuples
    
    # Process each detected circle
    for circle in circles[0, :]:
        center = (circle[0], circle[1])  # (cx, cy) in pixels
        radius = circle[2]               # radius in pixels

        # ======================================================================
        # TODO: CLASSIFY BALL COLOR
        # ======================================================================
        # Determine the color of this detected circle. Approaches to consider:
        #   - HSV color space analysis (hue values)
        #
        # Important: Focus analysis on pixels within the circle (using masks), not entire image
        #
        # Must classify as one of: 'red', 'orange', 'yellow', 'blue'
        # Set variable 'color' to the classification result
        # Use 'continue' to skip circles that don't match any target color
        #
        # Note: Red color may require special handling depending on color space!
        #
        # YOUR CODE HERE
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask_circle = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.circle(mask_circle, center, int(radius*0.8), 255, -1)
        h_vals = hsv[:, :, 0][mask_circle == 255]
        s_vals = hsv[:, :, 1][mask_circle == 255]
        v_vals = hsv[:, :, 2][mask_circle == 255]
        if h_vals.size == 0:
            continue
        h_mean = np.mean(h_vals)
        s_mean = np.mean(s_vals)
        v_mean = np.mean(v_vals)
        if s_mean < 50 or v_mean < 50:
            continue

        color = None
        # Handle red wrap-around
        if h_mean < 10 or h_mean > 170:
            color = 'red'
        elif 10 <= h_mean < 22:
            color = 'orange'
        elif 22 <= h_mean < 35:
            color = 'yellow'
        elif 90 <= h_mean < 130:
            color = 'blue'
        else:
            continue
        
        # Add to results
        result.append((color, center, radius))
        
        # ======================================================================
        # TODO: Draw detection visualization on image
        # ======================================================================
        # Hint: Draw circle outline, center point, and color label
        # Use cv2.circle() for shapes and cv2.putText() for text
        # YOUR CODE HERE
        cv2.circle(image, center, radius, (0, 255, 0), 2)  # Green circle
        cv2.circle(image, center, 3, (0, 0, 255), -1)      # Red center dot
        cv2.putText(image, color, (center[0] - 20, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display detection results
    cv2.imshow('Detection', image)
    cv2.waitKey(1)
    
    return result if result else None


# ============================================================================
# MOTION CONTROL: TRAJECTORY PLANNING AND EXECUTION
# ============================================================================

def move_trajectory(robot, target_pos, traj_time=TRAJECTORY_TIME):
    """
    Move robot to target position using smooth quintic trajectory.
    
    This function:
    1. Gets current robot position
    2. Plans a quintic (5th-order polynomial) trajectory in task space
    3. Converts entire trajectory to joint space using inverse kinematics
    4. Executes trajectory with precise timing
    
    Args:
        robot: Robot instance
        target_pos: Target position [x, y, z, pitch] in mm and degrees
        traj_time: Duration of trajectory in seconds
    """
    # ==========================================================================
    # TODO: Get current joint positions and end-effector position
    # ==========================================================================
    # YOUR CODE HERE
    readings = robot.get_joints_readings()
    q_now = readings[0, :]
    p_now = robot.get_ee_pos(q_now)[:4]
    
    
    # ==========================================================================
    # TODO: Create task-space trajectory from current to target position
    # ==========================================================================
    # Hint: Stack current_pos and target_pos, create TrajPlanner, call get_quintic_traj()
    # YOUR CODE HERE
    waypoints = np.vstack([p_now, np.asarray(target_pos, dtype=float)])
    planner = TrajPlanner(waypoints, traj_time, num_points=NUM_POINTS)
    p_traj = planner.get_quintic_traj()
    
    
    # ==========================================================================
    # TODO: Convert entire trajectory to joint space
    # ==========================================================================
    # YOUR CODE HERE
    q_traj = np.array([robot.get_ik(p.tolist()) for p in p_traj])
    
    
    # TODO: Calculate time step between waypoints
    dt = traj_time / max(NUM_POINTS - 1, 1)
    robot.write_time(dt)
    
    # ==========================================================================
    # TODO: Execute trajectory with precise timing (see Lab 4)
    # ==========================================================================
    # Hint: Record start time, then for each waypoint:
    #   - Calculate target time for this waypoint
    #   - Wait until that time
    #   - Send joint commands
    # This ensures smooth, consistent motion regardless of computation time
    # YOUR CODE HERE
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
    
    Sequence:
        1. Open gripper
        2. Move to approach position (above ball)
        3. Move down to grasp position
        4. Close gripper
        5. Lift ball to clear workspace
    
    Args:
        robot: Robot instance
        ball_pos: Ball position [x, y, z] in robot frame (mm)
    """
    print(f"Picking ball at {ball_pos}")
    
    # ==========================================================================
    # Open gripper
    # ==========================================================================
    # Use robot.write_gripper(1) for open, wait for motion
    robot.write_gripper(1)
    time.sleep(0.5)
    
    # ==========================================================================
    # Move to approach position (above the ball)
    # ==========================================================================
    # Create position [x, y, z_high, pitch] where z_high is ~100mm
    # Use steep pitch angle (e.g., -80°) for vertical approach
    approach = [ball_pos[0], ball_pos[1], 100, -80]   # may need tuning
    move_trajectory(robot, approach, TRAJECTORY_TIME)
    
    # ==========================================================================
    # Move down to grasp position
    # ==========================================================================
    # Lower z to just above table surface (e.g., 39mm for 30mm ball radius)
    # Adjust this height based on your table height and ball size!
    grasp = [ball_pos[0], ball_pos[1], 39, -80]       # may need tuning
    move_trajectory(robot, grasp, TRAJECTORY_TIME * 0.5)
    
    # ==========================================================================
    # Close gripper to grasp ball
    # ==========================================================================
    #  Use robot.write_gripper(0) for close, wait for secure grasp
    robot.write_gripper(0)
    time.sleep(1)
    
    # ==========================================================================
    # Lift ball to clear workspace
    # ==========================================================================
    # Move back up to approach height
    lift = [ball_pos[0], ball_pos[1], 100, -80] # may need tuning
    move_trajectory(robot, lift, TRAJECTORY_TIME * 0.5)


def place_ball(robot, color):
    """
    Place ball in the appropriate color-coded bin.
    
    Args:
        robot: Robot instance
        color: Ball color string ('red', 'orange', 'yellow', 'blue')
    """
    print(f"Placing {color} ball")
    
    # ==========================================================================
    # Get bin position for this color
    # ==========================================================================
    # Look up position in BINS dictionary
    bin_pos = BINS[color]
    
    # ==========================================================================
    # Move to bin location
    # ==========================================================================
    move_trajectory(robot, bin_pos, TRAJECTORY_TIME)
    
    # ==========================================================================
    # Release ball by opening gripper
    # ==========================================================================
    robot.write_gripper(1)
    time.sleep(1)


def go_home(robot):
    """
    Return robot to home position for next detection cycle.
    
    Args:
        robot: Robot instance
    """
    move_trajectory(robot, HOME_POSITION, TRAJECTORY_TIME)


# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================

def main():
    """
    Main control loop for the robotic sorting system.
    
    Workflow:
        1. Initialize robot, camera, and calibration
        2. Move to home position
        3. Loop:
           a. Capture image and detect balls
           b. Convert detected positions to robot frame
           c. Filter balls within workspace
           d. Pick and place first ball
           e. Repeat
    """
    print("="*60)
    print("Lab 8: Robotic Sorting System")
    print("="*60)
    
    # ==========================================================================
    # INITIALIZATION
    # ==========================================================================
    
    # TODO: Initialize robot, camera, and get intrinsics
    # Hint: Create Robot(), Realsense(), and get intrinsics
    # YOUR CODE HERE
    robot = Robot()
    camera = Realsense()
    intrinsics = camera.get_intrinsics()
    
    
    # ==========================================================================
    # TODO: Load camera-robot calibration matrix
    # ==========================================================================
    # Hint: Use np.load() to load 'camera_robot_transform.npy'
    # This matrix transforms points from camera frame to robot frame
    # YOUR CODE HERE
    T_cam_to_robot = np.load('camera_robot_transform.npy')
    
    
    # ==========================================================================
    # TODO: Setup robot in position control mode
    # ==========================================================================
    # Hint: Set mode to "position", enable motors, set default trajectory time
    # YOUR CODE HERE
    robot.write_mode("position")
    robot.write_time(2.0)
    
    
    # ==========================================================================
    # TODO: Move to home position
    # ==========================================================================
    # Hint: Use inverse kinematics to find joint angles, then command them
    # YOUR CODE HERE
    q_home = robot.get_ik(HOME_POSITION)
    robot.write_joints(q_home)
    time.sleep(2.5)
    
    
    # ==========================================================================
    # TODO: Open gripper initially
    # ==========================================================================
    # YOUR CODE HERE
    robot.write_gripper(1)
    time.sleep(0.5)
    
    
    print(f"\nReady! Using TRAJECTORY control")
    print("Press Ctrl+C to stop\n")
    
    # ==========================================================================
    # MAIN CONTROL LOOP
    # ==========================================================================
    
    try:
        iteration = 0
        
        while True:
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}")
            
            # ==================================================================
            # STEP 1: CAPTURE IMAGE AND DETECT BALLS
            # ==================================================================
            
            # TODO: Get camera frame
            # Hint: Use camera.get_frames() which returns (color, depth)
            # YOUR CODE HERE
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                print("No camera frame")
                time.sleep(0.2)
                iteration += 1
                continue
            
            
            # TODO: Detect balls in image
            # Hint: Call detect_balls() function
            # YOUR CODE HERE
            spheres = detect_balls(color_frame)
            
            # Check if any balls detected
            if spheres is None:
                print("No balls detected")
                time.sleep(1)
                iteration += 1
                continue
            
            print(f"Detected {len(spheres)} ball(s)")
            
            # ==================================================================
            # STEP 2: CONVERT DETECTIONS TO ROBOT FRAME
            # ==================================================================
            
            robot_spheres = []  # List to store (color, robot_position) tuples
            
            for color, (cx, cy), radius in spheres:
                
                # ==============================================================
                # TODO: Create corner points for PnP algorithm
                # ==============================================================
                # Hint: Create 4 points at [left, right, bottom, top] of circle
                # Format: [[cx - radius, cy], [cx + radius, cy], ...]
                # YOUR CODE HERE
                corners = np.array(
                    [
                        [cx - radius, cy],   # left
                        [cx + radius, cy],   # right
                        [cx, cy + radius],   # bottom
                        [cx, cy - radius],   # top
                    ],
                    dtype=np.float32
                )
                
                
                # ==============================================================
                # TODO: Get 3D position in camera frame using PnP
                # ==============================================================
                # Hint: Call get_ball_pose() with corners, intrinsics, and BALL_RADIUS
                # Returns (rotation, translation) - we only need translation
                # YOUR CODE HERE                
                _, tvec = get_ball_pose(corners, intrinsics, BALL_RADIUS)
                cam_pos = tvec.reshape(3)
                
                # ==============================================================
                # TODO: Transform position to robot frame
                # ==============================================================
                # Hint: 
                #   1. Flatten cam_pos and append 1 for homogeneous coordinates
                #   2. Multiply by transformation matrix: T_cam_to_robot @ pos_hom
                #   3. Extract first 3 elements for 3D position
                # YOUR CODE HERE
                pos_h = np.hstack([cam_pos, 1.0]).reshape(4, 1)
                rob_h = T_cam_to_robot @ pos_h
                robot_pos = rob_h[:3, 0].astype(float)
                
                
                # ==============================================================
                # Check if position is within workspace bounds
                # ==============================================================
                # Check if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX
                # Skip balls outside workspace for safety
                if not (X_MIN <= robot_pos[0] <= X_MAX and 
                        Y_MIN <= robot_pos[1] <= Y_MAX):
                    print(f"  Skipping {color} ball outside workspace: {robot_pos}")
                    continue
                
                
                robot_spheres.append((color, robot_pos))
                print(f"  {color}: {robot_pos}")
            
            # Check if any valid balls found
            if not robot_spheres:
                print("No balls in workspace")
                time.sleep(1)
                iteration += 1
                continue
            
            # ==================================================================
            # STEP 3: PICK AND PLACE FIRST BALL
            # ==================================================================
            
            # Get first ball from list
            color, pos = robot_spheres[0]
            
            print(f"\nSorting {color} ball at {pos}")
            
            # TODO: Execute pick-and-place sequence
            # Hint: Call pick_ball(), place_ball(), and go_home()
            # YOUR CODE HERE
            pick_ball(robot, pos)
            place_ball(robot, color)
            go_home(robot)
            
            
            iteration += 1
            time.sleep(1)  # Brief pause before next cycle
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        # TODO: stop the camera
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