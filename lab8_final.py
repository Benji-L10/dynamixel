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

# bring in all helpers/constants exactly as written
from lab8_helper import (
    BALL_RADIUS, TRAJECTORY_TIME, NUM_POINTS,
    X_MIN, X_MAX, Y_MIN, Y_MAX,
    HOME_POSITION, BINS,
    get_ball_pose, detect_balls,
    move_trajectory, pick_ball, place_ball, go_home
)

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
