"""
(c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
Lab 7 Starter Code: Position-Based Visual Servoing with PID Control
Implements real-time visual servoing to track an AprilTag target.
"""

import numpy as np
import cv2
import time

from classes.Robot import Robot
from classes.Realsense import Realsense
from classes.AprilTags import AprilTags

class PIDController:
    """
    PID controller for position-based visual servoing.
    """
    def __init__(self, dim=3, dt=0.05):
        """
        Initialize PID controller.
        
        Args:
            dim: Dimension of control (3 for x, y, z)
            dt: Control timestep in seconds (MUST match control loop timing!)
        """
        # =============================================================================
        # TODO: Initialize PID gains
        # Hint: Start with small values and tune accordingly
        # YOUR CODE HERE
        self.Kp = 1.7 * np.eye(dim)     # Proportional
        self.Ki = 0.0 * np.eye(dim)     # Integral
        self.Kd = 0.0 * np.eye(dim)     # Derivative
        
        
        # Initialize error tracking variables
        self.error_integral = np.zeros(dim)
        self.error_prev = np.zeros(dim)
        self.dt = dt
        
    def compute(self, error):
        """
        Compute PID control output.
        
        Args:
            error: Position error vector (mm)
            
        Returns:
            control_output: Velocity command (mm/s)
        """
        # =============================================================================
        # TODO: Implement PID control computation (from Pre-Lab 7)
        # Hint: output = Kp*error + Ki*integral + Kd*derivative
        # Don't forget to update error_integral and error_prev!
        # YOUR CODE HERE
        # Ensure 1D NumPy array
        e = np.asarray(error, dtype=float).ravel()

        # Integral term: accumulate error over time
        self.error_integral += e * self.dt

        # Derivative term: change in error over time
        e_dot = (e - self.error_prev) / self.dt

        # Store for next call
        self.error_prev = e.copy()

        # PID law: u = Kp e + Ki ∫e dt + Kd de/dt
        p_term = self.Kp @ e
        i_term = self.Ki @ self.error_integral
        d_term = self.Kd @ e_dot

        output = p_term + i_term + d_term
        return output


def main():
    """
    Main visual servoing control loop.
    """
    print("="*60)
    print("Lab 7: Position-Based Visual Servoing")
    print("="*60)
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    print("\nInitializing system...")
    
    # TODO: Initialize robot, camera, and AprilTag detector
    # Hint: Create Robot(), Realsense(), and AprilTags() instances
    # YOUR CODE HERE
    robot = Robot()
    camera = Realsense()
    detector = AprilTags()
    
    
    # TODO: Get camera intrinsics
    # Hint: Use camera.get_intrinsics()
    # YOUR CODE HERE
    intrinsics = camera.get_intrinsics()
    
    
    # TODO: Define control timestep (CRITICAL - must match PID dt!)
    # Hint: 0.05 seconds = 20 Hz, 0.02 seconds = 50 Hz
    # YOUR CODE HERE
    dt = 0.05  # Control loop period in seconds
    
    
    # TODO: Set AprilTag physical size in millimeters
    # Hint: Measure your actual tag!
    # YOUR CODE HERE
    TAG_SIZE = 40.0  # Update this value
    
    
    # TODO: Initialize PID controller with SAME dt as control loop
    # Hint: pid = PIDController(dim=3, dt=dt)
    # YOUR CODE HERE
    pid = PIDController(dim=3, dt=dt)
    
    
    # TODO: Load camera-robot calibration matrix
    # Hint: Use np.load('camera_robot_transform.npy')
    # YOUR CODE HERE
    try:
        T_cam_to_robot = np.load('camera_robot_transform.npy')
        print("Calibration loaded successfully")
    except FileNotFoundError:
        print("Error: camera_robot_transform.npy not found!")
        print("Please run lab6_2.py first to calibrate.")
        return
    
    
    # TODO: Define desired offset from tag (mm)
    # Hint: This defines where end-effector should be relative to tag
    # Example: [0, 0, 50] means 50mm above tag
    # YOUR CODE HERE
    target_offset = np.array([0, 0, 50])  # Adjust as needed
    print(f"Target offset from tag: {target_offset} mm")
    
    
    # =========================================================================
    # MOVE TO START POSITION
    # =========================================================================
    print("\nMoving to start position...")
    
    # Set position mode and trajectory time
    robot.write_mode("position")
    traj_time = 3.0
    robot.write_time(traj_time)
    
    # Start position: [x, y, z, gripper_angle] in mm and degrees
    start_position = [100, 0, 220, -15]
    start_joints = robot.get_ik(start_position)  # Replace with IK solution

    # TODO: Move to start position: write joints and wait for motion to complete
    # Hint: Use robot.get_ik(), robot.write_joints(), time.sleep()
    # YOUR CODE HERE
    robot.write_joints(start_joints)
    time.sleep(traj_time + 0.5)
    
    # TODO: Switch robot to velocity control mode
    # Hint: Use robot.write_mode("velocity")
    # YOUR CODE HERE
    robot.write_mode("velocity")
    
    print("Robot ready for visual servoing")
    
    
    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================
    print("\nStarting visual servoing control loop...")
    print("Press 'q' to quit\n")
    
    iteration = 0
    
    try:
        while True:
            # TODO: Record start time for fixed timestep enforcement
            # Hint: start_time = time.time()
            # YOUR CODE HERE
            start_time = time.time()
            
            
            # -----------------------------------------------------------------
            # STEP 1: CAPTURE FRAME AND DETECT TAG
            # -----------------------------------------------------------------
            
            # Get camera frame
            color_frame, _ = camera.get_frames()
            if color_frame is None:
                continue
            
            # TODO: Detect AprilTags in frame
            # Hint: Use detector.detect_tags(color_frame)
            # YOUR CODE HERE
            tags = detector.detect_tags(color_frame)
            
            
            # Check if any tags detected
            if len(tags) > 0:
                tag = tags[0]  # Use first detected tag
                
                # Draw tag detection on image for visualization
                color_frame = detector.draw_tags(color_frame, tag)
                
                # -----------------------------------------------------------------
                # STEP 2: GET TAG POSE IN CAMERA FRAME
                # -----------------------------------------------------------------
                
                # TODO: Get tag pose using PnP
                # Hint: detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
                # Returns: (rotation_matrix, translation_vector)
                # YOUR CODE HERE
                rot_matrix, trans_vector = detector.get_tag_pose(tag.corners, intrinsics, TAG_SIZE)
                
                
                # TODO: Extract tag position in camera frame (already in mm)
                # Hint: Flatten trans_vector to get 1D array
                # YOUR CODE HERE
                tag_pos_camera = trans_vector.flatten()
                
                
                # -----------------------------------------------------------------
                # STEP 3: TRANSFORM TO ROBOT FRAME
                # -----------------------------------------------------------------
                
                # TODO: Convert to homogeneous coordinates
                # Hint: Append 1 to make [x, y, z, 1]
                # YOUR CODE HERE
                tag_pos_camera_hom = np.hstack([tag_pos_camera, 1.0])  # (4,)
                
                
                # TODO: Apply camera-to-robot transformation
                # YOUR CODE HERE
                tag_pos_robot_hom = T_cam_to_robot @ tag_pos_camera_hom.reshape(4, 1)
                
                
                # TODO: Extract 3D position from homogeneous coordinates
                # Hint: Take first 3 elements
                # YOUR CODE HERE
                tag_pos_robot = tag_pos_robot_hom[:3, 0]
                
                
                # -----------------------------------------------------------------
                # STEP 4: CALCULATE DESIRED END-EFFECTOR POSITION
                # -----------------------------------------------------------------
                
                # TODO: Add offset to tag position to get desired EE position
                # Hint: desired_ee_pos = tag_pos_robot + target_offset
                # YOUR CODE HERE
                desired_ee_pos = tag_pos_robot + target_offset
                
                
                # -----------------------------------------------------------------
                # STEP 5: GET CURRENT END-EFFECTOR POSITION
                # -----------------------------------------------------------------
                
                # TODO: Get current joint positions
                # YOUR CODE HERE
                readings = robot.get_joints_readings()  # 3x4
                current_joints = readings[0, :]         # deg                
                
                # TODO: Get current end-effector position using forward kinematics
                # Hint: robot.get_ee_pos(current_joints) returns [x, y, z, ...]
                # Take only first 3 elements (position)
                # YOUR CODE HERE
                current_ee_pos = robot.get_ee_pos(current_joints)[:3]
                
                
                # -----------------------------------------------------------------
                # STEP 6: CALCULATE POSITION ERROR
                # -----------------------------------------------------------------
                
                # TODO: Compute position error
                # Hint: error = desired_position - current_position
                # YOUR CODE HERE
                error = (desired_ee_pos - current_ee_pos).astype(float)
                
                
                # -----------------------------------------------------------------
                # STEP 7: COMPUTE PID CONTROL OUTPUT
                # -----------------------------------------------------------------
                
                # TODO: Use PID controller to compute velocity command
                # Hint: Call pid.compute()
                # YOUR CODE HERE
                velocity_cmd = pid.compute(error)  # mm/s
                
                
                # -----------------------------------------------------------------
                # STEP 8: CONVERT TO JOINT VELOCITIES
                # -----------------------------------------------------------------
                
                # TODO: Get robot Jacobian at current configuration
                # Hint: robot.get_jacobian()
                # YOUR CODE HERE
                J = robot.get_jacobian(current_joints)  # 6x4, linear part in mm
                
                
                # TODO: Extract position part of Jacobian (first 3 rows)
                # Hint: J_linear = J[:3, :]
                # YOUR CODE HERE
                J_linear = J[:3, :]  # 3x4
                
                
                # TODO: Compute joint velocities using pseudo-inverse
                # Hint: joint_vel = pinv(J_linear) @ velocity_cmd
                # Use np.linalg.pinv()
                # YOUR CODE HERE
                qdot_rad = np.linalg.pinv(J_linear) @ velocity_cmd.reshape(3, 1)  # rad/s
                qdot_deg = np.rad2deg(qdot_rad).reshape(4).tolist()               # deg/s              
                
                # -----------------------------------------------------------------
                # STEP 9: COMMAND ROBOT
                # -----------------------------------------------------------------
                
                # TODO: Send joint velocities to robot
                # Hint: robot.write_velocities()
                # OpenManipulator-X has 4 joints (excluding gripper)
                # YOUR CODE HERE
                robot.write_velocities(qdot_deg)
                
                # -----------------------------------------------------------------
                # STEP 10: DISPLAY STATUS
                # -----------------------------------------------------------------
                
                # Print status every 40 iterations (~2 seconds at 20Hz)
                if iteration % 40 == 0:
                    print(f"\nIteration: {iteration}")
                    print(f"Tag position (robot): {tag_pos_robot}")
                    print(f"Current EE position:  {current_ee_pos}")
                    print(f"Desired EE position:  {desired_ee_pos}")
                    print(f"Error: {error} mm")
                    print(f"Error magnitude: {np.linalg.norm(error):.2f} mm")
                
            else:
                # -----------------------------------------------------------------
                # NO TAG DETECTED - STOP ROBOT
                # -----------------------------------------------------------------
                
                # TODO: Stop robot motion by sending zero velocities
                # Hint: robot.write_velocities([0, 0, 0, 0])
                # YOUR CODE HERE
                robot.write_velocities([0, 0, 0, 0])
                
                if iteration % 40 == 0:
                    print("\nNo AprilTag detected - robot stopped")
            
            
            # -----------------------------------------------------------------
            # DISPLAY AND USER INTERACTION
            # -----------------------------------------------------------------
            
            # Display camera image
            cv2.imshow('Visual Servoing', color_frame)
            key = cv2.waitKey(1)
            
            # Check for quit key press ('q' or ESC)
            if key & 0xFF == ord('q') or key == 27:
                print("\nQuitting...")
                break
            
            iteration += 1
            
            
            # -----------------------------------------------------------------
            # MAINTAIN FIXED TIMESTEP (CRITICAL!)
            # -----------------------------------------------------------------
            
            # Enforce consistent loop timing
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # =====================================================================
        # CLEANUP
        # =====================================================================
        print("\nStopping robot and cleaning up...")
        robot.write_velocities([0, 0, 0, 0])
        camera.stop()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()