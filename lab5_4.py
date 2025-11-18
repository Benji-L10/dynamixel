# Ben Ly, Electrical Engineering Department, Cal Poly
#Lab Assignment 5: Velocity Kinematics of Robot Manipulators
# 4. Velocity-based motion planning in task space

"""
Implement velocity-based trajectory tracking using inverse velocity kinematics. Unlike Lab 4 poly-
nomial trajectory generation, this approach commands joint velocities in real time to move the end-
eﬀector at constant speed toward target positions.
"""

import numpy as np
import time
import pickle
from classes.Robot import Robot

def collect_data():
    """
    Collect data for velocity-based trajectory tracking.
    Moves the robot through a triangular path using velocity control.
    """
    
    # =============================================================================
    # SETUP AND INITIALIZATION
    # =============================================================================
    
    # Create robot object
    robot = Robot()
    
    # Define task-space waypoints [x, y, z, alpha] in mm and degrees
    ee_poses = np.array([
        [25, -100, 150, 0],    # Waypoint 1
        [150, 80, 300, 0],     # Waypoint 2
        [250, -115, 75, 0],    # Waypoint 3
        [25, -100, 150, 0]     # Return to Waypoint 1
    ])
    
    # TODO: Compute IK for all waypoints
    # Store results in joint_angles array (4x4)
    print("Computing IK for waypoints...")
    joint_angles = np.zeros((len(ee_poses), 4))
    for i in range(len(ee_poses)):
        joint_angles[i, :] = robot.get_ik(ee_poses[i, :])

    # =============================================================================
    # CONTROL PARAMETERS
    # =============================================================================
    
    velocity_des = 50.0      # Desired task-space speed (mm/s)
    tolerance = 5.0          # Convergence tolerance (mm)
    max_joint_vel = 45.0     # Maximum joint velocity limit (deg/s) for safety
    
    print(f"\nControl parameters:")
    print(f"  Desired velocity: {velocity_des} mm/s")
    print(f"  Tolerance: {tolerance} mm")
    
    # =============================================================================
    # DATA STORAGE PRE-ALLOCATION
    # =============================================================================
    
    # Pre-allocate arrays for data collection (over-allocate for safety)
    max_samples = 10000
    data_time = np.zeros(max_samples)
    data_q = np.zeros((max_samples, 4))              # Joint angles (deg)
    data_q_dot = np.zeros((max_samples, 4))          # Joint velocities (deg/s)
    data_ee_pos = np.zeros((max_samples, 5))         # End-effector pose [x,y,z,pitch,yaw]
    data_ee_vel_cmd = np.zeros((max_samples, 3))     # Commanded EE velocity (mm/s)
    data_ee_vel_actual = np.zeros((max_samples, 6))  # Actual EE velocity (mm/s, rad/s)
    count = 0  # Sample counter
    
    
    # =============================================================================
    # ROBOT INITIALIZATION
    # =============================================================================
    
    print("\nInitializing robot...")
    # TODO: Enable motors
    robot.write_motor_state(True)
    
    # TODO: Move to starting position using position control
    print("Moving to start position...")
    robot.write_mode("position")
    robot.write_time(3.0)
    robot.write_joints(joint_angles[0, :])
    time.sleep(5.0)  # small guard (seconds)
    
    # TODO: Switch to velocity control mode
    # Hint: Use robot.write_mode("velocity")
    print("\nSwitching to velocity control mode...")
    robot.write_mode("velocity")
    
    
    # =============================================================================
    # VELOCITY-BASED TRAJECTORY TRACKING
    # =============================================================================
    
    print("\nStarting velocity-based trajectory tracking...")
    start_time = time.perf_counter()
    
    # Loop through waypoints 2, 3, 4 (indices 1, 2, 3)
    for i in range(1, len(ee_poses)):
        
        # Extract target position (first 3 elements: x, y, z)
        target = ee_poses[i][:3]
        print(f"\n--- Moving to Waypoint {i+1}: {target} ---")
        
        # Initialize distance to target
        distance = np.inf
        iteration = 0
        
        # Continue until within tolerance of target
        while distance > tolerance:
            
            loop_start = time.perf_counter()
            
            # -----------------------------------------------------------------
            # STEP 1: READ CURRENT STATE
            # -----------------------------------------------------------------
            
            # TODO: Read current joint angles and velocities
            readings = robot.get_joints_readings()    # shape (3,4): [deg; deg/s; mA]
            q_deg = readings[0, :].astype(float)
            q_dot_deg = readings[1, :].astype(float)
            currents_mA = readings[2, :].astype(float)
            
            # TODO: Convert joint velocities from deg/s to rad/s
            q_dot_rad = np.deg2rad(q_dot_deg)
            
            # TODO: Get current end-effector pose
            ee_pose = robot.get_ee_pos(q_deg)         # [x,y,z,pitch,yaw]
            current_pos = ee_pose[:3]
            
            
            # -----------------------------------------------------------------
            # STEP 2: COMPUTE DISTANCE AND DIRECTION TO TARGET
            # -----------------------------------------------------------------
            
            # TODO: Compute error vector (target - current position)
            error = target - current_pos
            
            # TODO: Compute distance to target (norm of error vector)
            distance = float(np.linalg.norm(error))
            
            # TODO: Compute unit direction vector
            # Hint: direction = error / distance (avoid division by zero)
            direction = (error / distance) if distance > 1e-9 else np.zeros(3)
            
            # -----------------------------------------------------------------
            # STEP 3: GENERATE DESIRED VELOCITY
            # -----------------------------------------------------------------
            
            # TODO: Scale direction by desired speed
            # (optionally slow near target; keeping constant per starter)
            speed = velocity_des
            v_des = speed * direction  # (3,)
            
            # TODO: Form 6D desired velocity vector [v_x, v_y, v_z, omega_x, omega_y, omega_z]
            # Hint: Stack v_des with zeros for angular velocity
            p_dot_des = np.hstack([v_des, np.zeros(3)])  # (6,)
            
            
            # -----------------------------------------------------------------
            # STEP 4: INVERSE VELOCITY KINEMATICS
            # -----------------------------------------------------------------
            
            # TODO: Get Jacobian at current configuration
            J = robot.get_jacobian(q_deg)  # (6x4)
            
            # TODO: Compute pseudo-inverse of Jacobian
            # Use damped least squares for robustness near singularities
            lam = 1e-3
            JT = J.T
            J_pinv = JT @ np.linalg.inv(J @ JT + (lam**2) * np.eye(6))
            
            # TODO: Compute required joint velocities (rad/s)
            q_dot_cmd_rad = J_pinv @ p_dot_des  # (4,)
            
            # TODO: Convert joint velocities from rad/s to deg/s
            q_dot_cmd_deg = np.rad2deg(q_dot_cmd_rad)
            
            # Safety clamp
            q_dot_cmd_deg = np.clip(q_dot_cmd_deg, -max_joint_vel, max_joint_vel)
            
            
            # -----------------------------------------------------------------
            # STEP 5: SEND VELOCITY COMMAND TO ROBOT
            # -----------------------------------------------------------------
            robot.write_velocities(q_dot_cmd_deg)
            
            
            # -----------------------------------------------------------------
            # STEP 6: VERIFY WITH FORWARD VELOCITY KINEMATICS
            # -----------------------------------------------------------------
            # Prefer robot.get_fwd_vel_kin if available; otherwise J*qdot
            if hasattr(robot, "get_fwd_vel_kin"):
                p_dot_actual = robot.get_fwd_vel_kin(
                    np.deg2rad(q_deg), q_dot_rad, current_ma=currents_mA
                )
            else:
                p_dot_actual = J @ q_dot_rad  # fallback (6,)
            
            
            # -----------------------------------------------------------------
            # STEP 7: DATA COLLECTION
            # -----------------------------------------------------------------
            
            if count < max_samples:
                data_time[count] = time.perf_counter() - start_time
                data_q[count, :] = q_deg
                data_q_dot[count, :] = q_dot_deg
                data_ee_pos[count, :] = ee_pose
                data_ee_vel_cmd[count, :] = v_des
                # ensure shape (6,) for storage
                data_ee_vel_actual[count, :6] = np.asarray(p_dot_actual).reshape(-1)[:6]
                count += 1
            
            iteration += 1
            
            # Small sleep to avoid saturating the bus/CPU; ~200 Hz loop
            dt = time.perf_counter() - loop_start
            if dt < 0.005:
                time.sleep(0.005 - dt)
        
        # End of while loop - target reached
        print(f"  Reached Waypoint {i+1}! Final distance: {distance:.2f} mm")
        
        # TODO: Stop robot briefly at waypoint
        # Hint: Send zero velocities and sleep briefly
        robot.write_velocities([0.0, 0.0, 0.0, 0.0])
        time.sleep(1.0)
    
    
    # =============================================================================
    # CLEANUP AND DATA SAVING
    # =============================================================================
    
    # TODO: Stop robot completely
    print("\nTrajectory complete! Stopping robot...")
    robot.write_velocities([0.0, 0.0, 0.0, 0.0])
    robot.close() # Shutdown

    total_time = time.perf_counter() - start_time
    print(f"\nTotal execution time: {total_time:.2f} s")
    print(f"Total samples collected: {count}")
    print(f"Average sample rate: {count/total_time:.1f} Hz")
    
    # Trim unused portions of pre-allocated arrays
    data_time = data_time[:count]
    data_q = data_q[:count, :]
    data_q_dot = data_q_dot[:count, :]
    data_ee_pos = data_ee_pos[:count, :]
    data_ee_vel_cmd = data_ee_vel_cmd[:count, :]
    data_ee_vel_actual = data_ee_vel_actual[:count, :]
    
    # TODO: Save all data to pickle file
    filename='lab5_4_data.pkl'
    # Create dictionary with all collected data and control parameters
    print(f"\nSaving data to {filename}...")
    data_dict = {
        'time': data_time,
        'ee_waypoints': ee_poses,
        'ik_waypoints_deg': joint_angles,
        'joint_angles': data_q,
        'joint_velocities_deg_s': data_q_dot,
        'ee_pose_meas': data_ee_pos,                 # [x,y,z,pitch,yaw]
        'ee_vel_cmd_mm_s': data_ee_vel_cmd,          # [vx,vy,vz]
        'ee_vel_actual': data_ee_vel_actual,         # [vx,vy,vz,wx,wy,wz]
        'velocity_des_mm_s': velocity_des,
        'tolerance_mm': tolerance,
        'max_joint_vel_deg_s': max_joint_vel,
    }
    
    # TODO: Write dictionary to pickle file
    with open(filename, "wb") as f:
        pickle.dump(data_dict, f)
    
    print("Data saved successfully!")


def plot_data():
    """
    Load data from lab5_4_data.pkl and create required plots:
      (a) 3D end-effector trajectory with waypoint markers
      (b) Linear speed magnitude: commanded vs actual
      (c) Linear velocity components (vx, vy, vz): commanded vs actual
      (d) Angular velocity components (wx, wy, wz): actual (commanded = 0)
    """
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    fname = "lab5_4_data.pkl"
    if not os.path.exists(fname):
        print(f"[plot_data] File not found: {fname}")
        return

    with open(fname, "rb") as f:
        D = pickle.load(f)

    # Unpack with safe fallbacks
    t = np.asarray(D.get("time", []), float)
    ee_wps = np.asarray(D.get("ee_waypoints", []), float)           # (M,4)
    ee_pose = np.asarray(D.get("ee_pose_meas", []), float)          # (N,5) [x,y,z,pitch,yaw]
    v_cmd = np.asarray(D.get("ee_vel_cmd_mm_s", []), float)         # (N,3) [vx,vy,vz]
    p_dot = np.asarray(D.get("ee_vel_actual", []), float)           # (N,6) [vx,vy,vz,wx,wy,wz]

    if any(arr.size == 0 for arr in [t, ee_pose, v_cmd, p_dot]):
        print("[plot_data] Missing or empty arrays in pickle; nothing to plot.")
        return

    # -----------------------
    # Figure (a): 3D path
    # -----------------------
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_pose[:, 0], ee_pose[:, 1], ee_pose[:, 2], lw=1.6, label="Measured path")
    if ee_wps.size:
        ax.scatter(ee_wps[:, 0], ee_wps[:, 1], ee_wps[:, 2], s=60, marker="o", label="Waypoints")
        # label first three as WP1..WP3 if present
        for i in range(min(3, ee_wps.shape[0])):
            x, y, z = ee_wps[i, :3]
            ax.text(x, y, z, f" WP{i+1}", fontsize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("(a) 3D End-Effector Trajectory")
    ax.legend()
    plt.tight_layout()

    # -----------------------
    # Figure (b): speed mag
    # -----------------------
    v_cmd_mag = np.linalg.norm(v_cmd, axis=1)
    v_act_mag = np.linalg.norm(p_dot[:, 0:3], axis=1)
    plt.figure(figsize=(9.5, 4.8))
    plt.plot(t, v_cmd_mag, label=r"$\|\mathbf{v}_{cmd}\|$ (mm/s)", linewidth=1.6)
    plt.plot(t, v_act_mag, "--", label=r"$\|\mathbf{v}_{act}\|$ (mm/s)", linewidth=1.6, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Linear speed (mm/s)")
    plt.title("(b) Linear Speed: Commanded vs Actual")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # -----------------------------------------------
    # Figure (c): linear velocity components vx,vy,vz
    # -----------------------------------------------
    labels = ["x", "y", "z"]
    linestyles_cmd = ["-", "-", "-"]
    linestyles_act = ["--", "--", "--"]
    plt.figure(figsize=(10.5, 8.0))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, v_cmd[:, i], linestyles_cmd[i], label=f"v{i+1} cmd ({labels[i]})", linewidth=1.6)
        plt.plot(t, p_dot[:, i], linestyles_act[i], label=f"v{i+1} actual ({labels[i]})", linewidth=1.6, alpha=0.9)
        plt.grid(True)
        plt.ylabel(f"v{labels[i]} (mm/s)")
        if i == 0:
            plt.title("(c) Linear Velocity Components: Commanded vs Actual")
        if i == 2:
            plt.xlabel("Time (s)")
        plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    # -------------------------------------------------
    # Figure (d): angular velocity components wx,wy,wz
    # -------------------------------------------------
    plt.figure(figsize=(10.5, 6.5))
    w = p_dot[:, 3:6]  # (N,3)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(t, w[:, i], "-", linewidth=1.6, label=f"w{labels[i]} actual")
        plt.grid(True)
        plt.ylabel(f"w{labels[i]} (rad/s)")
        if i == 0:
            plt.title("(d) Angular Velocity Components (Actual)")
        if i == 2:
            plt.xlabel("Time (s)")
        plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # Run data collection
    collect_data()
    # Plot data
    plot_data()