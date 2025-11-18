#!/usr/bin/env python3
"""
Lab 2 Analysis: Waypoint-based Motion Planning
Comprehensive analysis of joint-space and task-space trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import load_from_pickle
from classes.Robot import Robot

def load_and_analyze_data():
    """Load robot data and perform comprehensive analysis"""

    # Load data
    data = load_from_pickle("robot_data.pkl")

    timestamps_s = np.array(data["timestamps_s"], dtype=np.float64)
    waypoints_s = np.array(data["waypoints_s"], dtype=np.float64)
    joint_deg = np.array(data["joint_deg"], dtype=np.float64)
    xyz_mm = np.array(data["xyz_mm"], dtype=np.float64)
    waypoints_deg = np.array(data["waypoints_deg"], dtype=np.float64)
    traj_time_s = data["traj_time_s"]
    poll_dt = data["poll_dt"]

    # Calculate waypoint positions in task space using forward kinematics
    # We'll use the Robot class but handle connection errors gracefully
    try:
        robot = Robot()
        waypoints_xyz = []
        for wp in waypoints_deg:
            T = robot.get_fk(wp)
            waypoints_xyz.append([T[0, 3], T[1, 3], T[2, 3]])
        waypoints_xyz = np.array(waypoints_xyz)
        robot.close()
    except:
        # If robot connection fails, estimate waypoint positions from trajectory data
        print("Warning: Robot connection failed. Estimating waypoint positions from trajectory data.")
        waypoints_xyz = []

        # Find waypoint positions by looking at trajectory data at waypoint times
        for i, wp_time in enumerate(waypoints_s):
            if i == 0:
                # First waypoint is at start of trajectory
                waypoints_xyz.append(xyz_mm[0])
            else:
                # Find closest data point to waypoint time
                time_diff = np.abs(timestamps_s - wp_time)
                closest_idx = np.argmin(time_diff)
                waypoints_xyz.append(xyz_mm[closest_idx])

        waypoints_xyz = np.array(waypoints_xyz)

    return {
        'timestamps_s': timestamps_s,
        'waypoints_s': waypoints_s,
        'joint_deg': joint_deg,
        'xyz_mm': xyz_mm,
        'waypoints_deg': waypoints_deg,
        'waypoints_xyz': waypoints_xyz,
        'traj_time_s': traj_time_s,
        'poll_dt': poll_dt
    }

def create_joint_angles_plot(data):
    """Create plot (a): Joint angles vs time"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    titles = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, ax in enumerate(axes.ravel()):
        ax.plot(data['timestamps_s'], data['joint_deg'][:, idx],
                color=colors[idx], linewidth=2, label=titles[idx])
        ax.set_title(f"{titles[idx]} Trajectory", fontsize=12, fontweight='bold')
        ax.set_ylabel("Angle (deg)", fontsize=11)
        ax.grid(True, alpha=0.7)
        ax.legend()

        # Mark waypoint transitions
        for i, t in enumerate(data['waypoints_s']):
            if i > 0:  # Skip first waypoint (start)
                ax.axvline(x=t, color='gray', linestyle='--', alpha=0.6)
                ax.annotate(f'WP{i}', (t, ax.get_ylim()[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)

    axes[1, 0].set_xlabel("Time (s)", fontsize=11)
    axes[1, 1].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle("Joint Angles vs Time - Waypoint Motion Sequence",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("lab2_joint_angles_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def create_xz_time_plot(data):
    """Create plot (b): x and z vs time"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data['timestamps_s'], data['xyz_mm'][:, 0],
            'o-', linewidth=2, markersize=4, label="x (mm)", color='tab:blue')
    ax.plot(data['timestamps_s'], data['xyz_mm'][:, 2],
            's-', linewidth=2, markersize=4, label="z (mm)", color='tab:red')

    # Mark waypoint transitions
    for i, t in enumerate(data['waypoints_s']):
        if i > 0:  # Skip first waypoint (start)
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.6)
            ax.annotate(f'WP{i}', (t, max(data['xyz_mm'][:, 0].max(),
                                     data['xyz_mm'][:, 2].max())),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Position (mm)", fontsize=12)
    ax.set_title("End-Effector x and z Positions vs Time", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.7)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("lab2_xz_vs_time_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def create_xz_trajectory_plot(data):
    """Create plot (c): x-z trajectory with waypoints"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory
    ax.plot(data['xyz_mm'][:, 0], data['xyz_mm'][:, 2],
            'b-', linewidth=2, label="Actual Trajectory", alpha=0.8)

    # Plot waypoints
    colors = ['green', 'orange', 'orange', 'red']
    labels = ['Start', 'WP2', 'WP3', 'End']
    sizes = [120, 100, 100, 120]

    for i, wp in enumerate(data['waypoints_xyz']):
        x, z = wp[0], wp[2]  # Extract x and z coordinates
        ax.scatter(x, z, c=colors[i], s=sizes[i],
                  label=labels[i], zorder=5, edgecolors='black', linewidth=1)
        ax.annotate(f'WP{i+1}', (x, z), xytext=(8, 8),
                   textcoords='offset points', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Draw expected triangle edges
    for i in range(len(data['waypoints_xyz'])-1):
        ax.plot([data['waypoints_xyz'][i, 0], data['waypoints_xyz'][i+1, 0]],
                [data['waypoints_xyz'][i, 2], data['waypoints_xyz'][i+1, 2]],
                'k--', alpha=0.5, linewidth=1, label='Expected Path' if i == 0 else "")

    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("z (mm)", fontsize=12)
    ax.set_title("End-Effector Trajectory in x-z Plane", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.7)
    ax.legend(fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig("lab2_xz_trajectory_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def create_xy_trajectory_plot(data):
    """Create plot (d): x-y trajectory with waypoints"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory
    ax.plot(data['xyz_mm'][:, 0], data['xyz_mm'][:, 1],
            'b-', linewidth=2, label="Actual Trajectory", alpha=0.8)

    # Plot waypoints
    colors = ['green', 'orange', 'orange', 'red']
    labels = ['Start', 'WP2', 'WP3', 'End']
    sizes = [120, 100, 100, 120]

    for i, wp in enumerate(data['waypoints_xyz']):
        x, y = wp[0], wp[1]  # Extract x and y coordinates
        ax.scatter(x, y, c=colors[i], s=sizes[i],
                  label=labels[i], zorder=5, edgecolors='black', linewidth=1)
        ax.annotate(f'WP{i+1}', (x, y), xytext=(8, 8),
                   textcoords='offset points', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel("x (mm)", fontsize=12)
    ax.set_ylabel("y (mm)", fontsize=12)
    ax.set_title("End-Effector Trajectory in x-y Plane", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.7)
    ax.legend(fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig("lab2_xy_trajectory_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def analyze_joint_trajectories(data):
    """Analyze joint-space trajectory characteristics"""
    print("\n" + "="*60)
    print("JOINT-SPACE TRAJECTORY ANALYSIS")
    print("="*60)

    joint_names = ["Joint 1", "Joint 2", "Joint 3", "Joint 4"]

    for i, name in enumerate(joint_names):
        joint_data = data['joint_deg'][:, i]
        timestamps = data['timestamps_s']

        # Calculate velocity and acceleration
        dt = np.diff(timestamps)
        velocity = np.diff(joint_data) / dt
        acceleration = np.diff(velocity) / dt[1:]

        print(f"\n{name}:")
        print(f"  Range: {joint_data.min():.1f}° to {joint_data.max():.1f}°")
        print(f"  Total displacement: {joint_data[-1] - joint_data[0]:.1f}°")
        print(f"  Max velocity: {np.abs(velocity).max():.1f}°/s")
        print(f"  Max acceleration: {np.abs(acceleration).max():.1f}°/s²")

        # Analyze motion profile
        if len(velocity) > 0:
            accel_phases = np.sum(acceleration > 0.1)
            decel_phases = np.sum(acceleration < -0.1)
            const_phases = len(acceleration) - accel_phases - decel_phases

            print(f"  Motion profile: {accel_phases} accel, {const_phases} const, {decel_phases} decel samples")

def analyze_task_space_trajectory(data):
    """Analyze task-space trajectory characteristics"""
    print("\n" + "="*60)
    print("TASK-SPACE TRAJECTORY ANALYSIS")
    print("="*60)

    xyz = data['xyz_mm']
    waypoints_xyz = data['waypoints_xyz']

    # Calculate triangle side lengths
    sides = []
    for i in range(len(waypoints_xyz)-1):
        side_length = np.linalg.norm(waypoints_xyz[i+1] - waypoints_xyz[i])
        sides.append(side_length)
        print(f"Side {i+1} (WP{i+1} to WP{i+2}): {side_length:.1f} mm")

    # Calculate actual path lengths
    path_lengths = []
    for i in range(len(waypoints_xyz)-1):
        # Find data points in this segment
        start_time = data['waypoints_s'][i] if i > 0 else 0
        end_time = data['waypoints_s'][i+1] if i+1 < len(data['waypoints_s']) else data['timestamps_s'][-1]

        mask = (data['timestamps_s'] >= start_time) & (data['timestamps_s'] <= end_time)
        segment_xyz = xyz[mask]

        if len(segment_xyz) > 1:
            path_length = np.sum(np.linalg.norm(np.diff(segment_xyz, axis=0), axis=1))
            path_lengths.append(path_length)
            print(f"Actual path length segment {i+1}: {path_length:.1f} mm")

    # Analyze x-y motion
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    print(f"\nY-axis motion range: {y_range:.1f} mm")
    print(f"Y-axis motion is {'significant' if y_range > 5 else 'minimal'}")

    # Analyze timing
    segment_times = []
    for i in range(len(data['waypoints_s'])-1):
        segment_time = data['waypoints_s'][i+1] - data['waypoints_s'][i]
        segment_times.append(segment_time)
        print(f"Segment {i+1} duration: {segment_time:.2f} s")

    print(f"\nTiming analysis:")
    print(f"  Commanded time per segment: {data['traj_time_s']:.1f} s")
    print(f"  Average actual segment time: {np.mean(segment_times):.2f} s")
    print(f"  Time variation: {np.std(segment_times):.3f} s")

def print_implementation_details(data):
    """Print implementation details and analysis"""
    print("\n" + "="*60)
    print("IMPLEMENTATION DETAILS")
    print("="*60)

    print(f"Trajectory time between waypoints: {data['traj_time_s']:.1f} seconds")
    print(f"Rationale: Chosen to provide smooth motion while allowing adequate")
    print(f"           time for the robot to reach each waypoint without overshoot")

    print(f"\nData collection rate: {1/data['poll_dt']:.0f} samples per second")
    print(f"Sampling period: {data['poll_dt']*1000:.0f} ms")
    print(f"Total data points collected: {len(data['timestamps_s'])}")

    print(f"\nWaypoints (joint angles in degrees):")
    for i, wp in enumerate(data['waypoints_deg']):
        print(f"  WP{i+1}: {wp}")

    print(f"\nWaypoints (task space in mm):")
    for i, wp in enumerate(data['waypoints_xyz']):
        print(f"  WP{i+1}: x={wp[0]:.1f}, y={wp[1]:.1f}, z={wp[2]:.1f}")

    print(f"\nChallenges encountered:")
    print(f"  - Limited sampling rate (20 Hz) may miss high-frequency dynamics")
    print(f"  - Robot motion may not perfectly follow commanded trajectories")
    print(f"  - Waypoint timing synchronization required careful implementation")

def main():
    """Main analysis function"""
    print("Loading and analyzing robot motion data...")

    # Load data
    data = load_and_analyze_data()

    # Print implementation details
    print_implementation_details(data)

    # Create all plots
    print("\nGenerating plots...")
    create_joint_angles_plot(data)
    create_xz_time_plot(data)
    create_xz_trajectory_plot(data)
    create_xy_trajectory_plot(data)

    # Perform analysis
    analyze_joint_trajectories(data)
    analyze_task_space_trajectory(data)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("All plots saved with '_analysis' suffix for clarity.")

if __name__ == "__main__":
    main()
