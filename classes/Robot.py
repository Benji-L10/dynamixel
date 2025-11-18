# (c) 2025 S. Farzan, Electrical Engineering Department, Cal Poly
# Skeleton Robot class for OpenManipulator-X Robot for EE 471

import numpy as np

from dataclasses import dataclass

from .DX_XM430_W350 import DX_XM430_W350
from .OM_X_arm import OM_X_arm
from jaxtyping import Float64, Array
from beartype import beartype

# Type alias for a 1x4 array: [x, y, z, a]
JointPose = Float64[Array, "1 4"]


"""
Robot class for controlling the OpenManipulator-X Robot.
Inherits from OM_X_arm and provides methods specific to the robot's operation.
"""


@beartype
def get_joint_pose() -> JointPose:
    """Return a 1x4 array representing pose [x, y, z, a]."""
    # Example: valid
    pose = np.array([[0.1, 0.2, 0.3, 1.57]], dtype=np.float64)
    return pose


@dataclass
class _IKPair:
    """Holds both elbow-up and elbow-down IK solutions."""

    elbow_up: np.ndarray
    elbow_down: np.ndarray


class Robot(OM_X_arm):
    """
    Initialize the Robot class.
    Creates constants and connects via serial. Sets default mode and state.
    """

    def __init__(self):
        super().__init__()

        self.GRIP_OPEN_DEG = -45.0
        self.GRIP_CLOSE_DEG = +45.0
        self.GRIP_THRESH_DEG = 180.0

        # Robot Dimensions (in mm)
        self.mDim = [77, 130, 124, 126]
        self.mOtherDim = [128, 24]

        # IK constants (link lengths and offsets)
        self.mDim[0] = self.mDim[0]  # 77 mm
        self.L1 = self.mDim[0]  # 77 mm
        self.L2 = self.mDim[1]  # 130 mm
        self.L3 = self.mDim[2]  # 124 mm
        self.L4 = self.mDim[3]  # 126 mm
        self.L21 = self.mOtherDim[0]  # 128 mm
        self.L22 = self.mOtherDim[1]  # 24 mm

        self.DHTable = np.array(
            [
                [0.0, 77.0, 0.0, -(np.pi / 2)],  # θ1, d1, a1, α1
                [-(np.pi / 2 - np.asin(24 / 130)), 0.0, 130.0, 0.0],  # θ2, d2, a2, α2
                [+(np.pi / 2 - np.asin(24 / 130)), 0.0, 124.0, 0.0],  # θ3, d3, a3, α3
                [0.0, 0.0, 126.0, 0.0],  # θ4, d4, a4, α4
            ],
            dtype=np.float64,
        )

        # Set default mode and state
        # Change robot to position mode with torque enabled by default
        # Feel free to change this as desired
        self.write_mode("position")
        self.write_motor_state(True)

        # Set the robot to move between positions with a 5 second trajectory profile
        # change here or call writeTime in scripts to change
        self.write_time(5)

    def _set_time_profile_bit_all(self, enable: bool):
        """Turn the Drive Mode 'time-based profile' bit (bit 2) on/off for all joints."""
        DX = DX_XM430_W350
        # Read current drive modes
        dm = self.bulk_read_write(DX.DRIVE_MODE_LEN, DX.DRIVE_MODE, None)  # list[int]
        if not isinstance(dm, list) or len(dm) != len(self.motorIDs):
            raise RuntimeError("Failed to read DRIVE_MODE for all joints.")
        new_dm = []
        for v in dm:
            if enable:
                new_dm.append(v | 0b100)  # set bit 2
            else:
                new_dm.append(v & ~0b100)  # clear bit 2
        # Write back (bulk)
        self.bulk_read_write(DX.DRIVE_MODE_LEN, DX.DRIVE_MODE, new_dm)

    """
    Sends the joints to the desired angles.
    Parameters:
    goals (list of 1x4 float): Angles (degrees) for each of the joints to go to.
    """

    def write_joints(self, q_deg):
        """Send joint target angles in degrees (list/array length N)."""
        DX = DX_XM430_W350
        q_deg = list(q_deg)
        if len(q_deg) != len(self.motorIDs):
            raise ValueError(
                f"Expected {len(self.motorIDs)} joint angles, got {len(q_deg)}"
            )

        ticks = [
            int(round(angle * DX.TICKS_PER_DEG + DX.TICK_POS_OFFSET)) for angle in q_deg
        ]

        # If you're in normal position mode (not extended), keep values in [0, 4095]
        ticks = [max(0, min(int(DX.TICKS_PER_ROT - 1), t)) for t in ticks]

        self.bulk_read_write(DX.POS_LEN, DX.GOAL_POSITION, ticks)

    """
    Creates a time-based profile (trapezoidal) based on the desired times.
    This will cause write_position to take the desired number of seconds to reach the setpoint.
    Parameters:
    time (float): Total profile time in seconds. If 0, the profile will be disabled (be extra careful).
    acc_time (float, optional): Total acceleration time for ramp up and ramp down (individually, not combined). Defaults to time/3.
    """

    def write_time(self, total_time_s, acc_time_s=None):
        """Configure trapezoidal TIME profile for all joints."""
        if acc_time_s is None:
            acc_time_s = float(total_time_s) / 3.0

        # Enable time-based profile (bit 2) for all joints
        self._set_time_profile_bit_all(True)

        acc_ms = int(round(acc_time_s * DX_XM430_W350.MS_PER_S))
        tot_ms = int(round(float(total_time_s) * DX_XM430_W350.MS_PER_S))

        # Bulk write to all joints
        self.bulk_read_write(
            DX_XM430_W350.PROF_ACC_LEN,
            DX_XM430_W350.PROF_ACC,
            [acc_ms] * len(self.motorIDs),
        )
        self.bulk_read_write(
            DX_XM430_W350.PROF_VEL_LEN,
            DX_XM430_W350.PROF_VEL,
            [tot_ms] * len(self.motorIDs),
        )

    """
    Sets the gripper to be open or closed.
    Parameters:
    open (bool): True to set the gripper to open, False to close.
    """

    def write_gripper(self, is_open: bool):
        """Open/close gripper using fixed angles in position mode."""
        target = self.GRIP_OPEN_DEG if is_open else self.GRIP_CLOSE_DEG
        self.gripper.write_position(target)

    def read_gripper(self) -> float:
        """Return gripper joint position in degrees."""
        return self.gripper.read_position()

    def read_gripper_open(self) -> bool:
        return self.read_gripper() > self.GRIP_THRESH_DEG

    """
    Sets position holding for the joints on or off.
    Parameters:
    enable (bool): True to enable torque to hold the last set position for all joints, False to disable.
    """

    def write_motor_state(self, enable):
        state = 1 if enable else 0
        states = [state] * self.motorsNum  # Repeat the state for each motor
        self.bulk_read_write(
            DX_XM430_W350.TORQUE_ENABLE_LEN, DX_XM430_W350.TORQUE_ENABLE, states
        )

    """
    Supplies the joints with the desired currents.
    Parameters:
    currents (list of 1x4 float): Currents (mA) for each of the joints to be supplied.
    """

    def write_currents(self, currents):
        current_in_ticks = [
            round(current * DX_XM430_W350.TICKS_PER_mA) for current in currents
        ]
        self.bulk_read_write(
            DX_XM430_W350.CURR_LEN, DX_XM430_W350.GOAL_CURRENT, current_in_ticks
        )

    """
    Change the operating mode for all joints.
    Parameters:
    mode (str): New operating mode for all joints. Options include:
        "current": Current Control Mode (writeCurrent)
        "velocity": Velocity Control Mode (writeVelocity)
        "position": Position Control Mode (writePosition)
        "ext position": Extended Position Control Mode
        "curr position": Current-based Position Control Mode
        "pwm voltage": PWM Control Mode
    """

    def write_mode(self, mode):
        if mode in ["current", "c"]:
            write_mode = DX_XM430_W350.CURR_CNTR_MD
        elif mode in ["velocity", "v"]:
            write_mode = DX_XM430_W350.VEL_CNTR_MD
        elif mode in ["position", "p"]:
            write_mode = DX_XM430_W350.POS_CNTR_MD
        elif mode in ["ext position", "ep"]:
            write_mode = DX_XM430_W350.EXT_POS_CNTR_MD
        elif mode in ["curr position", "cp"]:
            write_mode = DX_XM430_W350.CURR_POS_CNTR_MD
        elif mode in ["pwm voltage", "pwm"]:
            write_mode = DX_XM430_W350.PWM_CNTR_MD
        else:
            raise ValueError(
                f"writeMode input cannot be '{mode}'. See implementation in DX_XM430_W350 class."
            )

        self.write_motor_state(False)
        write_modes = [
            write_mode
        ] * self.motorsNum  # Create a list with the mode value for each motor
        self.bulk_read_write(
            DX_XM430_W350.OPR_MODE_LEN, DX_XM430_W350.OPR_MODE, write_modes
        )
        self.write_motor_state(True)

    """
    Gets the current joint positions, velocities, and currents.
    Returns:
    numpy.ndarray: A 3x4 array containing the joints' positions (deg), velocities (deg/s), and currents (mA).
    """

    def get_joints_readings(self):
        """
        Returns a 3xN array: [deg; deg/s; mA] for the N arm joints (excludes gripper).
        """
        len(self.motorIDs)

        # Bulk read raw registers
        pos_u32 = self.bulk_read_write(
            DX_XM430_W350.POS_LEN, DX_XM430_W350.CURR_POSITION, None
        )  # list of ints
        vel_u32 = self.bulk_read_write(
            DX_XM430_W350.VEL_LEN, DX_XM430_W350.CURR_VELOCITY, None
        )
        cur_u16 = self.bulk_read_write(
            DX_XM430_W350.CURR_LEN, DX_XM430_W350.CURR_CURRENT, None
        )

        # Vectorize
        pos_u32 = np.array(pos_u32, dtype=np.uint32)
        vel_u32 = np.array(vel_u32, dtype=np.uint32)  # signed 32-bit
        cur_u16 = np.array(cur_u16, dtype=np.uint16)  # signed 16-bit

        # Convert signed types
        vel_i32 = (vel_u32.astype(np.int64) + (1 << 31)) % (1 << 32) - (1 << 31)
        vel_i32 = vel_i32.astype(np.int32)
        cur_i16 = (cur_u16.astype(np.int32) + (1 << 15)) % (1 << 16) - (1 << 15)
        cur_i16 = cur_i16.astype(np.int16)

        # Units
        q_deg = (
            pos_u32.astype(np.int64) - int(DX_XM430_W350.TICK_POS_OFFSET)
        ) / DX_XM430_W350.TICKS_PER_DEG
        qd_dps = vel_i32 / DX_XM430_W350.TICKS_PER_ANGVEL
        I_mA = cur_i16 / DX_XM430_W350.TICKS_PER_mA

        readings = np.vstack(
            [q_deg.astype(float), qd_dps.astype(float), I_mA.astype(float)]
        )
        return readings

    """
    Sends the joints to the desired velocities.
    Parameters:
    vels (list of 1x4 float): Angular velocities (deg/s) for each of the joints to go at.
    """

    def write_velocities(self, vels):
        """Send joint target velocities in deg/s (list/array length N)."""
        vels = list(vels)
        if len(vels) != len(self.motorIDs):
            raise ValueError(
                f"Expected {len(self.motorIDs)} velocities, got {len(vels)}"
            )

        ticks_per_s = [
            int(round(v * DX_XM430_W350.TICKS_PER_ANGVEL)) for v in vels
        ]  # signed
        self.bulk_read_write(
            DX_XM430_W350.VEL_LEN, DX_XM430_W350.GOAL_VELOCITY, ticks_per_s
        )

    def get_dh_row_mat(self, row):
        """
        Compute the Standard DH homogeneous transform A_i from a single DH row.

        Parameters
        ----------
        row : array-like, shape (4,)
            [theta, d, a, alpha] for one joint.

        Returns
        -------
        A : ndarray, shape (4,4)
        """

        theta, d, a, alpha = np.asarray(row, dtype=np.float64)
        A = np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0.0, np.sin(alpha), np.cos(alpha), d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return A

    def get_int_mat(self, joint_angles):
        """
        Build all intermediate DH transforms A_i for the provided joint angles.

        Parameters
        ----------
            joint_angles : array-like, shape (4,)
            Joint variables q in degrees: [q1, q2, q3, q4].
        Returns
        -------
        A_stack : ndarray, shape (4,4,4)
        A_stack[:, :, i] = A_{i+1}

        Steps
        -----
        1) Copy the base DH table.
        2) Add q (deg) to the theta column (col 0).
        3) For each row, compute A_i via get_dh_row_mat(...).
        """
        q_deg = np.asarray(joint_angles, dtype=np.float64).reshape(4)
        dh = self.DHTable.copy()
        dh[:, 0] += np.deg2rad(q_deg)   # add radians to theta column
        A_stack = np.empty((4, 4, 4), dtype=np.float64)
        for i in range(4):
            A_stack[:, :, i] = self.get_dh_row_mat(dh[i])
        return A_stack

    def get_fk(self, joint_angles):
        """
        Forward kinematics to the end-effector.

        Parameters
        ----------
        joint_angles : array-like, shape (4,)
        Joint variables q in degrees.

        Returns
        -------
        T : ndarray, shape (4,4)
        Homogeneous transform T^0_4 (base to end-effector).
        """
        A_stack = self.get_int_mat(joint_angles)
        T = np.eye(4, dtype=np.float64)
        for i in range(4):
            T = T @ A_stack[:, :, i]
        return T

    def get_current_fk(self):
        """
        Forward kinematics to the end-effector using current joint readings.

        Returns
        -------
        T : ndarray, shape (4,4)
        Homogeneous transform T^0_4 (base to end-effector).
        """
        readings = self.get_joints_readings()
        q_deg = readings[0, :]  # (4,)
        return self.get_fk(q_deg)

    def get_ee_pos(self, joint_angles):
        """
        Get the end-effector position from joint angles.

        Parameters
        ----------
        joint_angles : array-like, shape (4,)
        Joint variables q in degrees.

        Returns
        -------
        p : ndarray, shape (5,)
        End-effector position in mm.
        """
        T = self.get_fk(joint_angles)
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        q1, q2, q3, q4 = np.asarray(joint_angles, dtype=np.float64).reshape(4)
        yaw = q1
        pitch = -(q2 + q3 + q4)
        return np.array([x, y, z, pitch, yaw], dtype=np.float64)

    def get_ik(self, pose):
        """
        Calculate inverse kinematics for OpenManipulator-X.
        Input: pose = [x, y, z, alpha]  (mm, mm, mm, deg)
        Output: np.array([q1, q2, q3, q4]) in degrees (elbow-up)
        """
        x, y, z, alpha_deg = pose
        alpha = np.deg2rad(alpha_deg)

        # Compute geometric parameters
        r = np.sqrt(x**2 + y**2)
        r_w = r - (self.L4 * np.cos(alpha))
        z_w = z - self.L1 - (self.L4 * np.sin(alpha))
        d_w = np.sqrt(r_w**2 + z_w**2)

        # Check reachability
        if d_w > (self.L2 + self.L3) or d_w < abs(self.L2 - self.L3):
            raise ValueError("Pose unreachable: outside workspace")

        mu = np.arctan2(z_w, r_w)
        delta = np.arctan2(self.L22, self.L21)

        cos_beta = (self.L2**2 + self.L3**2 - d_w**2) / (2 * self.L2 * self.L3)
        cos_gamma = (d_w**2 + self.L2**2 - self.L3**2) / (2 * d_w * self.L2)

        sin_beta = np.sqrt(1.0 - cos_beta**2)
        sin_gamma = np.sqrt(1.0 - cos_gamma**2)

        beta = np.arctan2(sin_beta, cos_beta)
        gamma = np.arctan2(sin_gamma, cos_gamma)

        # Elbow up solutions
        q1 = np.degrees(np.arctan2(y, x))
        q2 = np.degrees(np.pi / 2 - delta - gamma - mu)
        q3 = np.degrees(np.pi / 2 + delta - beta)
        q4 = -alpha_deg - q2 - q3

        # Elbow down solutions
        beta_down = np.arctan2(-sin_beta, cos_beta)
        gamma_down = np.arctan2(-sin_gamma, cos_gamma)
        q1_down = np.degrees(np.arctan2(y, x))
        q2_down = np.degrees(np.pi / 2 - delta - gamma_down - mu)
        q3_down = np.degrees(np.pi / 2 + delta - beta_down)
        q4_down = -alpha_deg - q2 - q3

        # Return solutions in deg
        q_up = np.array([q1, q2, q3, q4])
        q_down = np.array([q1_down, q2_down, q3_down, q4_down])

        return q_up

    def get_jacobian(self, q_deg):
        """
        Compute the 6x4 geometric Jacobian J(q) for the OpenManipulator-X.
        Inputs:
        q_deg : array-like, shape (4,) joint angles in DEGREES [q1..q4]
        Returns:
        J : np.ndarray, shape (6,4)
            [ Jv ]  (mm)
            [ Jw ]  (dimensionless; angular part produces rad/s)
        Conventions (per Prelab 5):
        - Internally convert to radians for trig
        - q̇ in rad/s, linear vel in mm/s, angular vel in rad/s
        """
        q_deg = np.asarray(q_deg, dtype=float).reshape(4,)
        q = np.deg2rad(q_deg)  # radians for trig

        # 1) Build intermediate A_i using Lab 2 or DH fallback
        A_list = None
        if hasattr(self, "get_int_mat"):
            # Expect: self.get_int_mat(q_rad) -> list/array of 4 homogeneous Ai (4x4)
            A_list = self.get_int_mat(q_deg)
        else:
            # Fallback: build Ai from self.dh_table with standard DH: [theta(rad), d(mm), a(mm), alpha(rad)]
            if not hasattr(self, "dh_table"):
                raise RuntimeError("No get_int_mat() or dh_table available to build transforms.")
            th = (self.dh_table[:, 0] + q).astype(float)  # add joint motion to theta
            d  = self.dh_table[:, 1].astype(float)
            a  = self.dh_table[:, 2].astype(float)
            al = self.dh_table[:, 3].astype(float)

            A_list = []
            for i in range(4):
                ct, st = np.cos(th[i]), np.sin(th[i])
                ca, sa = np.cos(al[i]), np.sin(al[i])
                Ai = np.array([
                    [ ct, -st*ca,  st*sa, a[i]*ct],
                    [ st,  ct*ca, -ct*sa, a[i]*st],
                    [  0,     sa,     ca,     d[i]],
                    [  0,      0,      0,       1],
                ], dtype=float)
                A_list.append(Ai)

        # Ensure we have 4 links
        if isinstance(A_list, (list, tuple)):
            assert len(A_list) == 4, "Expected 4 intermediate transforms (A1..A4)."
        else:
            # Could be array with shape (4,4,4)
            assert len(A_list) == 4, "Expected 4 intermediate transforms (A1..A4)."

        # 2) Accumulate T^0_i correctly
        T_list = [np.eye(4)]
        if isinstance(A_list, np.ndarray):
            # A_list is (4,4,4) with A(:,:,i) = A_{i+1}
            T = np.eye(4)
            for i in range(4):
                Ai = A_list[:, :, i]
                T = T @ Ai
                T_list.append(T)
        else:
            # A_list is a list of four 4x4 matrices
            T = np.eye(4)
            for Ai in A_list:
                T = T @ Ai
                T_list.append(T)

        # 3) Extract origins o_i and z-axes z_i in base frame
        # o_i: first 3 of column 3, z_i: first 3 of rotation column 2 (indexing 0-based)
        # Using standard convention: z is the 3rd column of R (index 2)
        o = [T_i[0:3, 3] for T_i in T_list]              # o0..o4
        z = [T_list[i][0:3, 2] for i in range(0, 4)]     # z0..z3
        # Base frame z0, o0 already included via T_list[0] = I

        o4 = o[4]
        Jv_cols = []
        Jw_cols = []
        for i in range(4):
            zi = z[i]
            oi = o[i]
            Jv_cols.append(np.cross(zi, (o4 - oi)))  # mm
            Jw_cols.append(zi)                       # unitless

        Jv = np.column_stack(Jv_cols)  # (3,4)
        Jw = np.column_stack(Jw_cols)  # (3,4)
        J  = np.vstack((Jv, Jw))       # (6,4)

        return J
    
    def get_fwd_vel_kin(self, q_rad, qd_rad, current_ma=None):
        """
        Forward velocity kinematics:  ṗ = J(q) q̇

        Parameters
        ----------
        q_rad : array-like, shape (4,)
            Joint angles in radians [q1..q4].
        qd_rad : array-like, shape (4,)
            Joint velocities in rad/s.
        current_ma : array-like, shape (4,), optional
            Motor currents in mA (not used here; included for interface compatibility).

        Returns
        -------
        p_dot : ndarray, shape (6,)
            Spatial end-effector velocity [vx, vy, vz, wx, wy, wz],
            where linear components are in mm/s and angular components are in rad/s.
        """
        q = np.asarray(q_rad, dtype=float).reshape(4,)
        qd = np.asarray(qd_rad, dtype=float).reshape(4,)

        # get_jacobian expects joint angles in DEGREES and internally uses mm for linear parts.
        J = self.get_jacobian(np.rad2deg(q))   # (6x4)

        # pp_dot = J(q) q_dot  -> linear in mm/s, angular in rad/s
        p_dot = J @ qd                         # (6,)

        return p_dot