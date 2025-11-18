# Ben Ly, Electrical Engineering Department, Cal Poly
# classes/TrajPlanner.py
# Cubic (joint-space) and Quintic (task-space) multi-segment planner

import numpy as np

class TrajPlanner:
    """
    Trajectory Planner class for calculating trajectories for different polynomial orders and relevant coefficients.
    """

    def __init__(self, setpoints):
        """
        Initialize the TrajPlanner class.

        Parameters:
        setpoints (numpy array): Array of waypoints with shape (m, 4) where m >= 2
        """
        if not isinstance(setpoints, np.ndarray):
            raise TypeError("setpoints must be a numpy array")

        if setpoints.ndim != 2 or setpoints.shape[1] != 4:
            raise ValueError("setpoints must have shape (m, 4)")

        if setpoints.shape[0] < 2:
            raise ValueError("At least 2 waypoints required")

        self.setpoints = setpoints

    # CUBIC 

    def calc_cubic_coeff(self, t0, tf, p0, pf, v0, vf):
        """
        Given the initial time, final time, initial position, final position, initial velocity, and final velocity,
        returns cubic polynomial coefficients.

        Parameters:
        t0 (float): Start time of setPoint.
        tf (float): End time of setPoint.
        p0 (float): Position of current setPoint.
        pf (float): Position of next setPoint.
        v0 (float): Starting velocity of setPoint.
        vf (float): Velocity at next setPoint.

        Returns:
        numpy array: The calculated polynomial coefficients [a0, a1, a2, a3] for
                     p(t) = a0 + a1*t + a2*t^2 + a3*t^3
        """
        coeff_matrix = np.array([
            [1, t0, t0 ** 2, t0 ** 3],
            [0, 1, 2 * t0, 3 * t0 ** 2],
            [1, tf, tf ** 2, tf ** 3],
            [0, 1, 2 * tf, 3 * tf ** 2]
        ], dtype=float)
        qs = np.array([p0, v0, pf, vf], dtype=float)
        coeff = np.linalg.solve(coeff_matrix, qs)
        return coeff

    def calc_cubic_traj(self, traj_time, points_num, coeff):
        """
        Given the time between setpoints, number of points between waypoints, and polynomial coefficients,
        returns the cubic trajectory of waypoints for a single pair of setpoints.

        Parameters:
        traj_time (int): Time between setPoints.
        points_num (int): Number of waypoints between setpoints.
        coeff (numpy array): Polynomial coefficients for trajectory.

        Returns:
        numpy array: The calculated waypoints of shape (points_num,)
        """
        waypoints = np.zeros(points_num, dtype=float)
        times = np.linspace(0, traj_time, points_num + 2)[1:-1]  # interior only

        for k, t in enumerate(times):
            waypoints[k] = coeff[0] + coeff[1] * t + coeff[2] * t**2 + coeff[3] * t**3

        return waypoints

    def get_cubic_traj(self, traj_time: float, points_num: int):
        """
        Generate a complete cubic trajectory through all waypoints.

        Parameters:
        -----------
        traj_time : float
            Duration in seconds for each segment between consecutive waypoints
        points_num : int
            Number of intermediate samples per segment (not including endpoints)

        Returns:
        --------
        numpy array
            Array of shape (N, 5) where:
            - Column 0: time values (sec)
            - Columns 1-4: values for each dimension at each time point
            - N = (num_segments * (points_num + 1)) + 1

        Notes:
        ------
        - Initial and final velocities are set to zero
        - Intermediate velocities should be computed for smoothness
        """
        setpoints = self.setpoints
        num_segments = len(setpoints) - 1
        waypoints_list = np.zeros((num_segments * (points_num + 1) + 1, 5), dtype=float)

        for i in range(4):  # For each dimension
            count = 0
            for j in range(num_segments):  # For each segment
                v0 = 0.0
                vf = 0.0

                coeff = self.calc_cubic_coeff(0, traj_time, setpoints[j, i], setpoints[j + 1, i], v0, vf)

                # Store starting waypoint (only on first dimension to avoid redundancy)
                if i == 0:
                    waypoints_list[count, 1:] = setpoints[j, :]

                count += 1
                waypoints_list[count:count + points_num, i + 1] = self.calc_cubic_traj(traj_time, points_num, coeff)
                count += points_num

            # Store final waypoint (only on first dimension)
            if i == 0:
                waypoints_list[count, 1:] = setpoints[-1, :]

        # Generate time vector
        time = np.linspace(0, traj_time * num_segments, waypoints_list.shape[0])
        waypoints_list[:, 0] = time

        return waypoints_list

    # QUINTIC

    def calc_quintic_coeff(self, t0, tf, p0, pf, v0, vf, a0, af):
        """
        Given initial/final time, position, velocity, and acceleration,
        returns quintic polynomial coefficients.

        Parameters:
        t0, tf (float): Start/end times of the segment
        p0, pf (float): Positions at t0 and tf
        v0, vf (float): Velocities at t0 and tf
        a0, af (float): Accelerations at t0 and tf

        Returns:
        numpy array: The polynomial coefficients [b0..b5] for
                     p(t) = b0 + b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5
        """
        # 6 boundary conditions → 6x6 system
        coeff_matrix = np.array([
            [1, t0,      t0**2,       t0**3,        t0**4,         t0**5],
            [0, 1,       2*t0,        3*t0**2,      4*t0**3,       5*t0**4],
            [0, 0,       2,           6*t0,         12*t0**2,      20*t0**3],
            [1, tf,      tf**2,       tf**3,        tf**4,         tf**5],
            [0, 1,       2*tf,        3*tf**2,      4*tf**3,       5*tf**4],
            [0, 0,       2,           6*tf,         12*tf**2,      20*tf**3],
        ], dtype=float)
        qs = np.array([p0, v0, a0, pf, vf, af], dtype=float)
        coeff = np.linalg.solve(coeff_matrix, qs)
        return coeff

    def calc_quintic_traj(self, traj_time, points_num, coeff):
        """
        Given segment duration, number of interior samples, and quintic coefficients,
        returns the interior points for a single segment.

        Parameters:
        traj_time (float): Segment duration (seconds)
        points_num (int): Number of interior points (no endpoints)
        coeff (numpy array): [b0..b5]

        Returns:
        numpy array: The calculated waypoints of shape (points_num,)
        """
        b0, b1, b2, b3, b4, b5 = coeff
        waypoints = np.zeros(points_num, dtype=float)
        times = np.linspace(0, traj_time, points_num + 2)[1:-1]  # interior only

        for k, t in enumerate(times):
            waypoints[k] = (
                b0 + b1*t + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5
            )

        return waypoints

    def get_quintic_traj(self, traj_time: float, points_num: int):
        """
        Generate a complete quintic trajectory through all waypoints.

        Parameters:
        -----------
        traj_time : float
            Duration in seconds for each segment between consecutive waypoints
        points_num : int
            Number of intermediate samples per segment (not including endpoints)

        Returns:
        --------
        numpy array
            Array of shape (N, 5) where:
            - Column 0: time values (sec)
            - Columns 1-4: values for each dimension at each time point
            - N = (num_segments * (points_num + 1)) + 1

        Notes:
        ------
        - This version sets v=a=0 at all segment boundaries (rest-to-rest with smooth accel).
        - Mirrors the cubic implementation style (matrix solve + interior samples).
        """
        setpoints = self.setpoints
        num_segments = len(setpoints) - 1
        waypoints_list = np.zeros((num_segments * (points_num + 1) + 1, 5), dtype=float)

        for i in range(4):  # For each dimension
            count = 0
            for j in range(num_segments):  # For each segment
                v0 = 0.0; vf = 0.0
                a0 = 0.0; af = 0.0

                coeff = self.calc_quintic_coeff(
                    0.0, traj_time,
                    setpoints[j, i], setpoints[j + 1, i],
                    v0, vf, a0, af
                )

                # Store starting waypoint (only on first dimension to avoid redundancy)
                if i == 0:
                    waypoints_list[count, 1:] = setpoints[j, :]

                count += 1
                waypoints_list[count:count + points_num, i + 1] = self.calc_quintic_traj(traj_time, points_num, coeff)
                count += points_num

            # Store final waypoint (only on first dimension)
            if i == 0:
                waypoints_list[count, 1:] = setpoints[-1, :]

        # Generate time vector
        time = np.linspace(0, traj_time * num_segments, waypoints_list.shape[0])
        waypoints_list[:, 0] = time

        return waypoints_list
