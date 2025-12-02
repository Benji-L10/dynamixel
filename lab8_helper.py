# lab8_helper.py
# Shared helpers for Lab 8 (Option 2: PBVS)

import time
import numpy as np
import cv2

# ---------- Core config (shared) ----------
BALL_RADIUS_MM = 15.0
DT = 0.05  # 20 Hz

# PID (start with your Lab 7 tuned values)
Kp = np.diag([1.0, 1.0, 1.0])
Ki = np.diag([0.10, 0.10, 0.10])
Kd = np.diag([0.10, 0.10, 0.10])

# Joint safety
VEL_CLAMP_DEG_S = 60.0
VEL_DEADBAND_DEG_S = 0.5

# Approach / grasp / lift
APPROACH_OFFSET_MM = np.array([0.0, 0.0, 80.0])
GRASP_Z_MM = 39.0
LIFT_Z_MM  = 100.0

# ---------- Vision: detection & pose ----------

def detect_balls(image_bgr):
    """
    Detect colored balls via HSV segmentation + contour circularity.
    Returns: list[(color_str, (cx, cy), r_px)] or None.
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

            # quick HSV sanity (avoid dull/gray)
            mask_circle = np.zeros_like(mask)
            cv2.circle(mask_circle, (cx, cy), int(r * 0.8), 255, -1)
            s_mean = np.mean(hsv[:, :, 1][mask_circle == 255])
            v_mean = np.mean(hsv[:, :, 2][mask_circle == 255])
            if s_mean < 50 or v_mean < 50:
                continue

            results.append((cname, (cx, cy), r))

            # overlay (optional)
            cv2.circle(image_bgr, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(image_bgr, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(image_bgr, cname, (cx - 20, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Detection", image_bgr)
    cv2.waitKey(1)
    return results if results else None


def get_ball_pose(corners_px, intrinsics, radius_mm):
    """
    PnP using 4 "equator" points of the image circle.
    corners_px: 4x2 in order [left, right, bottom, top]
    Returns: (R, tvec) with tvec (3,1) in mm in camera frame
    """
    obj = np.array([[-radius_mm,   0.0, 0.0],
                    [ radius_mm,   0.0, 0.0],
                    [ 0.0,       -radius_mm, 0.0],
                    [ 0.0,        radius_mm, 0.0]], dtype=np.float32)
    img = np.asarray(corners_px, dtype=np.float32).reshape(-1, 2)

    K = np.array([[intrinsics.fx, 0.0,            intrinsics.ppx],
                  [0.0,           intrinsics.fy,  intrinsics.ppy],
                  [0.0,           0.0,            1.0]], dtype=np.float32)
    try:
        dist = np.array(intrinsics.coeffs, dtype=np.float32).reshape(-1, 1)
    except Exception:
        dist = np.zeros((5, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed for sphere")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


# ---------- PBVS controller ----------

class PBVSController:
    """
    PID in task space: v = Kp e + Ki ∫e dt + Kd de/dt
    qdot allocation: qdot = J_v^# v
    """
    def __init__(self, dt=DT, Kp_=Kp, Ki_=Ki, Kd_=Kd,
                 vel_clamp_deg_s=VEL_CLAMP_DEG_S, deadband_deg_s=VEL_DEADBAND_DEG_S):
        self.dt = dt
        self.Kp = Kp_
        self.Ki = Ki_
        self.Kd = Kd_
        self.vel_clamp = vel_clamp_deg_s
        self.deadband = deadband_deg_s
        self.e_int = np.zeros(3)
        self.e_prev = np.zeros(3)

    def step(self, robot, p_des_mm):
        # current joints/pose
        readings = robot.get_joints_readings()
        q_deg = readings[0, :]
        p_cur = robot.get_ee_pos(q_deg)[:3]  # mm

        # PID
        e = (np.asarray(p_des_mm) - p_cur).astype(float)
        self.e_int += e * self.dt
        e_dot = (e - self.e_prev) / self.dt
        self.e_prev = e.copy()

        v = (self.Kp @ e) + (self.Ki @ self.e_int) + (self.Kd @ e_dot)  # mm/s

        # Jacobian + allocation
        Jv = robot.get_jacobian(q_deg)[:3, :]  # 3x4
        Jpinv = np.linalg.pinv(Jv, rcond=1e-3)
        qdot_rad = (Jpinv @ v.reshape(3, 1)).reshape(4)

        # safety shaping
        qdot_deg = np.degrees(qdot_rad)
        qdot_deg = np.clip(qdot_deg, -self.vel_clamp, self.vel_clamp)
        qdot_deg[np.abs(qdot_deg) < self.deadband] = 0.0
        return e, qdot_deg


# ---------- Simple task-space trajectory (for transport/home) ----------

def move_trajectory(robot, target_pos_xyzp, traj_time_s, num_pts=100):
    """
    Linear interpolation in task space, executed via IK (open-loop).
    target_pos_xyzp: [x,y,z,pitch]
    """
    readings = robot.get_joints_readings()
    q_now = readings[0, :]
    p_now = robot.get_ee_pos(q_now)[:4]

    target = np.asarray(target_pos_xyzp, dtype=float)
    p_now = np.asarray(p_now, dtype=float)

    waypoints = np.column_stack([np.linspace(p_now[i], target[i], num_pts) for i in range(4)])
    dt = traj_time_s / max(num_pts - 1, 1)
    robot.write_time(dt)

    t0 = time.time()
    for i in range(num_pts):
        q_cmd = robot.get_ik(waypoints[i].tolist())
        t_target = t0 + i * dt
        while True:
            now = time.time()
            if now >= t_target:
                break
            time.sleep(max(0.0, t_target - now))
        robot.write_joints(q_cmd)
