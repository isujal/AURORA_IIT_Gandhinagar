#!/usr/bin/env python3
"""
combined_hw.py
==============
Runs the three combined.py trajectories on real OpenManipulator-X hardware.

Architecture mirrors robot_traffic_controller.py exactly:
  - OpenManipulatorX() for connection
  - ctrl.relative_joint_angle_to_raw() + ctrl.set_position() per waypoint
  - QuinticPolynomial generates smooth joint-space waypoints
  - stream_trajectory() sends waypoints at CTRL_HZ with real-time pacing

Sequence (X key):
    1. Translation X  — quintic, forward then reverse
    2. Translation Z  — quintic, forward then reverse
    3. Rotation J1    — approach, sweep, return

Controls:  X → sequence | H → home | Q → quit
"""

import time
import numpy as np
from src.robot_core.open_manipulator_x import OpenManipulatorX

# ============================================================
# GLOBAL STOP FLAG
# ============================================================
stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")


# ============================================================
# TRAJECTORY PARAMETERS  (all in radians, joint space)
# ============================================================

CTRL_HZ    = 50      # waypoints per second sent to motors
REST_SPEED = 15.0    # used only for move_to_rest_position()

# ── All joint angles solved by extract_ik_from_sim.py ──
# Cartesian targets from combined.py, IK solved via MuJoCo (sub-0.01mm error)
#
#   HOME    → EE [0.1417, 0.0210, 0.20] m
#   TX_END  → EE [0.2800, 0.0210, 0.20] m  (Translation X)
#   TZ_END  → EE [0.1417, 0.0210, 0.30] m  (Translation Z)
#   ROT_START/END → EE fixed at HOME, J4 swept -60°→+60°

HOME      = np.array([-2e-06, -0.79757,  0.660689,  0.136881])
TX_START  = HOME.copy()
TX_END    = np.array([-1e-06,  0.31651, -0.391795,  0.075285])
TX_TIME   = 2.0   # seconds one-way

TZ_START  = HOME.copy()
TZ_END    = np.array([-4e-06, -0.415029, -0.479124,  0.894153])
TZ_TIME   = 2.0

ROT_START    = np.array([ 0.0,     -0.576623,  0.964703, -1.047198])
ROT_END      = np.array([-2e-06,   -0.576632,  0.153861,  1.047198])
ROT_TIME     = 3.0
APPROACH_TIME = 1.5   # HOME → ROT_START  and  ROT_END → HOME


# ============================================================
# QUINTIC POLYNOMIAL  (identical to combined.py)
# ============================================================

class QuinticPolynomial:
    """Smooth 1-D p0→pf over T seconds. Zero vel & accel at endpoints."""
    def __init__(self, p0, pf, T):
        A = np.array([[  T**3,   T**4,    T**5],
                      [3*T**2, 4*T**3,  5*T**4],
                      [6*T,   12*T**2, 20*T**3]])
        a3, a4, a5 = np.linalg.solve(A, [pf - p0, 0.0, 0.0])
        self.c = [p0, 0.0, 0.0, a3, a4, a5]

    def position(self, t):
        return sum(c * t**i for i, c in enumerate(self.c))


# ============================================================
# BUILD JOINT-SPACE WAYPOINT LIST
# ============================================================

def make_joint_trajectory(q_start, q_end, motion_time, ctrl_hz=CTRL_HZ):
    """
    Pre-compute all waypoints using per-joint quintic polynomials.
    Returns list of np.array shape (4,), one per control step.
    """
    n_joints = len(q_start)
    polys    = [QuinticPolynomial(float(q_start[j]), float(q_end[j]), motion_time)
                for j in range(n_joints)]
    n_steps  = int(motion_time * ctrl_hz)
    return [np.array([polys[j].position(i / ctrl_hz) for j in range(n_joints)])
            for i in range(n_steps + 1)]


# ============================================================
# MOVE ONE WAYPOINT  (from robot_traffic_controller.py)
# ============================================================

def move_robot_pose(ctrl, joint_ids, pose):
    """Send one joint-angle array to all motors. No sleep inside."""
    for i, jid in enumerate(joint_ids):
        raw = ctrl.relative_joint_angle_to_raw(jid, float(pose[i]))
        ctrl.set_position(jid, raw)


# ============================================================
# STREAM TRAJECTORY  (hardware equivalent of sim's step loop)
# ============================================================

def stream_trajectory(ctrl, joint_ids, waypoints, ctrl_hz=CTRL_HZ, label=""):
    """
    Send every waypoint to the motors at ctrl_hz with real-time pacing.

    Equivalent to combined.py's:
        set_ctrl(data, q_cmd)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

    Uses perf_counter to compensate for any drift — no fixed sleep.
    """
    global stop_requested
    if stop_requested:
        return

    print(f"  → {label}  ({len(waypoints)} pts @ {ctrl_hz} Hz) …")
    t_start = time.perf_counter()

    for i, wp in enumerate(waypoints):
        if stop_requested:
            print("    [stopped]")
            return

        move_robot_pose(ctrl, joint_ids, wp)

        # Real-time pacing: sleep only the remaining time to next step
        t_target  = t_start + (i + 1) / ctrl_hz
        remaining = t_target - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    print(f"  ✓ {label} done")


# ============================================================
# READ CURRENT JOINT ANGLES SAFELY
# ============================================================

def read_current_pose(robot, ctrl, joint_ids):
    """Try robot.get_joint_angles(), then raw per-motor, then HOME."""
    # Attempt 1: clean robot-level API
    try:
        angles = robot.get_joint_angles()
        if angles is not None and len(angles) >= len(joint_ids):
            return np.array(angles[:len(joint_ids)], dtype=float)
    except Exception:
        pass

    # Attempt 2: per-motor raw read
    try:
        pose = []
        for jid in joint_ids:
            raw   = ctrl.get_position(jid)
            angle = ctrl.raw_to_relative_joint_angle(jid, raw)
            pose.append(float(angle))
        return np.array(pose)
    except Exception as e:
        print(f"  [warn] pose read failed ({e}), assuming HOME")
        return HOME.copy()


# ============================================================
# SEGMENT 1: TRANSLATION X
# ============================================================

def run_translation_x(ctrl, joint_ids):
    print("\n  [1/3] Translation X  →←")
    wps = make_joint_trajectory(TX_START, TX_END, TX_TIME)
    stream_trajectory(ctrl, joint_ids, wps, label="TX forward")
    stream_trajectory(ctrl, joint_ids, list(reversed(wps)), label="TX reverse")


# ============================================================
# SEGMENT 2: TRANSLATION Z
# ============================================================

def run_translation_z(ctrl, joint_ids):
    print("\n  [2/3] Translation Z  ↑↓")
    wps = make_joint_trajectory(TZ_START, TZ_END, TZ_TIME)
    stream_trajectory(ctrl, joint_ids, wps, label="TZ forward")
    stream_trajectory(ctrl, joint_ids, list(reversed(wps)), label="TZ reverse")


# ============================================================
# SEGMENT 3: ROTATION
# ============================================================

def run_rotation(ctrl, joint_ids, q_current):
    print("\n  [3/3] Rotation  ←→")

    # Approach: wherever we are → ROT_START
    wps_app = make_joint_trajectory(q_current, ROT_START, APPROACH_TIME)
    stream_trajectory(ctrl, joint_ids, wps_app, label="Rot approach")

    # Sweep: ROT_START → ROT_END
    wps_rot = make_joint_trajectory(ROT_START, ROT_END, ROT_TIME)
    stream_trajectory(ctrl, joint_ids, wps_rot, label="Rot sweep")

    # Return: ROT_END → HOME
    wps_ret = make_joint_trajectory(ROT_END, HOME, APPROACH_TIME)
    stream_trajectory(ctrl, joint_ids, wps_ret, label="Rot return")


# ============================================================
# MOVE TO HOME
# ============================================================

def move_to_home(ctrl, joint_ids, q_current, duration=1.5):
    print("  → Homing …")
    wps = make_joint_trajectory(q_current, HOME, duration)
    stream_trajectory(ctrl, joint_ids, wps, label="Home")


# ============================================================
# FULL SEQUENCE
# ============================================================

def run_sequence(robot, ctrl, joint_ids):
    global stop_requested
    print("\n" + "="*55)
    print("  STARTING FULL SEQUENCE")
    print("="*55)

    q_now = read_current_pose(robot, ctrl, joint_ids)
    move_to_home(ctrl, joint_ids, q_now, duration=1.5)
    if stop_requested: return

    run_translation_x(ctrl, joint_ids)
    if stop_requested: return

    run_translation_z(ctrl, joint_ids)
    if stop_requested: return

    # After TZ returns to HOME, q_current is HOME
    run_rotation(ctrl, joint_ids, HOME.copy())
    if stop_requested: return

    print("\n" + "="*55)
    print("  SEQUENCE COMPLETE — press X to repeat")
    print("="*55 + "\n")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_challenge(params):
    global stop_requested
    stop_requested = False
    robot = None

    try:
        try:
            import keyboard
        except ImportError:
            print("pip install keyboard")
            return {"success": False}

        print("Initializing OpenManipulator-X …")
        robot = OpenManipulatorX()

        if not robot.is_connected:
            print("Robot not connected.")
            return {"success": False}

        robot.ping_motors()

        ctrl      = robot.controller
        joint_ids = robot.joint_ids

        # Enable torque + position control mode (same as traffic controller)
        for jid in joint_ids:
            ctrl.enable_torque(jid)
        for jid in joint_ids:
            ctrl.change_operating_mode(jid, 3)

        # Move to rest at startup
        robot.move_to_rest_position(wait_time=4.0, speed=REST_SPEED)

        # Smooth move from rest → HOME
        q_now = read_current_pose(robot, ctrl, joint_ids)
        move_to_home(ctrl, joint_ids, q_now, duration=2.0)

        print("\n=== Combined Hardware Trajectories ===")
        print("  X  →  Full sequence  (TX + TZ + Rotation)")
        print("  H  →  Home")
        print("  Q  →  Quit\n")

        current_command = None

        # Main loop — same structure as robot_traffic_controller.py
        while not stop_requested:

            if keyboard.is_pressed('x') and current_command != 'x':
                print("[X] Full sequence")
                current_command = 'x'
                run_sequence(robot, ctrl, joint_ids)
                time.sleep(0.3)

            elif keyboard.is_pressed('h') and current_command != 'h':
                print("[H] Home")
                current_command = 'h'
                q_now = read_current_pose(robot, ctrl, joint_ids)
                move_to_home(ctrl, joint_ids, q_now, duration=1.5)
                time.sleep(0.3)

            elif keyboard.is_pressed('q'):
                print("Quitting …")
                break

            if not any(keyboard.is_pressed(k) for k in ('x', 'h', 'q')):
                current_command = None

            time.sleep(0.02)

        print("Control loop exited.")
        return {"success": True}

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    finally:
        print("Cleaning up …")
        if robot is not None:
            try:
                robot.move_to_rest_position(wait_time=8.0, speed=REST_SPEED)
            except Exception:
                pass
            try:
                for jid in robot.joint_ids:
                    robot.controller.disable_torque(jid)
            except Exception:
                pass
        print("Done.")