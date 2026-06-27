#!/usr/bin/env python3
"""
rotate_ee_y.py
==============
OMX — EE XYZ position FIXED. Joint4 (wrist) rotates about Y.

Strategy:
  - Pre-solve IK for start pose → smoothly move there first
  - Joints 1,2,3 hold EE position via 3-DOF IK every step
  - Joint4 sweeps independently via quintic polynomial
  - t=0.0 will show correct EE position with near-zero drift

Fixed EE position : [0.1417, 0.0210, 0.2045] m

Controls
--------
    X  →  Rotate Joint4 from J4_START to J4_END
    H  →  Home (all joints to 0)
    Q  →  Quit
"""

import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# MODEL PATH
# ============================================================
MODEL_PATH = Path(
    r"C:\Users\thein\OneDrive\Documents\AURORA"
    r"\robotis_mujoco_menagerie-main"
    r"\robotis_open_manipulator_x"
    r"\scene.xml"
)


# ============================================================
# JOINT LAYOUT
# ============================================================
N_ARM  = 4
N_CTRL = 5
ARM_MIN = np.array([-3.14159, -1.5, -1.5, -1.7 ])
ARM_MAX = np.array([ 3.14159,  1.5,  1.4,  1.97])

# ============================================================
# FIXED EE POSITION  ← edit this
# ============================================================
FIXED_POS = np.array([0.1417, 0.0210, 0.20])   # metres

# ============================================================
# JOINT4 SWEEP PARAMETERS  ← edit these 
# ============================================================
J4_START    = -np.pi / 3    # -45°
J4_END      =  np.pi / 3    # +45°
MOTION_TIME =  6.0           # seconds

# ============================================================
# PHYSICS RATES
# ============================================================
DT             = 0.002
CTRL_HZ        = 50
STEPS_PER_CTRL = int(round(1.0 / (CTRL_HZ * DT)))   # 10
HOME_STEPS     = 100
APPROACH_STEPS = 80    # steps to move from current pose → start pose


# ============================================================
# MUJOCO HELPERS
# ============================================================

def get_ee_pos(model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")
    return data.xpos[body_id].copy()


def get_current_q(data):
    return data.qpos[:N_ARM].copy()


def set_ctrl(data, q_arm):
    data.ctrl[:N_ARM] = np.clip(q_arm, ARM_MIN, ARM_MAX)
    data.ctrl[4]      = 0.0   # gripper open


def step_physics(model, data, viewer):
    for _ in range(STEPS_PER_CTRL):
        mujoco.mj_step(model, data)
    viewer.sync()


# ============================================================
# 3-DOF POSITION-ONLY IK  (Joint1, Joint2, Joint3)
# Joint4 is passed in separately and held fixed during solve
# so the Jacobian reflects true constrained geometry
# ============================================================

def inverse_kinematics_3dof(model, data, target_pos, q3_init, j4_fixed,
                              tolerance=1e-4, max_iter=300, lam=0.05):
    """
    Solves [q1, q2, q3] for a given target_pos.
    j4_fixed is held constant during the solve.
    Returns [q1, q2, q3].
    """
    q3      = np.array(q3_init, dtype=float)
    eps     = 1e-4
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")

    qpos_backup = data.qpos.copy()
    ctrl_backup = data.ctrl.copy()

    def fwd(qq3):
        data.qpos[:N_ARM] = [qq3[0], qq3[1], qq3[2], j4_fixed]
        mujoco.mj_forward(model, data)

    for _ in range(max_iter):
        fwd(q3)
        pos0 = data.xpos[body_id].copy()
        err  = target_pos - pos0
        if np.linalg.norm(err) < tolerance:
            break

        J = np.zeros((3, 3))
        for j in range(3):
            qp = q3.copy(); qp[j] += eps
            fwd(qp)
            J[:, j] = (data.xpos[body_id] - pos0) / eps

        JJT = J @ J.T
        dq  = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), err)
        q3[0] = np.clip(q3[0] + dq[0], ARM_MIN[0], ARM_MAX[0])
        q3[1] = np.clip(q3[1] + dq[1], ARM_MIN[1], ARM_MAX[1])
        q3[2] = np.clip(q3[2] + dq[2], ARM_MIN[2], ARM_MAX[2])

    data.qpos[:] = qpos_backup
    data.ctrl[:] = ctrl_backup
    mujoco.mj_forward(model, data)

    return q3


# ============================================================
# QUINTIC POLYNOMIAL
# ============================================================

class QuinticPolynomial:
    def __init__(self, p0, pf, T):
        A = np.array([[T**3,   T**4,    T**5  ],
                      [3*T**2, 4*T**3,  5*T**4],
                      [6*T,   12*T**2, 20*T**3]])
        a3, a4, a5 = np.linalg.solve(A, [pf - p0, 0., 0.])
        self.coeffs = [p0, 0., 0., a3, a4, a5]

    def position(self, t):
        return sum(c * t**i for i, c in enumerate(self.coeffs))


# ============================================================
# MOTION
# ============================================================

def move_to_home(model, data, viewer):
    q_now = get_current_q(data)
    print("  → Homing …")
    for i in range(HOME_STEPS + 1):
        alpha = i / HOME_STEPS
        set_ctrl(data, (1 - alpha) * q_now)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)
    print("  ✓ Home")


def execute_rotation(model, data, viewer):
    traj_j4 = QuinticPolynomial(J4_START, J4_END, MOTION_TIME)
    n_steps  = int(MOTION_TIME * CTRL_HZ)

    # ----------------------------------------------------------
    # STEP 1: Pre-solve start pose fully before any motion
    # Run IK many times from current q until fully converged
    # ----------------------------------------------------------
    print(f"\n  Pre-solving start pose (FIXED_POS={FIXED_POS}, J4={np.rad2deg(J4_START):.1f}°) …")
    q_now = get_current_q(data)
    q3    = q_now[:3].copy()

    for _ in range(20):   # 20 passes guarantees convergence
        q3 = inverse_kinematics_3dof(model, data, FIXED_POS, q3, J4_START)

    q_start_full = np.array([q3[0], q3[1], q3[2], J4_START])

    # Verify pre-solve result
    data.qpos[:N_ARM] = q_start_full
    mujoco.mj_forward(model, data)
    ee_check = get_ee_pos(model, data)
    print(f"  Pre-solve EE : [{ee_check[0]:.4f}, {ee_check[1]:.4f}, {ee_check[2]:.4f}]  "
          f"drift={np.linalg.norm(ee_check - FIXED_POS)*1000:.2f} mm")

    # Restore
    data.qpos[:N_ARM] = q_now
    mujoco.mj_forward(model, data)

    # ----------------------------------------------------------
    # STEP 2: Smoothly move to start pose
    # ----------------------------------------------------------
    print(f"  Moving to start pose …")
    q_from = get_current_q(data)
    for i in range(APPROACH_STEPS + 1):
        alpha = i / APPROACH_STEPS
        q_cmd = (1 - alpha) * q_from + alpha * q_start_full
        set_ctrl(data, q_cmd)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

    ee_at_start = get_ee_pos(model, data)
    print(f"  At start pose. EE=[{ee_at_start[0]:.4f}, {ee_at_start[1]:.4f}, "
          f"{ee_at_start[2]:.4f}]  "
          f"drift={np.linalg.norm(ee_at_start - FIXED_POS)*1000:.2f} mm")

    # ----------------------------------------------------------
    # STEP 3: Rotation loop
    # ----------------------------------------------------------
    print(f"\n  Joint4 rotation  (EE position FIXED)")
    print(f"  Fixed EE : {FIXED_POS} m")
    print(f"  Joint4   : {np.rad2deg(J4_START):.1f}° → {np.rad2deg(J4_END):.1f}°"
          f"  over {MOTION_TIME:.1f} s\n")

    # Warm-start from actual qpos after approach
    q_now = get_current_q(data)
    q3    = q_now[:3].copy()

    for i in range(n_steps + 1):
        t  = i / CTRL_HZ
        j4 = traj_j4.position(t)

        # Solve J1,J2,J3 to hold FIXED_POS with current j4
        q3 = inverse_kinematics_3dof(model, data, FIXED_POS, q3, j4)

        q_cmd = np.array([q3[0], q3[1], q3[2], j4])
        set_ctrl(data, q_cmd)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

        if i % int(0.5 * CTRL_HZ) == 0:
            ee        = get_ee_pos(model, data)
            pos_drift = np.linalg.norm(ee - FIXED_POS)
            print(f"    t={t:4.1f}s  "
                  f"J4={np.rad2deg(j4):+7.1f}°  "
                  f"EE=[{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]  "
                  f"drift={pos_drift*1000:.2f} mm  "
                  f"q_deg={np.rad2deg(q_cmd).round(1)}")

    ee_final  = get_ee_pos(model, data)
    print(f"\n  ✓ Done  |  final drift = {np.linalg.norm(ee_final - FIXED_POS)*1000:.2f} mm")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Not found: {MODEL_PATH}")

    model  = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data   = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    try:
        import keyboard
    except ImportError:
        print("pip install keyboard"); return

    # Start at home
    data.qpos[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(0.5)

    ee_home = get_ee_pos(model, data)
    print(f"\n  EE at home : [{ee_home[0]:.4f}, {ee_home[1]:.4f}, {ee_home[2]:.4f}] m")
    print(f"  FIXED_POS  : {FIXED_POS} m")
    print(f"\n=== Controls ===")
    print(f"  X → rotate J4 {np.rad2deg(J4_START):.0f}° → {np.rad2deg(J4_END):.0f}°  (EE fixed)")
    print(f"  H → home")
    print(f"  Q → quit\n")

    current_cmd = None

    try:
        while viewer.is_running():

            if keyboard.is_pressed('x') and current_cmd != 'x':
                current_cmd = 'x'
                print("[X] Executing rotation …")
                execute_rotation(model, data, viewer)
                time.sleep(0.3)

            elif keyboard.is_pressed('h') and current_cmd != 'h':
                current_cmd = 'h'
                print("[H] Homing …")
                move_to_home(model, data, viewer)
                time.sleep(0.3)

            elif keyboard.is_pressed('q'):
                print("Quit."); break

            if not any(keyboard.is_pressed(k) for k in ('x', 'h', 'q')):
                current_cmd = None

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Homing …")
        if viewer.is_running():
            move_to_home(model, data, viewer)
        print("Done.")


if __name__ == "__main__":
    main()