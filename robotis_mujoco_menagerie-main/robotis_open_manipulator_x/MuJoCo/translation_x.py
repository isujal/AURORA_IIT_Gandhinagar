#!/usr/bin/env python3
"""
straight_line_x_flat_gripper.py
================================
OMX straight-line trajectory from START to END point.
Gripper (Joint5/slide) locked at 0 (open).
Flat gripper constraint: Joint4 = -(Joint2 + Joint3)

START : [0.2417, 0.0210, 0.10]  m
END   : [0.1017, 0.2100, 0.10]  m

IK solves Joint1, Joint2, Joint3 freely.
Joint4 is computed from flat-gripper constraint.
Joint5 (gripper slide) is locked at 0.

Controls
--------
    X  →  Execute straight-line move START → END
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
# TRAJECTORY PARAMETERS  ← edit these
# ============================================================
START_POS = np.array([0.1417, 0.0210, 0.20])   # [x, y, z] metres
END_POS   = np.array([0.28, 0.0210, 0.20])   # [x, y, z] metres

MOTION_TIME = 4.0   # seconds

# Joint5 (Gripper slide) locked value
# 0.0 = fully open, 0.019 = fully closed
J5_LOCKED = 0.0

# ============================================================
# PHYSICS RATES
# ============================================================
DT             = 0.002
CTRL_HZ        = 50
STEPS_PER_CTRL = int(round(1.0 / (CTRL_HZ * DT)))   # 10
HOME_STEPS     = 100


# ============================================================
# FLAT GRIPPER CONSTRAINT  (Joint4 locked to Joint2 + Joint3)
# Joint4 = -(Joint2 + Joint3) → gripper stays parallel to ground
# ============================================================

def apply_flat_gripper_constraint(q1, q2, q3):
    """
    IK gives [q1, q2, q3].
    Joint4 is computed to keep gripper flat.
    Joint5 is locked at J5_LOCKED.

    Returns full ctrl array [q1, q2, q3, q4, j5_locked].
    """
    q4 = -(q2 + q3)
    q4 = np.clip(q4, ARM_MIN[3], ARM_MAX[3])
    return np.array([q1, q2, q3, q4, J5_LOCKED])


# ============================================================
# MUJOCO HELPERS
# ============================================================

def get_ee_pos(model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")
    return data.xpos[body_id].copy()


def get_current_q(data):
    return data.qpos[:N_ARM].copy()


def set_ctrl(data, ctrl5):
    """Write all 5 ctrl values (arm joints + gripper)."""
    data.ctrl[:N_ARM] = np.clip(ctrl5[:N_ARM], ARM_MIN, ARM_MAX)
    data.ctrl[4]      = J5_LOCKED


def step_physics(model, data, viewer):
    for _ in range(STEPS_PER_CTRL):
        mujoco.mj_step(model, data)
    viewer.sync()


# ============================================================
# 3-DOF NUMERICAL IK  (Joint1, Joint2, Joint3)
# Joint4 from flat-gripper constraint, Joint5 locked
# ============================================================

def inverse_kinematics_3dof(model, data, target_pos, q3_init,
                              tolerance=1e-4, max_iter=300, lam=0.05):
    """
    Solves Joint1, Joint2, Joint3.
    At each iteration, Joint4 is set via flat-gripper constraint
    so the Jacobian reflects true constrained geometry.

    q3_init : [q1, q2, q3]
    Returns  : [q1, q2, q3]
    """
    q3      = np.array(q3_init, dtype=float)
    eps     = 1e-4
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")

    qpos_backup = data.qpos.copy()
    ctrl_backup = data.ctrl.copy()

    def set_q3(qq3):
        ctrl5 = apply_flat_gripper_constraint(qq3[0], qq3[1], qq3[2])
        data.qpos[:N_ARM] = ctrl5[:N_ARM]
        data.qpos[4]      = J5_LOCKED
        mujoco.mj_forward(model, data)

    for _ in range(max_iter):
        set_q3(q3)
        pos0 = data.xpos[body_id].copy()
        err  = target_pos - pos0
        if np.linalg.norm(err) < tolerance:
            break

        # Jacobian: 3 × 3
        J = np.zeros((3, 3))
        for j in range(3):
            qp = q3.copy(); qp[j] += eps
            set_q3(qp)
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
# QUINTIC POLYNOMIAL  (per axis)
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
        alpha  = i / HOME_STEPS
        q_interp = (1 - alpha) * q_now
        ctrl5  = apply_flat_gripper_constraint(q_interp[0], q_interp[1], q_interp[2])
        set_ctrl(data, ctrl5)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)
    print("  ✓ Home")


def execute_straight_line(model, data, viewer):
    # One quintic per axis
    traj_x = QuinticPolynomial(START_POS[0], END_POS[0], MOTION_TIME)
    traj_y = QuinticPolynomial(START_POS[1], END_POS[1], MOTION_TIME)
    traj_z = QuinticPolynomial(START_POS[2], END_POS[2], MOTION_TIME)
    n_steps = int(MOTION_TIME * CTRL_HZ)

    dist = np.linalg.norm(END_POS - START_POS)
    print(f"\n  Straight line  (flat gripper | J5 locked)")
    print(f"  START : [{START_POS[0]:.4f}, {START_POS[1]:.4f}, {START_POS[2]:.4f}] m")
    print(f"  END   : [{END_POS[0]:.4f}, {END_POS[1]:.4f}, {END_POS[2]:.4f}] m")
    print(f"  Dist  : {dist*1000:.1f} mm  over {MOTION_TIME:.1f} s")
    print(f"  Constraint: Joint4 = -(Joint2 + Joint3)  |  Joint5 = {J5_LOCKED}\n")

    # Warm-start IK from current Joint1, 2, 3
    q_now = get_current_q(data)
    q3    = q_now[:3].copy()

    for i in range(n_steps + 1):
        t      = i / CTRL_HZ
        target = np.array([traj_x.position(t),
                           traj_y.position(t),
                           traj_z.position(t)])

        # 3-DOF IK
        q3    = inverse_kinematics_3dof(model, data, target, q3)

        # Full ctrl with constraint
        ctrl5 = apply_flat_gripper_constraint(q3[0], q3[1], q3[2])
        set_ctrl(data, ctrl5)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

        if i % int(0.5 * CTRL_HZ) == 0:
            ee      = get_ee_pos(model, data)
            q_act   = get_current_q(data)
            flatness = np.rad2deg(q_act[1] + q_act[2] + q_act[3])
            print(f"    t={t:4.1f}s  "
                  f"target=[{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]  "
                  f"EE=[{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]  "
                  f"flat={flatness:+.2f}°  "
                  f"q_deg={np.rad2deg(q_act).round(1)}")

    ee_final = get_ee_pos(model, data)
    err = np.linalg.norm(END_POS - ee_final)
    print(f"\n  ✓ Done  EE error = {err*1000:.2f} mm")


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

    data.qpos[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    viewer.sync()
    time.sleep(0.5)

    ee_home = get_ee_pos(model, data)
    print(f"\n  EE at home: X={ee_home[0]*1000:.2f}mm  "
          f"Y={ee_home[1]*1000:.2f}mm  Z={ee_home[2]*1000:.2f}mm")
    print(f"\n=== Controls ===")
    print(f"  X → line from {START_POS} → {END_POS}")
    print(f"  H → home")
    print(f"  Q → quit\n")

    current_cmd = None

    try:
        while viewer.is_running():
            if keyboard.is_pressed('x') and current_cmd != 'x':
                current_cmd = 'x'
                print("[X] Running trajectory …")
                execute_straight_line(model, data, viewer)
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
        print("Homing before exit …")
        if viewer.is_running():
            move_to_home(model, data, viewer)
        print("Done.")


if __name__ == "__main__":
    main()