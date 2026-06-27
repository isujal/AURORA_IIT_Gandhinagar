#!/usr/bin/env python3
"""
straight_line_z_flat_gripper.py
================================
OMX straight-line trajectory along Z axis with gripper always
kept FLAT (parallel to the ground).

GIMBAL LOCK CONSTRAINT
----------------------
Joint2, Joint3, Joint4 all rotate about the Y axis (from the XML).
To keep the gripper flat (pointing forward, horizontal) at all times:

    Joint4 = -(Joint2 + Joint3)

This cancels whatever the shoulder+elbow do, locking the gripper
orientation to the world frame — intentional gimbal lock.

The IK only solves for Joint1, Joint2, Joint3 (3 unknowns for x,y,z).
Joint4 is then computed analytically from the constraint.
Joint5 (Gripper slide) stays at 0 (open).

Controls
--------
    X  →  Execute Z sweep (gripper stays flat throughout)
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
N_ARM  = 4      # Joint1..4
N_CTRL = 5      # Joint1..4 + Gripper

# Joint limits from open_manipulator_x.xml
ARM_MIN = np.array([-3.14159, -1.5, -1.5, -1.7 ])
ARM_MAX = np.array([ 3.14159,  1.5,  1.4,  1.97])

# ============================================================
# TRAJECTORY PARAMETERS  ← edit these
# ============================================================
LINE_X  = 0.1417    # constant X (m)   workspace: 0.03 – 0.30
LINE_Y  = 0.0210    # constant Y (m)   workspace: -0.20 – +0.20

Z_START = 0.20      # sweep start (m)  workspace: -0.01 – 0.33
Z_END   = 0.30      # sweep end   (m)

MOTION_TIME = 4.0   # seconds

# ============================================================
# PHYSICS RATES
# ============================================================
DT             = 0.002
CTRL_HZ        = 50
STEPS_PER_CTRL = int(round(1.0 / (CTRL_HZ * DT)))   # 10
HOME_STEPS     = 100


# ============================================================
# GIMBAL LOCK CONSTRAINT
# Joint4 always cancels Joint2 + Joint3 → gripper stays flat
# ============================================================

def apply_flat_gripper_constraint(q3):
    """
    Given q = [q1, q2, q3] from the 3-DOF IK,
    compute q4 so gripper stays parallel to the ground.

    Constraint:  q4 = -(q2 + q3)
    Returns full 4-DOF array [q1, q2, q3, q4].
    """
    q1, q2, q3_val = q3
    q4 = -(q2 + q3_val)
    q4 = np.clip(q4, ARM_MIN[3], ARM_MAX[3])
    return np.array([q1, q2, q3_val, q4])


# ============================================================
# MUJOCO HELPERS
# ============================================================

def get_ee_pos(model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")
    return data.xpos[body_id].copy()


def get_current_q(data):
    return data.qpos[:N_ARM].copy()


def set_ctrl(data, q_full4):
    """Write all 4 arm joints to ctrl. Gripper stays at 0 (open)."""
    data.ctrl[:N_ARM] = np.clip(q_full4, ARM_MIN, ARM_MAX)
    data.ctrl[4]      = 0.0   # gripper open


def step_physics(model, data, viewer):
    for _ in range(STEPS_PER_CTRL):
        mujoco.mj_step(model, data)
    viewer.sync()


# ============================================================
# 3-DOF NUMERICAL IK  (solves Joint1, Joint2, Joint3 only)
# Joint4 is handled by the flat-gripper constraint after IK.
# ============================================================

def inverse_kinematics_3dof(model, data, target_pos, q3_init,
                              tolerance=1e-4, max_iter=300, lam=0.05):
    """
    IK for Joint1, Joint2, Joint3 only.
    At each iteration, Joint4 is set via the flat-gripper constraint
    so the Jacobian reflects the true constrained geometry.

    q3_init : array of shape (3,) — [q1, q2, q3] initial guess
    Returns  : array of shape (3,) — [q1, q2, q3] solution
    """
    q3      = np.array(q3_init, dtype=float)
    eps     = 1e-4
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")

    qpos_backup = data.qpos.copy()
    ctrl_backup = data.ctrl.copy()

    def set_q3(qq3):
        """Write [q1,q2,q3] + constrained q4 into qpos and mj_forward."""
        q_full = apply_flat_gripper_constraint(qq3)
        data.qpos[:N_ARM] = q_full
        mujoco.mj_forward(model, data)

    for _ in range(max_iter):
        set_q3(q3)
        pos0 = data.xpos[body_id].copy()
        err  = target_pos - pos0
        if np.linalg.norm(err) < tolerance:
            break

        # Jacobian: perturb each of the 3 free joints
        J = np.zeros((3, 3))
        for j in range(3):
            qp    = q3.copy(); qp[j] += eps
            set_q3(qp)
            J[:, j] = (data.xpos[body_id] - pos0) / eps

        JJT = J @ J.T
        dq  = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), err)

        # Clip only the 3 free joints
        q3_new = q3 + dq
        q3_new[0] = np.clip(q3_new[0], ARM_MIN[0], ARM_MAX[0])
        q3_new[1] = np.clip(q3_new[1], ARM_MIN[1], ARM_MAX[1])
        q3_new[2] = np.clip(q3_new[2], ARM_MIN[2], ARM_MAX[2])
        q3 = q3_new

    # Restore sim state
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
        alpha  = i / HOME_STEPS
        q_interp = (1 - alpha) * q_now   # toward zeros
        set_ctrl(data, q_interp)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)
    print("  ✓ Home")


def execute_straight_line(model, data, viewer):
    traj_z  = QuinticPolynomial(Z_START, Z_END, MOTION_TIME)
    n_steps = int(MOTION_TIME * CTRL_HZ)

    print(f"\n  Straight line along Z  (flat gripper constraint active)")
    print(f"  X = {LINE_X:.4f} m  (constant)")
    print(f"  Y = {LINE_Y:.4f} m  (constant)")
    print(f"  Z : {Z_START:.4f} → {Z_END:.4f} m  over {MOTION_TIME:.1f} s")
    print(f"  Constraint: Joint4 = -(Joint2 + Joint3)\n")

    # Warm-start: use current Joint1,2,3 as initial guess
    q_now  = get_current_q(data)
    q3     = q_now[:3].copy()   # [q1, q2, q3] only

    for i in range(n_steps + 1):
        t      = i / CTRL_HZ
        target = np.array([LINE_X, LINE_Y, traj_z.position(t)])

        # Solve 3-DOF IK (Joint1, 2, 3)
        q3 = inverse_kinematics_3dof(model, data, target, q3)

        # Apply flat-gripper constraint to get Joint4
        q_full = apply_flat_gripper_constraint(q3)

        # Send to actuators
        set_ctrl(data, q_full)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

        # Log every 0.5 s
        if i % int(0.5 * CTRL_HZ) == 0:
            ee = get_ee_pos(model, data)
            q_actual = get_current_q(data)
            wrist_check = q_actual[1] + q_actual[2] + q_actual[3]
            print(f"    t={t:4.1f}s  "
                  f"Z_target={target[2]:.4f}  "
                  f"EE=[{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]  "
                  f"q4_cmd={np.rad2deg(q_full[3]):+.1f}°  "
                  f"J2+J3+J4={np.rad2deg(wrist_check):+.2f}° (→0 = flat)")

    ee_final = get_ee_pos(model, data)
    err = abs(ee_final[2] - Z_END)
    print(f"\n  ✓ Done  Z error = {err*1000:.2f} mm")


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
    print(f"  X → Z sweep  ({Z_START:.3f} → {Z_END:.3f} m)  [flat gripper]")
    print(f"  H → home")
    print(f"  Q → quit\n")

    current_cmd = None

    try:
        while viewer.is_running():
            if keyboard.is_pressed('x') and current_cmd != 'x':
                current_cmd = 'x'
                print("[X] Running Z trajectory with flat gripper …")
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