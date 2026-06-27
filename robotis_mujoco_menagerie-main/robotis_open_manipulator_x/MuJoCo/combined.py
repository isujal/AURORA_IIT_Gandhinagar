#!/usr/bin/env python3
"""
combined_sequence.py
====================
OMX — Single X keypress runs one full sequence:

    1. Translation X   (home → end, then REVERSE back to home)
    2. Translation Z   (home → end, then REVERSE back to home)
    3. Rotation        (J4: J4_START → J4_END, EE fixed, then home)

Home EE position : [0.1417, 0.0210, 0.20] m

REVERSE PATH: translation X and Z return by replaying their
recorded joint commands in reverse — exact same path, backwards.

Controls
--------
    X  →  Run full sequence
    H  →  Home only
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
# HOME EE POSITION
# ============================================================
HOME_EE = np.array([0.1417, 0.0210, 0.20])

# ============================================================
# TRANSLATION X PARAMETERS
# ============================================================
X_START_POS   = np.array([0.1417, 0.0210, 0.20])
X_END_POS     = np.array([0.28,   0.0210, 0.20])
X_MOTION_TIME = 2.0

# ============================================================
# TRANSLATION Z PARAMETERS
# ============================================================
LINE_X        = 0.1417
LINE_Y        = 0.0210
Z_START       = 0.20
Z_END         = 0.30
Z_MOTION_TIME = 2.0

# ============================================================
# ROTATION PARAMETERS
# ============================================================
FIXED_POS       = np.array([0.1417, 0.0210, 0.20])
J4_START        = -np.pi / 3
J4_END          =  np.pi / 3
ROT_MOTION_TIME = 3.0
APPROACH_STEPS  = 80

# ============================================================
# PHYSICS RATES
# ============================================================
DT             = 0.002
CTRL_HZ        = 50
STEPS_PER_CTRL = int(round(1.0 / (CTRL_HZ * DT)))
HOME_STEPS     = 100
J5_LOCKED      = 0.0


# ============================================================
# FLAT GRIPPER CONSTRAINT  J4 = -(J2 + J3)
# ============================================================

def apply_flat(q1, q2, q3):
    q4 = np.clip(-(q2 + q3), ARM_MIN[3], ARM_MAX[3])
    return np.array([q1, q2, q3, q4])


# ============================================================
# MUJOCO HELPERS
# ============================================================

def get_ee_pos(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")
    return data.xpos[bid].copy()

def get_current_q(data):
    return data.qpos[:N_ARM].copy()

def set_ctrl(data, q4):
    data.ctrl[:N_ARM] = np.clip(q4, ARM_MIN, ARM_MAX)
    data.ctrl[4]      = J5_LOCKED

def step_physics(model, data, viewer):
    for _ in range(STEPS_PER_CTRL):
        mujoco.mj_step(model, data)
    viewer.sync()


# ============================================================
# IK — flat gripper (translation scripts)
# ============================================================

def ik_flat(model, data, target_pos, q3_init,
            tolerance=1e-4, max_iter=300, lam=0.05):
    q3  = np.array(q3_init, dtype=float)
    eps = 1e-4
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")
    qb  = data.qpos.copy(); cb = data.ctrl.copy()

    def fwd(qq3):
        data.qpos[:N_ARM] = apply_flat(qq3[0], qq3[1], qq3[2])
        mujoco.mj_forward(model, data)

    for _ in range(max_iter):
        fwd(q3)
        pos0 = data.xpos[bid].copy()
        err  = target_pos - pos0
        if np.linalg.norm(err) < tolerance:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            qp = q3.copy(); qp[j] += eps
            fwd(qp)
            J[:, j] = (data.xpos[bid] - pos0) / eps
        JJT = J @ J.T
        dq  = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), err)
        q3[0] = np.clip(q3[0] + dq[0], ARM_MIN[0], ARM_MAX[0])
        q3[1] = np.clip(q3[1] + dq[1], ARM_MIN[1], ARM_MAX[1])
        q3[2] = np.clip(q3[2] + dq[2], ARM_MIN[2], ARM_MAX[2])

    data.qpos[:] = qb; data.ctrl[:] = cb
    mujoco.mj_forward(model, data)
    return q3


# ============================================================
# IK — free J4 (rotation)
# ============================================================

def ik_free(model, data, target_pos, q3_init, j4_fixed,
            tolerance=1e-4, max_iter=300, lam=0.05):
    q3  = np.array(q3_init, dtype=float)
    eps = 1e-4
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")
    qb  = data.qpos.copy(); cb = data.ctrl.copy()

    def fwd(qq3):
        data.qpos[:N_ARM] = [qq3[0], qq3[1], qq3[2], j4_fixed]
        mujoco.mj_forward(model, data)

    for _ in range(max_iter):
        fwd(q3)
        pos0 = data.xpos[bid].copy()
        err  = target_pos - pos0
        if np.linalg.norm(err) < tolerance:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            qp = q3.copy(); qp[j] += eps
            fwd(qp)
            J[:, j] = (data.xpos[bid] - pos0) / eps
        JJT = J @ J.T
        dq  = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), err)
        q3[0] = np.clip(q3[0] + dq[0], ARM_MIN[0], ARM_MAX[0])
        q3[1] = np.clip(q3[1] + dq[1], ARM_MIN[1], ARM_MAX[1])
        q3[2] = np.clip(q3[2] + dq[2], ARM_MIN[2], ARM_MAX[2])

    data.qpos[:] = qb; data.ctrl[:] = cb
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
        self.c = [p0, 0., 0., a3, a4, a5]
    def position(self, t):
        return sum(c * t**i for i, c in enumerate(self.c))


# ============================================================
# REPLAY RECORDED COMMANDS IN REVERSE
# ============================================================

def replay_reverse(model, data, viewer, recorded_cmds, label=""):
    """
    Replay a list of recorded joint commands in reverse order.
    recorded_cmds : list of np.array shape (4,), one per ctrl step
    This traces exactly the same path backwards.
    """
    print(f"  ← Returning via reverse path ({label}) …")
    for q_cmd in reversed(recorded_cmds):
        set_ctrl(data, q_cmd)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)
    ee = get_ee_pos(model, data)
    print(f"  ✓ Reverse done  EE={ee.round(4)}  "
          f"drift={np.linalg.norm(ee - HOME_EE)*1000:.2f}mm from home")


# ============================================================
# HOME via IK  (used at startup and after rotation)
# ============================================================

def move_to_home_ik(model, data, viewer):
    print("  → Homing (IK) …")
    q_now = get_current_q(data)
    q3    = q_now[:3].copy()
    for _ in range(20):
        q3 = ik_flat(model, data, HOME_EE, q3)
    q_home = apply_flat(q3[0], q3[1], q3[2])

    q_from = get_current_q(data)
    for i in range(HOME_STEPS + 1):
        alpha = i / HOME_STEPS
        q_cmd = (1 - alpha) * q_from + alpha * q_home
        set_ctrl(data, q_cmd)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

    ee = get_ee_pos(model, data)
    print(f"  ✓ Home  EE={ee.round(4)}  "
          f"drift={np.linalg.norm(ee - HOME_EE)*1000:.2f}mm")


# ============================================================
# STEP 1: TRANSLATION X  (forward + record, then reverse)
# ============================================================

def run_translation_x(model, data, viewer):
    traj_x  = QuinticPolynomial(X_START_POS[0], X_END_POS[0], X_MOTION_TIME)
    traj_y  = QuinticPolynomial(X_START_POS[1], X_END_POS[1], X_MOTION_TIME)
    traj_z  = QuinticPolynomial(X_START_POS[2], X_END_POS[2], X_MOTION_TIME)
    n_steps = int(X_MOTION_TIME * CTRL_HZ)

    print(f"\n  [1/3] Translation X  →")
    print(f"  {X_START_POS} → {X_END_POS}  T={X_MOTION_TIME}s\n")

    q_now    = get_current_q(data)
    q3       = q_now[:3].copy()
    recorded = []   # store every ctrl command for reverse replay

    for i in range(n_steps + 1):
        t      = i / CTRL_HZ
        target = np.array([traj_x.position(t),
                           traj_y.position(t),
                           traj_z.position(t)])
        q3    = ik_flat(model, data, target, q3)
        q_cmd = apply_flat(q3[0], q3[1], q3[2])

        recorded.append(q_cmd.copy())   # ← record

        set_ctrl(data, q_cmd)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

        if i % int(0.5 * CTRL_HZ) == 0:
            ee = get_ee_pos(model, data)
            print(f"    → t={t:4.1f}s  target={target.round(4)}  "
                  f"EE={ee.round(4)}  err={np.linalg.norm(ee-target)*1000:.2f}mm")

    ee_f = get_ee_pos(model, data)
    print(f"  ✓ Forward done  err={np.linalg.norm(ee_f-X_END_POS)*1000:.2f}mm")

    # Reverse: replay recorded commands backwards
    replay_reverse(model, data, viewer, recorded, "Translation X")


# ============================================================
# STEP 2: TRANSLATION Z  (forward + record, then reverse)
# ============================================================

def run_translation_z(model, data, viewer):
    traj_z  = QuinticPolynomial(Z_START, Z_END, Z_MOTION_TIME)
    n_steps = int(Z_MOTION_TIME * CTRL_HZ)

    print(f"\n  [2/3] Translation Z  →")
    print(f"  X={LINE_X}  Y={LINE_Y}  Z: {Z_START}→{Z_END}  T={Z_MOTION_TIME}s\n")

    q_now    = get_current_q(data)
    q3       = q_now[:3].copy()
    recorded = []   # ← record forward commands

    for i in range(n_steps + 1):
        t      = i / CTRL_HZ
        target = np.array([LINE_X, LINE_Y, traj_z.position(t)])
        q3    = ik_flat(model, data, target, q3)
        q_cmd = apply_flat(q3[0], q3[1], q3[2])

        recorded.append(q_cmd.copy())   # ← record

        set_ctrl(data, q_cmd)
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)

        if i % int(0.5 * CTRL_HZ) == 0:
            ee = get_ee_pos(model, data)
            print(f"    → t={t:4.1f}s  Z_target={target[2]:.4f}  "
                  f"EE={ee.round(4)}  err={abs(ee[2]-target[2])*1000:.2f}mm")

    ee_f = get_ee_pos(model, data)
    print(f"  ✓ Forward done  Z_err={abs(ee_f[2]-Z_END)*1000:.2f}mm")

    # Reverse: replay backwards
    replay_reverse(model, data, viewer, recorded, "Translation Z")


# ============================================================
# STEP 3: ROTATION  (forward only, then IK home)
# ============================================================

def run_rotation(model, data, viewer):
    traj_j4 = QuinticPolynomial(J4_START, J4_END, ROT_MOTION_TIME)
    n_steps  = int(ROT_MOTION_TIME * CTRL_HZ)

    print(f"\n  [3/3] Rotation  (EE fixed at {FIXED_POS})")
    print(f"  J4: {np.rad2deg(J4_START):.0f}°→{np.rad2deg(J4_END):.0f}°  T={ROT_MOTION_TIME}s")

    # Pre-solve start pose
    print(f"  Pre-solving …")
    q_now = get_current_q(data)
    q3    = q_now[:3].copy()
    for _ in range(20):
        q3 = ik_free(model, data, FIXED_POS, q3, J4_START)
    q_start_full = np.array([q3[0], q3[1], q3[2], J4_START])

    data.qpos[:N_ARM] = q_start_full; mujoco.mj_forward(model, data)
    ee_check = get_ee_pos(model, data)
    print(f"  Pre-solve EE={ee_check.round(4)}  "
          f"drift={np.linalg.norm(ee_check-FIXED_POS)*1000:.2f}mm")
    data.qpos[:N_ARM] = q_now; mujoco.mj_forward(model, data)

    # Approach via IK each step
    print(f"  Moving to start pose …")
    q3_app  = get_current_q(data)[:3].copy()
    j4_from = get_current_q(data)[3]
    for i in range(APPROACH_STEPS + 1):
        alpha      = i / APPROACH_STEPS
        j4_a       = (1 - alpha) * j4_from + alpha * J4_START
        tip_target = (1 - alpha) * get_ee_pos(model, data) + alpha * FIXED_POS
        for _ in range(3):
            q3_app = ik_free(model, data, tip_target, q3_app, j4_a)
        set_ctrl(data, np.array([q3_app[0], q3_app[1], q3_app[2], j4_a]))
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)
        q3_app = get_current_q(data)[:3].copy()

    ee_s = get_ee_pos(model, data)
    print(f"  At start. EE={ee_s.round(4)}  "
          f"drift={np.linalg.norm(ee_s-FIXED_POS)*1000:.2f}mm\n")

    # Rotation loop
    q3 = get_current_q(data)[:3].copy()
    for i in range(n_steps + 1):
        t  = i / CTRL_HZ
        j4 = traj_j4.position(t)
        for _ in range(3):
            q3 = ik_free(model, data, FIXED_POS, q3, j4)
        set_ctrl(data, np.array([q3[0], q3[1], q3[2], j4]))
        step_physics(model, data, viewer)
        time.sleep(1.0 / CTRL_HZ)
        q3 = get_current_q(data)[:3].copy()

        if i % int(0.5 * CTRL_HZ) == 0:
            ee    = get_ee_pos(model, data)
            drift = np.linalg.norm(ee - FIXED_POS)
            print(f"    → t={t:4.1f}s  J4={np.rad2deg(j4):+7.1f}°  "
                  f"EE={ee.round(4)}  drift={drift*1000:.2f}mm")

    ee_f = get_ee_pos(model, data)
    print(f"  ✓ Rotation done  drift={np.linalg.norm(ee_f-FIXED_POS)*1000:.2f}mm")


# ============================================================
# FULL SEQUENCE
# ============================================================

def run_sequence(model, data, viewer):
    print("\n" + "="*60)
    print("  STARTING FULL SEQUENCE")
    print("="*60)

    # 1. Translation X  (forward then reverse along same path)
    run_translation_x(model, data, viewer)

    # 2. Translation Z  (forward then reverse along same path)
    run_translation_z(model, data, viewer)

    # 3. Rotation  (forward, then IK home)
    run_rotation(model, data, viewer)
    print("\n  → Returning to home after rotation …")
    move_to_home_ik(model, data, viewer)

    print("\n" + "="*60)
    print("  SEQUENCE COMPLETE — press X to repeat")
    print("="*60 + "\n")


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

    # Startup: move to home EE position
    data.qpos[:] = 0.0; data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data); viewer.sync()
    time.sleep(0.5)
    print("  Moving to home EE position at startup …")
    move_to_home_ik(model, data, viewer)

    ee_home = get_ee_pos(model, data)
    print(f"\n  EE at home : {ee_home.round(4)} m")
    print(f"\n=== Controls ===")
    print(f"  X → full sequence  (X → ← | Z → ← | Rot → home)")
    print(f"  H → home only")
    print(f"  Q → quit\n")

    current_cmd = None

    try:
        while viewer.is_running():
            if keyboard.is_pressed('x') and current_cmd != 'x':
                current_cmd = 'x'
                run_sequence(model, data, viewer)
                time.sleep(0.3)

            elif keyboard.is_pressed('h') and current_cmd != 'h':
                current_cmd = 'h'
                print("[H] Homing …")
                move_to_home_ik(model, data, viewer)
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
            move_to_home_ik(model, data, viewer)
        print("Done.")


if __name__ == "__main__":
    main()