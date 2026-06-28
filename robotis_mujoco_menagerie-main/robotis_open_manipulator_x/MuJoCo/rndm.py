#!/usr/bin/env python3
"""
extract_ik_from_sim.py
======================
Run this ONCE on the machine that has the MuJoCo scene.xml.
It uses the exact same IK functions from combined.py to solve
the joint angles for each Cartesian endpoint, then prints
the np.array values ready to paste into combined_hw.py.

Usage:
    python extract_ik_from_sim.py
"""

import numpy as np
import mujoco
from pathlib import Path

# ── CHANGE THIS to your actual scene.xml path ──
MODEL_PATH = Path(
    r"C:\Users\thein\OneDrive\Documents\AURORA"
    r"\robotis_mujoco_menagerie-main"
    r"\robotis_open_manipulator_x"
    r"\scene.xml"
)

# ── Targets from combined.py ──
HOME_EE   = np.array([0.1417, 0.0210, 0.20])
X_END_POS = np.array([0.28,   0.0210, 0.20])
Z_END_POS = np.array([0.1417, 0.0210, 0.30])
J4_START  = -np.pi / 3
J4_END    =  np.pi / 3

N_ARM   = 4
ARM_MIN = np.array([-3.14159, -1.5, -1.5, -1.7 ])
ARM_MAX = np.array([ 3.14159,  1.5,  1.4,  1.97])


def apply_flat(q1, q2, q3):
    q4 = np.clip(-(q2 + q3), ARM_MIN[3], ARM_MAX[3])
    return np.array([q1, q2, q3, q4])


def get_ee(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_left")
    return data.xpos[bid].copy()


def ik_flat(model, data, target, q3_init, tol=1e-5, max_iter=500, lam=0.01):
    """Exact copy of combined.py ik_flat."""
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
        err  = target - pos0
        if np.linalg.norm(err) < tol:
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


def ik_free(model, data, target, q3_init, j4_fixed, tol=1e-5, max_iter=500, lam=0.01):
    """Exact copy of combined.py ik_free."""
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
        err  = target - pos0
        if np.linalg.norm(err) < tol:
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


def solve_and_print(label, model, data, target, q3_init, j4_fixed=None, iters=30):
    if j4_fixed is None:
        q3 = q3_init.copy()
        for _ in range(iters):
            q3 = ik_flat(model, data, target, q3)
        q_full = apply_flat(q3[0], q3[1], q3[2])
    else:
        q3 = q3_init.copy()
        for _ in range(iters):
            q3 = ik_free(model, data, target, q3, j4_fixed)
        q_full = np.array([q3[0], q3[1], q3[2], j4_fixed])

    # Verify
    data.qpos[:N_ARM] = q_full
    mujoco.mj_forward(model, data)
    ee = get_ee(model, data)
    err_mm = np.linalg.norm(ee - target) * 1000

    print(f"\n  {label}")
    print(f"    target  : {target}")
    print(f"    q (rad) : {q_full.round(6).tolist()}")
    print(f"    q (deg) : {np.rad2deg(q_full).round(3).tolist()}")
    print(f"    FK EE   : {ee.round(5)}")
    print(f"    error   : {err_mm:.3f} mm")

    # Restore
    data.qpos[:N_ARM] = [0.0]*N_ARM
    mujoco.mj_forward(model, data)

    return q_full


def main():
    print(f"Loading model: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"ERROR: scene.xml not found at {MODEL_PATH}")
        print("Update MODEL_PATH in this script to your actual path.")
        return

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    np.set_printoptions(precision=6, suppress=True)

    print("\n" + "="*55)
    print("  Solving IK for combined.py endpoints")
    print("="*55)

    q3_init = np.array([0.0, 0.0, -0.174533])

    home_q = solve_and_print("HOME / TX_START / TZ_START",
                              model, data, HOME_EE, q3_init)

    q3_init = home_q[:3].copy()
    tx_q    = solve_and_print("TX_END  [0.28, 0.021, 0.20]",
                               model, data, X_END_POS, q3_init)

    q3_init = home_q[:3].copy()
    tz_q    = solve_and_print("TZ_END  [0.1417, 0.021, 0.30]",
                               model, data, Z_END_POS, q3_init)

    q3_init = home_q[:3].copy()
    rot_start_q = solve_and_print("ROT_START (J4=-60°, EE@HOME)",
                                   model, data, HOME_EE, q3_init,
                                   j4_fixed=J4_START)

    q3_init = rot_start_q[:3].copy()
    rot_end_q = solve_and_print("ROT_END  (J4=+60°, EE@HOME)",
                                  model, data, HOME_EE, q3_init,
                                  j4_fixed=J4_END)

    print("\n\n" + "="*55)
    print("  PASTE THESE INTO combined_hw.py")
    print("="*55)
    print(f"\nHOME      = np.array({home_q.round(6).tolist()})")
    print(f"TX_START  = HOME.copy()")
    print(f"TX_END    = np.array({tx_q.round(6).tolist()})")
    print(f"TZ_START  = HOME.copy()")
    print(f"TZ_END    = np.array({tz_q.round(6).tolist()})")
    print(f"ROT_START = np.array({rot_start_q.round(6).tolist()})")
    print(f"ROT_END   = np.array({rot_end_q.round(6).tolist()})")

    print("\n" + "="*55)
    print("  Done. Copy the lines above into combined_hw.py")
    print("="*55)


if __name__ == "__main__":
    main()