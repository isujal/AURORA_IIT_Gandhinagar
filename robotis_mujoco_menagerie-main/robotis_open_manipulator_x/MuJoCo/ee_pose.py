#!/usr/bin/env python3
"""
get_ee_home.py
==============
Prints the end-effector (gripper tip) position when all joints are at 0 rad.
Uses the same DH parameters and MODEL_PATH as straight_line_y.py.
"""

from pathlib import Path
import mujoco
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
# METHOD 1 — MuJoCo: read body position directly from the sim
# ============================================================
def get_ee_mujoco():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    # All joints already zero at init, just forward-propagate
    mujoco.mj_forward(model, data)

    # gripper_left body is the end-effector in the XML
    body_id = model.body("gripper_left").id
    pos     = data.xpos[body_id].copy()
    return pos

# ============================================================
# METHOD 2 — DH FK: compute analytically
# ============================================================
L1  = 0.012;   L2  = 0.0595
L3  = 0.128;   L4  = 0.124;   L5  = 0.130

def _dh(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,  sa,     ca,    d   ],
        [0.,  0.,     0.,    1.  ],
    ])

def fk_dh(q):
    q1, q2, q3, q4 = q
    T = (  _dh(L1, L2,       np.pi/2, q1)
         @ _dh(L3, 0.,       0.,      q2)
         @ _dh(L4, 0.,       0.,      q3)
         @ _dh(L5, 0.,       0.,      q4) )
    return T[:3, 3]

# ============================================================
# MAIN
# ============================================================
q_home = [0.0, 0.0, 0.0, 0.0]

print("=" * 45)
print("  EE position at all-zero joints")
print("=" * 45)

# Method 1 — MuJoCo body position
pos_mj = get_ee_mujoco()
print(f"\n  MuJoCo (gripper_left body):")
print(f"    X = {pos_mj[0]*1000:8.2f} mm  ({pos_mj[0]:.6f} m)")
print(f"    Y = {pos_mj[1]*1000:8.2f} mm  ({pos_mj[1]:.6f} m)")
print(f"    Z = {pos_mj[2]*1000:8.2f} mm  ({pos_mj[2]:.6f} m)")

# Method 2 — DH FK
pos_dh = fk_dh(q_home)
print(f"\n  DH Forward Kinematics:")
print(f"    X = {pos_dh[0]*1000:8.2f} mm  ({pos_dh[0]:.6f} m)")
print(f"    Y = {pos_dh[1]*1000:8.2f} mm  ({pos_dh[1]:.6f} m)")
print(f"    Z = {pos_dh[2]*1000:8.2f} mm  ({pos_dh[2]:.6f} m)")

print(f"\n  Difference (MuJoCo - DH):")
diff = pos_mj - pos_dh
print(f"    ΔX = {diff[0]*1000:.3f} mm")
print(f"    ΔY = {diff[1]*1000:.3f} mm")
print(f"    ΔZ = {diff[2]*1000:.3f} mm")

print("=" * 45)