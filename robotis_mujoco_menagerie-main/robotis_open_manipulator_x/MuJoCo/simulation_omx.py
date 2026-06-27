#!/usr/bin/env python3
"""
robot_traffic_controller_mujoco.py
===================================
Traffic Controller (STOP / GO / LEFT / RIGHT / HOME) running entirely
inside a MuJoCo simulation of the OpenManipulator-X.

Replaces every real-hardware call from the original
robot_traffic_controller.py with MuJoCo equivalents, following exactly
the same pattern used by the LEAP-hand MediaPipe demo:

    data.qpos[:] = target_qpos
    mujoco.mj_forward(model, data)
    viewer.sync()

Dependencies
------------
    pip install mujoco keyboard numpy

Controls
--------
    G  →  GO      (arm points forward / horizontal)
    S  →  STOP    (arm points upward)
    L  →  LEFT    (base rotated +90 °)
    R  →  RIGHT   (base rotated −90 °)
    H  →  HOME    (rest / neutral)
    Q  →  QUIT
"""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# MODEL PATH
# ============================================================
# Point this to your OpenManipulator-X MJCF / URDF file.
# By default we look two levels up from this script, mirroring
# the LEAP-hand demo layout:
#   <repo_root>/models/open_manipulator_x.xml
#
# Override by setting the env-var  OMX_MODEL_PATH  or editing
# MODEL_PATH directly.
# ============================================================

import os

_default_model = Path(r"C:\Users\thein\OneDrive\Documents\AURORA\robotis_mujoco_menagerie-main\robotis_open_manipulator_x\scene.xml")

MODEL_PATH = Path(os.environ.get("OMX_MODEL_PATH", _default_model))

# ============================================================
# JOINT LIMITS  (radians)   — OpenManipulator-X Dynamixel XM430
# ============================================================
# These match the physical limits used in the original controller
# and the AURORA MJCF model.  Adjust if your MJCF sets tighter
# limits via <joint range="...">.

JOINT_MIN = np.array([-2.827, -1.571, -1.571, -1.745])   # [joint1..4]
JOINT_MAX = np.array([ 2.827,  1.571,  1.571,  1.745])

# Number of actuated joints on OMX
N_JOINTS = 4

# ============================================================
# NAMED POSES  (joint angles in radians)
# ============================================================
# Layout: [joint1_base_rotation, joint2_shoulder, joint3_elbow, joint4_wrist]
#
# TODO: Tune these angles for your specific MJCF / physical setup.
#       The values below are sensible starting points.

POSES = {
    # Arm points straight up — classic STOP signal
    "STOP":  np.array([ 0.0,   0.0,  -0.1745, -1.5708]),

    # Arm level, pointing forward — GO
    "GO":    np.array([ 0.0,   0.0,  -0.1745,  0.0   ]),

    # Base rotated +90 ° (left from robot's perspective)
    "LEFT":  np.array([ 1.5708, 0.0, -0.1745,  0.0   ]),

    # Base rotated −90 ° (right)
    "RIGHT": np.array([-1.5708, 0.0, -0.1745,  0.0   ]),

    # Safe neutral / rest position
    "HOME":  np.array([ 0.0,   0.0,  -0.1745,  0.0   ]),
}

# ============================================================
# MOTION PARAMETERS
# ============================================================

INTERP_STEPS = 60    # number of qpos interpolation steps per move
STEP_SLEEP   = 1/60  # seconds between steps  (~60 fps)

# ============================================================
# HELPERS
# ============================================================

def clamp_qpos(qpos: np.ndarray) -> np.ndarray:
    """Clamp joint angles to hardware limits."""
    return np.clip(qpos, JOINT_MIN, JOINT_MAX)


def move_to_pose(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    viewer,
    target_angles: np.ndarray,
    steps: int = INTERP_STEPS,
) -> None:
    """
    Smoothly interpolate the simulation from its current qpos to
    target_angles, syncing the viewer at every step.

    This mirrors the LEAP-hand demo pattern:
        data.qpos[:] = interpolated_value
        mujoco.mj_forward(model, data)
        viewer.sync()

    Parameters
    ----------
    model         : MuJoCo model
    data          : MuJoCo data
    viewer        : passive viewer (mujoco.viewer.launch_passive)
    target_angles : desired joint angles (radians), shape (N_JOINTS,)
    steps         : interpolation resolution
    """
    start_qpos = data.qpos[:N_JOINTS].copy()
    target_qpos = clamp_qpos(target_angles)

    for step in range(steps + 1):
        alpha = step / steps
        interp = (1.0 - alpha) * start_qpos + alpha * target_qpos

        data.qpos[:N_JOINTS] = interp
        mujoco.mj_forward(model, data)
        viewer.sync()

        time.sleep(STEP_SLEEP)


def reset_to_home(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    viewer,
) -> None:
    """Instantly snap the robot to HOME (used at startup)."""
    data.qpos[:N_JOINTS] = POSES["HOME"]
    mujoco.mj_forward(model, data)
    viewer.sync()

# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # 1. Load MuJoCo model
    # --------------------------------------------------------
    print(f"Loading model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Set the OMX_MODEL_PATH environment variable or edit MODEL_PATH "
            "at the top of this script."
        )

    if MODEL_PATH.suffix == ".urdf":
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    else:
        # .xml  /  .mjcf
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    data = mujoco.MjData(model)

    # --------------------------------------------------------
    # 2. Launch passive viewer  (same as LEAP-hand demo)
    # --------------------------------------------------------
    viewer = mujoco.viewer.launch_passive(model, data)

    # --------------------------------------------------------
    # 3. Import keyboard
    # --------------------------------------------------------
    try:
        import keyboard
    except ImportError:
        print("Install the keyboard package first:  pip install keyboard")
        return

    # --------------------------------------------------------
    # 4. Move to HOME
    # --------------------------------------------------------
    print("Moving to HOME…")
    reset_to_home(model, data, viewer)
    time.sleep(0.5)

    # --------------------------------------------------------
    # 5. Print help
    # --------------------------------------------------------
    print("\n=== MuJoCo Robot Traffic Controller ===")
    print("  G  →  GO    (arm forward / horizontal)")
    print("  S  →  STOP  (arm pointing up)")
    print("  L  →  LEFT  (base +90 °)")
    print("  R  →  RIGHT (base −90 °)")
    print("  H  →  HOME  (neutral)")
    print("  Q  →  QUIT")
    print("=========================================\n")

    # --------------------------------------------------------
    # 6. Control loop
    # --------------------------------------------------------
    #   Same debounce pattern as original:
    #   track current_command, only act on a new key press.

    KEY_MAP = {
        'g': ("GO",    POSES["GO"]),
        's': ("STOP",  POSES["STOP"]),
        'l': ("LEFT",  POSES["LEFT"]),
        'r': ("RIGHT", POSES["RIGHT"]),
        'h': ("HOME",  POSES["HOME"]),
    }

    current_command: str | None = None

    try:
        while viewer.is_running():

            acted = False

            for key, (label, angles) in KEY_MAP.items():
                if keyboard.is_pressed(key) and current_command != key:
                    print(f"  → {label}")
                    move_to_pose(model, data, viewer, angles)
                    current_command = key
                    acted = True
                    time.sleep(0.3)   # debounce
                    break

            if keyboard.is_pressed('q'):
                print("Quitting…")
                break

            # Reset debounce once all tracked keys are released
            if not any(keyboard.is_pressed(k) for k in KEY_MAP):
                current_command = None

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        # --------------------------------------------------------
        # 7. Return to HOME before exiting  (mirrors finally block)
        # --------------------------------------------------------
        print("Returning to HOME…")
        if viewer.is_running():
            move_to_pose(model, data, viewer, POSES["HOME"], steps=40)
        print("Done.")


if __name__ == "__main__":
    main()