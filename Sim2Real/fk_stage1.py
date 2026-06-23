# -----------------------------------------------------------------------------
# Part 1 - Forward Kinematics with Sim-to-Real
# -----------------------------------------------------------------------------
# Provided automatically:
#   sim2real - SimToReal instance (controls sim + hardware simultaneously)
#   robot    - Robot model with forward_kinematics() method
#   model    - mujoco.MjModel
#   data     - mujoco.MjData
#
# API:
#   sim2real.set_joint_positions(q)   -> send joint angles to sim + hardware
#   sim2real.get_hardware_positions() -> read actual motor angles (if connected)
#   robot.forward_kinematics(q)       -> 4x4 homogeneous transform T
#                                        T[:3, 3] = end-effector position
#
# -----------------------------------------------------------------------------
import numpy as np
import time

# -- Set to True for sim + real hardware, False for simulation only ---------
USE_HARDWARE = False  # TODO: change to True to also move the real robot

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested - execution will terminate safely.")


def run_challenge(sim2real, robot, model, data):
    global stop_requested
    stop_requested = False

    if hasattr(sim2real, 'hardware_enabled'):
        sim2real.hardware_enabled = USE_HARDWARE
    print(f"Hardware enabled: {USE_HARDWARE}")

    # -- TODO: Set your target joint angles (in DEGREES) ----------------------
    # Choose three joint angles within the joint limits.
    q_target_deg = np.array([90, -90, 45])

    q_target = np.deg2rad(q_target_deg)

    # -- Forward Kinematics - predict end-effector pose ------------------------
    print("\n===== FORWARD KINEMATICS =====")
    print(f"Target joint angles (deg): {q_target_deg}")

    # TODO: call robot.forward_kinematics() with the target angles in radians
    T_end_effector = robot.forward_kinematics(q_target)

    ee_position    = T_end_effector[:3, 3]
    ee_orientation = T_end_effector[:3, :3]

    print(f"\nPredicted end-effector position (m): {ee_position}")
    print(f"\nPredicted orientation matrix:")
    print(ee_orientation)

    # -- Reset to home ----------------------------------------------------------
    sim2real.set_joint_positions(np.zeros(3))
    time.sleep(0.5)

    # -- Animate home -> target -------------------------------------------------
    motion_time = 3.0
    dt          = 0.01
    steps       = int(motion_time / dt)

    print(f"\n===== Starting motion ({motion_time} s) =====")

    ee_body_id = model.body("gripper_link").id

    for i in range(steps):
        if stop_requested:
            print("Motion stopped by user.")
            break

        alpha = i / (steps - 1) if steps > 1 else 1.0

        # TODO: linearly interpolate from np.zeros(3) to q_target using alpha
        q = alpha * q_target

        sim2real.set_joint_positions(q)

        if i % 50 == 0 or i == steps - 1:
            ee_mj = data.xpos[ee_body_id].copy()
            T_fk  = robot.forward_kinematics(q)
            print(f"Step {i+1:3d}/{steps}  q(deg)={np.rad2deg(q).round(1)}  "
                  f"MuJoCo EE={ee_mj.round(4)}  FK={T_fk[:3,3].round(4)}")
            if USE_HARDWARE:
                hw = sim2real.get_hardware_positions()
                if hw is not None:
                    print(f"  Hardware = {np.rad2deg(hw).round(1)} deg")

        time.sleep(dt)

    # -- Final results ----------------------------------------------------------
    ee_final = data.xpos[ee_body_id].copy()
    T_final  = robot.forward_kinematics(q_target)
    fk_final = T_final[:3, 3]

    print("\n" + "=" * 50)
    print("           FINAL RESULTS")
    print("=" * 50)
    print(f"Target joint angles (deg): {q_target_deg}")
    print(f"\n--- MuJoCo Simulation ---")
    print(f"  EE position (MuJoCo): {ee_final.round(4)}")
    print(f"  EE position (FK)    : {fk_final.round(4)}")
    print(f"  Orientation matrix  :")
    for row in T_final[:3, :3]:
        print(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")

    if USE_HARDWARE:
        print("\n--- Hardware ---")
        hw_pos = sim2real.get_hardware_positions()
        if hw_pos is not None:
            T_hw   = robot.forward_kinematics(hw_pos)
            hw_ee  = T_hw[:3, 3]
            print(f"  Joint angles (deg): {np.rad2deg(hw_pos).round(1)}")
            print(f"  EE position (m)   : {hw_ee.round(4)}")
        else:
            print("  Hardware readback not available.")

    print("\n" + "=" * 50)
    print("           Motion complete")
    print("=" * 50)

    return {
        "completed_successfully": not stop_requested,
        "ee_position_fk":         fk_final.tolist(),
        "ee_position_mujoco":     ee_final.tolist(),
        "q_target_deg":           q_target_deg.tolist(),
    }
