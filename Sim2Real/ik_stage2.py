# -----------------------------------------------------------------------------
# Part 2 - Sim-to-Real Animation
# -----------------------------------------------------------------------------
# Provided automatically:
#   sim2real - SimToReal instance (controls sim + hardware simultaneously)
#   robot    - Robot model for FK verification
#   model    - mujoco.MjModel
#   data     - mujoco.MjData
#
# KEY API:
#   sim2real.set_joint_positions(q)   -> send angles to sim + hardware
#   sim2real.get_hardware_positions() -> read actual motor angles (if connected)
#
# Paste your q_solution from Part 1 output below, then complete the animation.
# -----------------------------------------------------------------------------
import numpy as np
import time

# -- TODO: Paste your IK solution (radians) from Part 1 output here -----------
q_solution = np.array([0.4823, 0.1156, -0.2341])  # replace with your Part 1 values

# -- Hardware toggle -----------------------------------------------------------
USE_HARDWARE = False   # set True to also move the real Robox

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

    # -- Reset to home ----------------------------------------------------------
    sim2real.set_joint_positions(np.zeros(3))
    time.sleep(0.5)

    # -- Animate home -> IK solution --------------------------------------------
    ee_body_id  = model.body("gripper_link").id
    motion_time = 3.0
    dt          = 0.01
    steps       = int(motion_time / dt)

    print(f"\n===== Animating: home -> IK solution ({motion_time} s) =====")

    for i in range(steps):
        if stop_requested:
            print("Motion stopped by user.")
            break

        alpha = i / (steps - 1) if steps > 1 else 1.0

        # TODO: linearly interpolate from np.zeros(3) to q_solution using alpha
        q = alpha * q_solution

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
    T_final  = robot.forward_kinematics(q_solution)
    fk_final = T_final[:3, 3]

    print("\n===== Motion complete =====")
    print(f"Final EE (MuJoCo) : {ee_final.round(4)}")
    print(f"Final EE (FK)     : {fk_final.round(4)}")

    return {
        "completed_successfully": not stop_requested,
        "final_ee_mujoco":        ee_final.tolist(),
        "final_ee_fk":            fk_final.tolist(),
    }
