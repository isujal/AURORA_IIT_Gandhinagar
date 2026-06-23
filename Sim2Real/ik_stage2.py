# -----------------------------------------------------------------------------
# Robotic Arm Controller with Real-Time Obstacle Avoidance
# -----------------------------------------------------------------------------
# Provided automatically:
#   robot    - Robot model with inverse_kinematics() and forward_kinematics()
#   sim2real - SimToReal instance
#   model    - mujoco.MjModel
#   data     - mujoco.MjData
#
# HOW IT WORKS:
#   1. Solves IK for the target position
#   2. During motion, continuously checks Euclidean distance from EE to obstacle
#   3. When EE gets close to obstacle, joints are nudged away using Jacobian
#   4. When EE clears the obstacle, normal LERP motion resumes toward target
# -----------------------------------------------------------------------------
import numpy as np
import time

# =============================================================================
# USER CONFIGURATION -- only change things here
# =============================================================================
TARGET_POSITION  = np.array([0.15, 0.08, 0.0])   # where arm must reach
AVOID_COORD      = np.array([0.10, 0.05, 0.0])   # coordinate to avoid
AVOID_RADIUS     = 0.04    # danger zone radius in metres
REPULSE_GAIN     = 0.5     # how hard joints push away (increase if not enough)
DANGER_SCALE     = 3.0     # repulsion starts at DANGER_SCALE * AVOID_RADIUS
MOTION_TIME      = 4.0     # total motion duration in seconds
DT               = 0.01    # time step (100 Hz)
USE_HARDWARE     = False   # set True to also drive real robot
# =============================================================================

stop_requested = False


def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested - will terminate safely.")


# -----------------------------------------------------------------------------
# MATHS UTILITIES
# -----------------------------------------------------------------------------

def euclidean(a, b):
    """Straight-line distance between two 3D points."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def repulsion_weight(ee_pos, avoid, radius, scale):
    """
    Converts Euclidean distance into a smooth 0->1 repulsion weight.

    Zones:
      dist >= scale*radius  ->  w = 0.0   (far away, no effect)
      dist == radius        ->  w = 1.0   (at boundary, max repulsion)
      dist <  radius        ->  w = 1.0   (inside zone, max repulsion)

    Linear ramp between the two thresholds:
        w = (influence_dist - dist) / (influence_dist - radius)
    """
    dist           = euclidean(ee_pos, avoid)
    influence_dist = scale * radius

    if dist >= influence_dist:
        return 0.0
    if dist <= radius:
        return 1.0

    w = (influence_dist - dist) / (influence_dist - radius)
    return float(np.clip(w, 0.0, 1.0))


def numerical_jacobian(robot, q, eps=1e-4):
    """
    Numerically estimate the 3xN Jacobian matrix by finite differences.

    J[:, i] = (FK(q + eps*e_i)[:3,3] - FK(q)[:3,3]) / eps

    Each column tells us: if joint i moves by a tiny amount,
    how much does the end-effector position change in XYZ?
    """
    n  = len(q)
    J  = np.zeros((3, n))
    p0 = robot.forward_kinematics(q)[:3, 3]

    for i in range(n):
        q_plus     = q.copy()
        q_plus[i] += eps
        p_plus     = robot.forward_kinematics(q_plus)[:3, 3]
        J[:, i]    = (p_plus - p0) / eps

    return J


def repulsion_delta_q(robot, q, ee_pos, avoid, weight, gain):
    """
    Compute a joint-space nudge that pushes EE away from obstacle.

    Steps:
      1. repulse_vec = unit vector from obstacle -> EE  (push-away direction)
      2. J = numerical Jacobian at current q
      3. dq = J^T @ repulse_vec   (map Cartesian repulsion to joint space)
      4. normalise dq, scale by weight * gain
    """
    avoid      = np.array(avoid)
    ee_pos     = np.array(ee_pos)
    diff       = ee_pos - avoid
    dist       = np.linalg.norm(diff)

    if dist < 1e-6:
        repulse_vec = np.array([0.0, 1.0, 0.0])   # fallback if exactly on obstacle
    else:
        repulse_vec = diff / dist                  # unit vector away from obstacle

    J          = numerical_jacobian(robot, q)
    dq         = J.T @ repulse_vec                 # project into joint space

    norm = np.linalg.norm(dq)
    if norm > 1e-6:
        dq = dq / norm                             # normalise

    return dq * weight * gain


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def run_challenge(sim2real, robot, model, data):
    global stop_requested
    stop_requested = False

    if hasattr(sim2real, 'hardware_enabled'):
        sim2real.hardware_enabled = USE_HARDWARE

    ee_body_id = model.body("gripper_link").id
    q_home     = np.zeros(3)
    steps      = int(MOTION_TIME / DT)

    print("=" * 60)
    print("  OBSTACLE-AWARE ARM CONTROLLER")
    print("=" * 60)
    print(f"  Target   : {TARGET_POSITION}")
    print(f"  Obstacle : {AVOID_COORD}  radius={AVOID_RADIUS} m")
    print(f"  Hardware : {USE_HARDWARE}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Solve IK for target
    # -------------------------------------------------------------------------
    print("\n[1] Solving IK for target...")
    q_target = robot.inverse_kinematics(
        TARGET_POSITION,
        np.eye(3),
        np.array([0.5, 0.3, 0.2])
    )
    T_verify  = robot.forward_kinematics(q_target)
    ee_verify = T_verify[:3, 3]
    ik_error  = euclidean(TARGET_POSITION, ee_verify)

    print(f"    Joint angles (deg) : {np.rad2deg(q_target).round(2)}")
    print(f"    Achieved position  : {ee_verify.round(4)}")
    print(f"    IK error           : {ik_error:.8f} m")

    if ik_error > 0.01:
        print("    WARNING: IK error > 1 cm. Target may be outside workspace.")

    # -------------------------------------------------------------------------
    # STEP 2: Reset to home
    # -------------------------------------------------------------------------
    print("\n[2] Resetting to home...")
    sim2real.set_joint_positions(q_home)
    time.sleep(0.5)

    # -------------------------------------------------------------------------
    # STEP 3: Motion loop with real-time obstacle avoidance
    # -------------------------------------------------------------------------
    print(f"\n[3] Starting motion ({MOTION_TIME}s, {steps} steps)...")
    print(f"    Repulsion gain={REPULSE_GAIN}, "
          f"influence zone={DANGER_SCALE * AVOID_RADIUS:.3f} m")
    print("-" * 60)

    avoiding       = False   # track whether we are currently in avoidance mode
    max_repulse_w  = 0.0     # for logging

    for i in range(steps):
        if stop_requested:
            print("Motion stopped by user.")
            break

        # -- Alpha: normalised time 0.0 -> 1.0 --------------------------------
        alpha = i / (steps - 1) if steps > 1 else 1.0

        # -- Base LERP: q_home -> q_target ------------------------------------
        # q_lerp = (1 - alpha)*q_home + alpha*q_target
        # Since q_home = zeros: q_lerp = alpha * q_target
        q_lerp = (1 - alpha) * q_home + alpha * q_target

        # -- Get current EE position from MuJoCo physics ----------------------
        ee_pos = data.xpos[ee_body_id].copy()

        # -- Euclidean distance from EE to obstacle ---------------------------
        dist = euclidean(ee_pos, AVOID_COORD)

        # -- Repulsion weight (0 = safe, 1 = at obstacle boundary) ------------
        w = repulsion_weight(ee_pos, AVOID_COORD, AVOID_RADIUS, DANGER_SCALE)

        if w > 0:
            # -- IN AVOIDANCE MODE -------------------------------------------
            # Compute joint-space nudge pushing EE away from obstacle
            dq_rep = repulsion_delta_q(
                robot, q_lerp, ee_pos, AVOID_COORD, w, REPULSE_GAIN
            )
            q_cmd = q_lerp + dq_rep

            if not avoiding:
                print(f"\n  >>> AVOIDANCE ON  at step {i+1}  "
                      f"dist={dist:.4f} m  w={w:.3f}")
                avoiding = True

            if w > max_repulse_w:
                max_repulse_w = w

        else:
            # -- NORMAL MODE: pure LERP toward target -------------------------
            q_cmd = q_lerp

            if avoiding:
                print(f"  >>> AVOIDANCE OFF at step {i+1}  "
                      f"dist={dist:.4f} m  (peak w={max_repulse_w:.3f})")
                avoiding      = False
                max_repulse_w = 0.0

        # -- Send joint command to sim (and hardware if enabled) --------------
        sim2real.set_joint_positions(q_cmd)

        # -- Log every 30 steps -----------------------------------------------
        if i % 30 == 0 or i == steps - 1:
            T_fk = robot.forward_kinematics(q_cmd)
            print(
                f"  step {i+1:3d}/{steps} | "
                f"alpha={alpha:.2f} | "
                f"dist_obs={dist:.4f}m | "
                f"w={w:.2f} | "
                f"q={np.rad2deg(q_cmd).round(1)} | "
                f"EE={ee_pos.round(3)}"
            )
            if USE_HARDWARE:
                hw = sim2real.get_hardware_positions()
                if hw is not None:
                    print(f"    HW={np.rad2deg(hw).round(1)} deg")

        time.sleep(DT)

    # -------------------------------------------------------------------------
    # STEP 4: Final report
    # -------------------------------------------------------------------------
    ee_final      = data.xpos[ee_body_id].copy()
    T_final       = robot.forward_kinematics(q_target)
    fk_final      = T_final[:3, 3]
    final_dist_obs = euclidean(ee_final, AVOID_COORD)
    final_err     = euclidean(ee_final, TARGET_POSITION)

    print("\n" + "=" * 60)
    print("  MOTION COMPLETE")
    print("=" * 60)
    print(f"  Final EE (MuJoCo)       : {ee_final.round(4)}")
    print(f"  Final EE (FK)           : {fk_final.round(4)}")
    print(f"  Error to target         : {final_err:.4f} m")
    print(f"  Final dist to obstacle  : {final_dist_obs:.4f} m  "
          f"({'SAFE' if final_dist_obs > AVOID_RADIUS else 'WARN: inside zone'})")
    print("=" * 60)

    return {
        "completed_successfully": not stop_requested,
        "final_ee_mujoco":        ee_final.tolist(),
        "final_ee_fk":            fk_final.tolist(),
        "error_to_target":        float(final_err),
        "final_dist_to_obstacle": float(final_dist_obs),
    }