import numpy as np
import time

USE_HARDWARE = False

# Obstacle parameters (metres)
OBSTACLE_POS    = np.array([0.15, 0.15])   # (15 cm, 15 cm) in XY
OBSTACLE_RADIUS = 0.03                      # 3 cm safety zone

stop_requested = False


def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested - execution will terminate safely.")


def _check_obstacle(ee_pos, step=None):
    """
    Returns True if EE is inside the obstacle safety zone.
    ee_pos: full 3D position from FK or MuJoCo (we only use X and Y)
    """
    ee_xy   = np.array([ee_pos[0], ee_pos[1]])
    dist    = np.linalg.norm(ee_xy - OBSTACLE_POS)
    in_zone = dist < OBSTACLE_RADIUS

    if in_zone and step is not None:
        print(f"  ⚠ WARNING step {step}: EE at ({ee_pos[0]*100:.1f},{ee_pos[1]*100:.1f}) cm "
              f"is {dist*100:.1f} cm from obstacle — INSIDE safety zone!")
    return in_zone


def run_challenge(sim2real, robot, model, data):
    global stop_requested
    stop_requested = False

    if hasattr(sim2real, 'hardware_enabled'):
        sim2real.hardware_enabled = USE_HARDWARE
    print(f"Hardware enabled: {USE_HARDWARE}")


    q_target_deg = np.array([60, -60, -20])
    q_target     = np.deg2rad(q_target_deg)

    # ── Pre-flight FK check BEFORE moving anything ─────────────────────────
    print("\n===== FORWARD KINEMATICS CHECK =====")
    T_check  = robot.forward_kinematics(q_target)
    ee_check = T_check[:3, 3]
    dist_to_obs = np.linalg.norm(ee_check[:2] - OBSTACLE_POS)

    print(f"Joint angles (deg)       : {q_target_deg}")
    print(f"Predicted EE position (m): {ee_check.round(4)}")
    print(f"Predicted EE (cm)        : ({ee_check[0]*100:.2f}, {ee_check[1]*100:.2f})")
    print(f"Distance to obstacle     : {dist_to_obs*100:.2f} cm  "
          f"({'✓ SAFE' if dist_to_obs >= OBSTACLE_RADIUS else '✗ TOO CLOSE — adjust angles!'})")

    if dist_to_obs < OBSTACLE_RADIUS:
        print("\n⛔ Final target too close to obstacle. Adjust q_target_deg and re-run.")
        return {"completed_successfully": False, "reason": "target inside obstacle zone"}

    # ── Pre-flight: check entire interpolated path ─────────────────────────
    print("\n===== PRE-FLIGHT PATH CHECK (all 300 steps) =====")
    path_safe    = True
    worst_dist   = float('inf')

    for i in range(300):
        alpha_pre = i / 299
        q_pre     = alpha_pre * q_target
        T_pre     = robot.forward_kinematics(q_pre)
        ee_pre    = T_pre[:3, 3]
        d         = np.linalg.norm(ee_pre[:2] - OBSTACLE_POS)
        if d < worst_dist:
            worst_dist = d
        if d < OBSTACLE_RADIUS:
            print(f"  ⚠ Path conflict at step {i}: "
                  f"EE=({ee_pre[0]*100:.1f},{ee_pre[1]*100:.1f}) cm, "
                  f"dist={d*100:.1f} cm")
            path_safe = False

    print(f"Worst clearance along path : {worst_dist*100:.2f} cm")
    if path_safe:
        print("✓ Path is collision-free in FK prediction — safe to run.")
    else:
        print("✗ Path intersects obstacle zone — adjust q_target_deg!")
        return {"completed_successfully": False, "reason": "path through obstacle"}

    # ── Reset to home ──────────────────────────────────────────────────────
    sim2real.set_joint_positions(np.zeros(3))
    time.sleep(0.5)

    # ── Animate home → target ──────────────────────────────────────────────
    motion_time = 3.0
    dt          = 0.01
    steps       = int(motion_time / dt)   # 300 steps

    print(f"\n===== Animating to target ({motion_time}s, {steps} steps) =====")
    print(f"Obstacle at ({OBSTACLE_POS[0]*100:.0f}, {OBSTACLE_POS[1]*100:.0f}) cm, "
          f"safety radius = {OBSTACLE_RADIUS*100:.0f} cm\n")

    ee_body_id       = model.body("gripper_link").id
    collision_detected = False

    for i in range(steps):
        if stop_requested:
            print("Motion stopped by user.")
            break

        # alpha: 0.0 at step 0 → 1.0 at final step
        alpha = i / (steps - 1) if steps > 1 else 1.0

        # ── TODO 2: Linear interpolation ──────────────────────────────────
        # lerp(start, end, t) = start + t*(end - start)
        # start = zeros(3), end = q_target
        # → q = 0 + alpha*(q_target - 0) = alpha * q_target
        q = alpha * q_target

        sim2real.set_joint_positions(q)

        # ── Live obstacle check at every step ─────────────────────────────
        ee_mj = data.xpos[ee_body_id].copy()
        if _check_obstacle(ee_mj, step=i+1):
            collision_detected = True
            # Don't stop — let it finish so we see the full path,
            # but flag it so we block hardware execution

        # ── Print every 50 steps ──────────────────────────────────────────
        if i % 50 == 0 or i == steps - 1:
            T_fk   = robot.forward_kinematics(q)
            ee_fk  = T_fk[:3, 3]
            d_live = np.linalg.norm(ee_mj[:2] - OBSTACLE_POS)
            print(f"Step {i+1:3d}/{steps} | "
                  f"q(deg)={np.rad2deg(q).round(1)} | "
                  f"MuJoCo EE=({ee_mj[0]*100:.1f},{ee_mj[1]*100:.1f}) cm | "
                  f"FK=({ee_fk[0]*100:.1f},{ee_fk[1]*100:.1f}) cm | "
                  f"dist_obs={d_live*100:.1f} cm")

        time.sleep(dt)

    # ── Final results ──────────────────────────────────────────────────────
    ee_final = data.xpos[ee_body_id].copy()
    T_final  = robot.forward_kinematics(q_target)
    fk_final = T_final[:3, 3]

    print("\n" + "=" * 55)
    print("              FINAL RESULTS")
    print("=" * 55)
    print(f"Target joint angles (deg)  : {q_target_deg}")
    print(f"Final EE (MuJoCo) (cm)     : ({ee_final[0]*100:.2f}, {ee_final[1]*100:.2f})")
    print(f"Final EE (FK)     (cm)     : ({fk_final[0]*100:.2f}, {fk_final[1]*100:.2f})")
    print(f"Distance to obstacle (final): {np.linalg.norm(ee_final[:2]-OBSTACLE_POS)*100:.2f} cm")

    if collision_detected:
        print("\n⛔ Collision detected during motion — do NOT enable hardware.")
        print("   Adjust q_target_deg and re-run simulation.")
    else:
        print("\n✓ No collision detected.")
        if not USE_HARDWARE:
            print("  Set USE_HARDWARE = True and re-run to execute on real robot.")

    print("=" * 55)

    return {
        "completed_successfully": not stop_requested and not collision_detected,
        "collision_detected":     collision_detected,
        "final_ee_mujoco":        ee_final.tolist(),
        "final_ee_fk":            fk_final.tolist(),
        "q_target_deg":           q_target_deg.tolist(),
        "worst_path_clearance_cm": round(worst_dist * 100, 2),
    }