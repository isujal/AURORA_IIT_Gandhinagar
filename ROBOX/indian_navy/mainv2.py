"""
Final Challenge 12 code with x, y, z coordinates of the laser hit on the wall and whether it is on the wall or not. The code will run a raster scan pattern with joint 2 moving in position mode and joint 3 moving in velocity mode.
The laser hit will be computed based on the actual joint angles and the robot's forward kinematics, projecting a ray from the end-effector to determine where it intersects with a wall at a fixed X coordinate.
The results will be printed in real-time during the scan, and a final report will be shown when the scan is stopped.
"""


import time
import math
import numpy as np
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode
from robot_core.robot_model import Robot

""" git final push"""

stop_requested = False
_robot = None
_actuator_controller = None

# ── Laser projection parameters ────────────────────────────────────────────────
LASER_HEIGHT_ABOVE_EE = 5.0    # cm — laser emitter is this far above the end-effector
LASER_TILT_DEG        = 15.0   # deg — laser beam tilts upward from the EE plane
WALL_X                = -25.0  # cm — X coordinate of the target wall
WALL_Y_MIN            =  0.2   # cm — valid wall Y lower bound
WALL_Y_MAX            = 15.0   # cm — valid wall Y upper bound


# ── Laser hit computation ──────────────────────────────────────────────────────
def compute_laser_hit(actual_angles):
    """
    Compute where the laser beam hits the wall (WALL_X plane).

    Uses _robot.forward_kinematics() for the true end-effector pose, then fires
    a ray from (ee_x, ee_y, ee_z + LASER_HEIGHT_ABOVE_EE) in the direction the
    end-effector is pointing (XY-plane yaw), tilted upward by LASER_TILT_DEG.

    Parameters
    ----------
    actual_angles : list[float]
        Live joint angles in radians for [J1, J2, J3].

    Returns
    -------
    hit : tuple(float, float, float) | None
        (x, y, z) in metres of the wall intersection, or None if unreachable.
    on_wall : bool | str
        True if the hit is within [WALL_Y_MIN, WALL_Y_MAX], False if outside,
        or an error string if the computation failed.
    """
    if _robot is None:
        return None, "Robot model not loaded"

    T = _robot.forward_kinematics(actual_angles)
    ee_pos = T[:3, 3]           # metres
    ee_rot = T[:3, :3]

    # End-effector pointing direction in XY plane (yaw angle)
    ee_yaw = math.atan2(ee_rot[1, 0], ee_rot[0, 0])

    # Convert laser parameters to metres for consistency with FK output
    laser_height_m = LASER_HEIGHT_ABOVE_EE / 100.0
    wall_x_m       = WALL_X / 100.0
    wall_y_min_m   = WALL_Y_MIN / 100.0
    wall_y_max_m   = WALL_Y_MAX / 100.0

    laser_origin = np.array([
        ee_pos[0],
        ee_pos[1],
        ee_pos[2] + laser_height_m
    ])

    tilt = math.radians(LASER_TILT_DEG)
    laser_dir = np.array([
        math.cos(tilt) * math.cos(ee_yaw),
        math.cos(tilt) * math.sin(ee_yaw),
        math.sin(tilt)
    ])

    if abs(laser_dir[0]) < 1e-9:
        return None, "Laser is parallel to wall"

    t = (wall_x_m - laser_origin[0]) / laser_dir[0]

    if t < 0:
        return None, "Wall is behind the laser direction"

    hit = laser_origin + t * laser_dir
    on_wall = wall_y_min_m <= hit[1] <= wall_y_max_m
    return (round(hit[0], 4), round(hit[1], 4), round(hit[2], 4)), on_wall


def _format_laser_report(actual_angles, label=""):
    """
    Return a formatted string block for the laser hit at the given joint angles.
    Converts hit coordinates from metres to cm for readability.
    """
    hit, on_wall = compute_laser_hit(actual_angles)
    lines = []
    if label:
        lines.append(f"  --- Laser hit {label} ---")
    if hit is None:
        lines.append(f"  Laser wall hit  : N/A ({on_wall})")
    else:
        hit_cm = (hit[0] * 100, hit[1] * 100, hit[2] * 100)
        status = "ON WALL ✓" if on_wall else "OFF WALL ✗"
        lines.append(f"  Laser hits wall : X={hit_cm[0]:+7.2f} cm  "
                     f"Y={hit_cm[1]:+7.2f} cm  Z={hit_cm[2]:+7.2f} cm  [{status}]")
    return "\n".join(lines)


# ── FK + laser report (used by stop_code and natural end) ─────────────────────
def _print_fk_now():
    if _robot is None or _actuator_controller is None:
        print("[FK] Controller not ready yet.")
        return

    actual_angles = []
    for jid in [1, 2, 3]:
        actual = _actuator_controller.relative_joint_angle(jid)
        actual_angles.append(actual)

    j1_deg =  math.degrees(actual_angles[0])
    j2_deg = -math.degrees(actual_angles[1])   # display-corrected
    j3_deg =  math.degrees(actual_angles[2])

    print("\n" + "=" * 60)
    print("FROZEN — Joint angles at halt:")
    print(f"  J1 = {j1_deg:+7.3f}°")
    print(f"  J2 = {j2_deg:+7.3f}°  (display-corrected)")
    print(f"  J3 = {j3_deg:+7.3f}°")

    T_actual = _robot.forward_kinematics(actual_angles)
    actual_pos = T_actual[:3, 3]
    actual_rot = T_actual[:3, :3]
    actual_dir = math.atan2(actual_rot[1, 0], actual_rot[0, 0])

    print("\n===== ACTUAL END-EFFECTOR POSITION =====")
    print(f"  x = {actual_pos[0]:.4f} m")
    print(f"  y = {actual_pos[1]:.4f} m")
    print(f"  z = {actual_pos[2]:.4f} m")
    print(f"  orientation (XY plane) = {np.degrees(actual_dir):.2f} deg")
    print(f"  distance from origin   = {np.sqrt(actual_pos[0]**2 + actual_pos[1]**2)*1000:.1f} mm")

    print("\n===== LASER WALL PROJECTION =====")
    print(_format_laser_report(actual_angles))
    print("=" * 60)


# ── Stop callback ──────────────────────────────────────────────────────────────
def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested — storing final joint angles.")
    _print_fk_now()


# ── Main challenge ─────────────────────────────────────────────────────────────
def run_challenge(params):
    global stop_requested, _robot, _actuator_controller
    stop_requested = False

    update_rate = 0.02

    j2_start_deg        = -50.0
    j3_lower_start_deg  = 116.0
    j3_upper_start_deg  = 133.0

    j2_step_deg         = -9.0
    j3_step_deg         = -8.50
    min_j3_upper_deg    = 16.0

    j3_velocity         = 20.0
    wall_distance       = -0.25

    try:
        _robot = Robot.from_config("robot_parameters.json")
        _actuator_controller = ActuatorController("actuator_config.json")
        joint1_id = 1
        joint2_id = 2
        joint3_id = 3

        # ── Setup Joint 2 — POSITION mode ─────────────────────────────────
        print("[SETUP] Configuring J2 — POSITION mode...")
        _actuator_controller.disable_torque(joint2_id)
        time.sleep(0.3)
        _actuator_controller.change_operating_mode(joint2_id, OperatingMode.POSITION)
        time.sleep(0.3)
        _actuator_controller.enable_torque(joint2_id)
        time.sleep(0.3)

        # ── Setup Joint 3 — VELOCITY mode ─────────────────────────────────
        print("[SETUP] Configuring J3 — VELOCITY mode...")
        _actuator_controller.disable_torque(joint3_id)
        time.sleep(0.3)
        _actuator_controller.change_operating_mode(joint3_id, OperatingMode.VELOCITY)
        time.sleep(0.3)
        _actuator_controller.enable_torque(joint3_id)
        time.sleep(0.3)

        # ── Move J2 to initial position ────────────────────────────────────
        j2_deg         = j2_start_deg
        j3_lower_deg   = j3_lower_start_deg
        j3_upper_deg   = j3_upper_start_deg

        print(f"[SETUP] Moving J2 to initial position {j2_deg:.1f}°...")
        j2_init_rad = math.radians(j2_deg)
        raw2 = _actuator_controller.relative_joint_angle_to_raw(joint2_id, -j2_init_rad)
        _actuator_controller.set_position(joint2_id, int(raw2))
        time.sleep(2.0)

        # ── Read and print initial state of ALL joints ─────────────────────
        j1_init = _actuator_controller.relative_joint_angle(joint1_id)
        j2_init = _actuator_controller.relative_joint_angle(joint2_id)
        j3_init = _actuator_controller.relative_joint_angle(joint3_id)
        init_angles = [j1_init, j2_init, j3_init]

        print("\n" + "=" * 60)
        print("[INIT] Live joint positions at start:")
        print(f"  J1 = {math.degrees(j1_init):+7.2f}°")
        print(f"  J2 = {-math.degrees(j2_init):+7.2f}°  (display-corrected)")
        print(f"  J3 = {math.degrees(j3_init):+7.2f}°")
        print(_format_laser_report(init_angles, label="at init"))
        print("=" * 60)

        print(f"\n[SETUP] J2 settled. J3 starting sweep from {j3_upper_deg:.1f}° -> {j3_lower_deg:.1f}°")
        print("=" * 60)
        print("Starting raster scan with auto-shifting rows. Click Stop Code to halt early.")
        print("=" * 60)

        final_j1_deg = None
        final_j2_deg = None
        final_j3_deg = None

        row_idx = 0

        # ── Raster scan loop ───────────────────────────────────────────────
        while not stop_requested:

            if j3_upper_deg <= min_j3_upper_deg:
                print(f"\n[STOP] J3 upper limit ({j3_upper_deg:.1f}°) reached min "
                      f"({min_j3_upper_deg:.1f}°) — stopping scan.")
                break

            row_idx += 1
            j3_lower = math.radians(j3_lower_deg)
            j3_upper = math.radians(j3_upper_deg)

            print(f"\n{'='*60}")
            print(f"[ROW {row_idx}] J2 -> {j2_deg:.1f}° | J3 sweep: {j3_upper_deg:.1f}° -> {j3_lower_deg:.1f}°")
            print(f"{'='*60}")

            j2_rad = math.radians(j2_deg)
            raw2 = _actuator_controller.relative_joint_angle_to_raw(joint2_id, -j2_rad)
            _actuator_controller.set_position(joint2_id, int(raw2))
            time.sleep(1.5)

            j1_pos = _actuator_controller.relative_joint_angle(joint1_id)
            j2_actual = _actuator_controller.relative_joint_angle(joint2_id)
            j3_pos = _actuator_controller.relative_joint_angle(joint3_id)
            print(f"[ROW {row_idx}] J2 settled:")
            print(f"  J1 = {math.degrees(j1_pos):+7.2f}°")
            print(f"  J2 = {-math.degrees(j2_actual):+7.2f}°  (target: {j2_deg:.1f}°)")
            print(f"  J3 = {math.degrees(j3_pos):+7.2f}°")

            # ── Move J3 to upper limit before sweep ────────────────────────
            j3_pos = _actuator_controller.relative_joint_angle(joint3_id)
            if j3_pos < j3_upper - math.radians(2.0):
                print(f"\n[ROW {row_idx}] Positioning J3 to upper limit {j3_upper_deg:.1f}°...")
                _actuator_controller.set_velocity(joint3_id, j3_velocity)
                while True:
                    if stop_requested:
                        break
                    j1_pos  = _actuator_controller.relative_joint_angle(joint1_id)
                    j2_pos  = _actuator_controller.relative_joint_angle(joint2_id)
                    j3_pos  = _actuator_controller.relative_joint_angle(joint3_id)
                    angles  = [j1_pos, j2_pos, j3_pos]
                    hit, on_wall = compute_laser_hit(angles)
                    laser_str = "N/A"
                    if hit:
                        hx, hy, hz = hit[0]*100, hit[1]*100, hit[2]*100
                        laser_str = f"X={hx:+6.1f} Y={hy:+6.1f} Z={hz:+6.1f} cm"

                    print(f"  [POSITIONING] J1={math.degrees(j1_pos):+7.2f}° | "
                          f"J2={-math.degrees(j2_pos):+7.2f}° | "
                          f"J3={math.degrees(j3_pos):+7.2f}°  "
                          f"-> target upper={j3_upper_deg:.1f}°  "
                          f"| Laser: {laser_str}")
                    if j3_pos >= j3_upper:
                        break
                    time.sleep(update_rate)
                _actuator_controller.set_velocity(joint3_id, 0.0)
                time.sleep(0.3)

            if stop_requested:
                break

            # ── Sweep J3 downward ──────────────────────────────────────────
            print(f"\n[ROW {row_idx}] Sweeping J3 downward...")
            print(f"  {'TIME':>6} | {'J1':>8} | {'J2':>8} | {'J3':>8} | {'LASER HIT (cm)':>38} | RANGE")
            print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*38}-+-{'-'*20}")
            _actuator_controller.set_velocity(joint3_id, -j3_velocity)

            sweep_start = time.time()
            while not stop_requested:
                elapsed = time.time() - sweep_start
                j1_pos  = _actuator_controller.relative_joint_angle(joint1_id)
                j2_pos  = _actuator_controller.relative_joint_angle(joint2_id)
                j3_pos  = _actuator_controller.relative_joint_angle(joint3_id)
                angles  = [j1_pos, j2_pos, j3_pos]

                hit, on_wall = compute_laser_hit(angles)
                if hit:
                    hx, hy, hz = hit[0]*100, hit[1]*100, hit[2]*100
                    wall_flag  = "✓" if on_wall else "✗"
                    laser_col  = f"X={hx:+6.1f} Y={hy:+6.1f} Z={hz:+5.1f} [{wall_flag}]"
                else:
                    laser_col = f"{'N/A':>38}"

                print(f"  {elapsed:6.2f}s | "
                      f"J1={math.degrees(j1_pos):+7.2f}° | "
                      f"J2={-math.degrees(j2_pos):+7.2f}° | "
                      f"J3={math.degrees(j3_pos):+7.2f}° | "
                      f"{laser_col:>38} | "
                      f"[{j3_lower_deg:.1f}° -> {j3_upper_deg:.1f}°]")

                if j3_pos <= j3_lower:
                    _actuator_controller.set_velocity(joint3_id, 0.0)
                    print(f"\n[ROW {row_idx}] J3 reached lower limit {j3_lower_deg:.1f}° — moving to next row.")
                    break

                time.sleep(update_rate)

            if stop_requested:
                break

            j2_deg       += j2_step_deg
            j3_lower_deg += j3_step_deg
            j3_upper_deg += j3_step_deg
            print(f"\n[ROW {row_idx}] Row done — next row: J2={j2_deg:.1f}°, "
                  f"J3=[{j3_lower_deg:.1f}°, {j3_upper_deg:.1f}°]")

        # ── Stop Code triggered (or natural stop) ──────────────────────────
        _actuator_controller.set_velocity(joint3_id, 0.0)
        time.sleep(0.1)

        j1_final = _actuator_controller.relative_joint_angle(joint1_id)
        j2_final = _actuator_controller.relative_joint_angle(joint2_id)
        j3_final = _actuator_controller.relative_joint_angle(joint3_id)

        final_j1_deg = math.degrees(j1_final)
        final_j2_deg = -math.degrees(j2_final)   # display-corrected
        final_j3_deg = math.degrees(j3_final)

        print("\n" + "=" * 60)
        print("SCAN ENDED — Final joint angles:")
        print(f"  J1 = {final_j1_deg:+7.3f}°")
        print(f"  J2 = {final_j2_deg:+7.3f}°  (display-corrected)")
        print(f"  J3 = {final_j3_deg:+7.3f}°")
        print("=" * 60)

        # FK + laser only on natural end; stop_code() already fired it instantly
        if not stop_requested:
            _print_fk_now()

        return {
            "completed_successfully": not stop_requested,
            "final_joint1_deg":       final_j1_deg,
            "final_joint2_deg":       final_j2_deg,
            "final_joint3_deg":       final_j3_deg,
            "rows_completed":         row_idx,
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

    finally:
        if _actuator_controller is not None:
            try:
                _actuator_controller.set_velocity(joint3_id, 0.0)
                time.sleep(0.2)
                _actuator_controller.disable_torque(joint1_id)
                _actuator_controller.disable_torque(joint2_id)
                _actuator_controller.disable_torque(joint3_id)
                print("Cleanup done — torque disabled on all joints.")
            except:
                pass


if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Challenge results: {results}")