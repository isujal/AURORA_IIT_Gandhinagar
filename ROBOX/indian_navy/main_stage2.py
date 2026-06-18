import time
import math
import numpy as np
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode
from robot_core.robot_model import Robot

stop_requested = False
_robot = None
_actuator_controller = None


def _print_fk_now():
    if _robot is None or _actuator_controller is None:
        print("[FK] Controller not ready yet.")
        return

    # Read live joint angles exactly as forward_kinematics.py does
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

    # FK on actual angles — same as forward_kinematics.py
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
    print("=" * 60)


def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested — storing final joint angles.")
    _print_fk_now()


def run_challenge(params):
    global stop_requested, _robot, _actuator_controller
    stop_requested = False

    update_rate = 0.02

    j2_start_deg = -50.0
    j3_lower_start_deg = 116.0
    j3_upper_start_deg = 133.0

    j2_step_deg = -9.0
    j3_step_deg = -8.50
    min_j3_upper_deg = 16.0

    j3_velocity = 20.0
    wall_distance = -0.25

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
        j2_deg = j2_start_deg
        j3_lower_deg = j3_lower_start_deg
        j3_upper_deg = j3_upper_start_deg

        print(f"[SETUP] Moving J2 to initial position {j2_deg:.1f}°...")
        j2_init_rad = math.radians(j2_deg)
        raw2 = _actuator_controller.relative_joint_angle_to_raw(joint2_id, -j2_init_rad)
        _actuator_controller.set_position(joint2_id, int(raw2))
        time.sleep(2.0)

        # ── Read and print initial state of ALL joints ─────────────────────
        j1_init = _actuator_controller.relative_joint_angle(joint1_id)
        j2_init = _actuator_controller.relative_joint_angle(joint2_id)
        j3_init = _actuator_controller.relative_joint_angle(joint3_id)
        print("\n" + "=" * 60)
        print("[INIT] Live joint positions at start:")
        print(f"  J1 = {math.degrees(j1_init):+7.2f}°")
        print(f"  J2 = {-math.degrees(j2_init):+7.2f}°  (display-corrected)")
        print(f"  J3 = {math.degrees(j3_init):+7.2f}°")
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
                    j1_pos = _actuator_controller.relative_joint_angle(joint1_id)
                    j2_pos = _actuator_controller.relative_joint_angle(joint2_id)
                    j3_pos = _actuator_controller.relative_joint_angle(joint3_id)
                    print(f"  [POSITIONING] J1={math.degrees(j1_pos):+7.2f}° | "
                          f"J2={-math.degrees(j2_pos):+7.2f}° | "
                          f"J3={math.degrees(j3_pos):+7.2f}°  -> target upper={j3_upper_deg:.1f}°")
                    if j3_pos >= j3_upper:
                        break
                    time.sleep(update_rate)
                _actuator_controller.set_velocity(joint3_id, 0.0)
                time.sleep(0.3)

            if stop_requested:
                break

            # ── Sweep J3 downward ──────────────────────────────────────────
            print(f"\n[ROW {row_idx}] Sweeping J3 downward...")
            print(f"  {'TIME':>6} | {'J1':>8} | {'J2':>8} | {'J3':>8} | SWEEP RANGE")
            print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}")
            _actuator_controller.set_velocity(joint3_id, -j3_velocity)

            sweep_start = time.time()
            while not stop_requested:
                elapsed = time.time() - sweep_start
                j1_pos = _actuator_controller.relative_joint_angle(joint1_id)
                j2_pos = _actuator_controller.relative_joint_angle(joint2_id)
                j3_pos = _actuator_controller.relative_joint_angle(joint3_id)

                print(f"  {elapsed:6.2f}s | "
                      f"J1={math.degrees(j1_pos):+7.2f}° | "
                      f"J2={-math.degrees(j2_pos):+7.2f}° | "
                      f"J3={math.degrees(j3_pos):+7.2f}° | "
                      f"[{j3_lower_deg:.1f}° -> {j3_upper_deg:.1f}°]")

                if j3_pos <= j3_lower:
                    _actuator_controller.set_velocity(joint3_id, 0.0)
                    print(f"\n[ROW {row_idx}] J3 reached lower limit {j3_lower_deg:.1f}° — moving to next row.")
                    break

                time.sleep(update_rate)

            if stop_requested:
                break

            j2_deg += j2_step_deg
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

        # ── FK on final angles — only runs on natural end, not Stop Code ──
        # (Stop Code already printed FK instantly via stop_code())
        if not stop_requested:
            _print_fk_now()

        return {
            "completed_successfully": not stop_requested,
            "final_joint1_deg": final_j1_deg,
            "final_joint2_deg": final_j2_deg,
            "final_joint3_deg": final_j3_deg,
            "rows_completed": row_idx,
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