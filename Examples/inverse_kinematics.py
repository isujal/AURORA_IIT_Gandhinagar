import numpy as np
import math
import time
from robot_core.robot_model import Robot
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True


def wait_until_settled(actuator_controller, joint_ids, targets_rad,
                       threshold_deg=3.0, timeout=8.0):
    """Poll joints until all within threshold of target or timeout."""
    start  = time.time()
    errors = [999] * len(joint_ids)
    while (time.time() - start) < timeout:
        for i, (jid, target) in enumerate(zip(joint_ids, targets_rad)):
            actual    = actuator_controller.relative_joint_angle(jid)
            errors[i] = abs(math.degrees(actual - target))
        if all(e < threshold_deg for e in errors):
            print(f"  Settled in {time.time()-start:.1f}s  "
                  f"max_error={max(errors):.1f}°")
            return True
        time.sleep(0.05)
    print(f"  WARNING: Timeout {timeout}s — "
          f"errors={[f'{e:.1f}°' for e in errors]}")
    return False


def move_to_ik(actuator_controller, robot, joint_ids,
               desired_pos, desired_orientation,
               threshold_deg=3.0, timeout=8.0):
    """
    Solves inverse kinematics for desired EE pose and moves the arm there.

    Args:
        actuator_controller  : ActuatorController instance
        robot                : Robot instance
        joint_ids            : e.g. [1, 2, 3]
        desired_pos          : [x, y, z] in metres e.g. [0.075, 0.2, 0]
        desired_orientation  : 3x3 rotation matrix (np.array)
        threshold_deg        : settle accuracy in degrees
        timeout              : max wait per pose in seconds

    Returns:
        calculated_joint_angles — list of joint angles in radians
    """
    print("\n── Inverse Kinematics ───────────────────────────────────────")
    print(f"  Target position    : x={desired_pos[0]*1000:.1f}mm  "
          f"y={desired_pos[1]*1000:.1f}mm  "
          f"z={desired_pos[2]*1000:.1f}mm")
    print(f"  Target orientation :\n{np.round(desired_orientation, 4)}")

    # ── Solve IK ──────────────────────────────────────────────────────────────
    calculated_joint_angles = robot.inverse_kinematics(
        desired_pos, desired_orientation
    )

    print("\n── IK Solution (joint angles) ───────────────────────────────")
    for i, (jid, angle) in enumerate(zip(joint_ids, calculated_joint_angles)):
        print(f"  J{jid}: {math.degrees(angle):+.2f}°  ({angle:.4f} rad)")

    # ── Convert to raw and send position commands ─────────────────────────────
    print("\n── Sending position commands ────────────────────────────────")
    for jid, angle in zip(joint_ids, calculated_joint_angles):
        raw = actuator_controller.relative_joint_angle_to_raw(jid, angle)
        actuator_controller.set_position(jid, int(raw))
        print(f"  J{jid}: {math.degrees(angle):+.2f}°  raw={int(raw)}")

    # ── Wait for joints to settle ─────────────────────────────────────────────
    print(f"\nWaiting for joints to settle "
          f"(threshold={threshold_deg}°, timeout={timeout}s)...")
    wait_until_settled(actuator_controller, joint_ids,
                       calculated_joint_angles, threshold_deg, timeout)

    # ── Read back actual angles and verify ────────────────────────────────────
    print("\n── Actual vs IK Target ──────────────────────────────────────")
    actual_angles = []
    for jid, ik_angle in zip(joint_ids, calculated_joint_angles):
        actual = actuator_controller.relative_joint_angle(jid)
        actual_angles.append(actual)
        error  = math.degrees(actual - ik_angle)
        status = "✅" if abs(error) < threshold_deg else "⚠️ "
        print(f"  {status} J{jid}: ik={math.degrees(ik_angle):+6.2f}°  "
              f"actual={math.degrees(actual):+6.2f}°  "
              f"error={error:+5.2f}°")

    # ── FK verification — check where arm actually ended up ───────────────────
    T        = robot.forward_kinematics(actual_angles)
    actual_p = T[:3, 3]
    pos_err  = np.linalg.norm(np.array(desired_pos) - actual_p) * 1000

    print("\n── FK Verification ──────────────────────────────────────────")
    print(f"  Target EE : ({desired_pos[0]*1000:+.1f}, "
          f"{desired_pos[1]*1000:+.1f}, "
          f"{desired_pos[2]*1000:+.1f}) mm")
    print(f"  Actual EE : ({actual_p[0]*1000:+.1f}, "
          f"{actual_p[1]*1000:+.1f}, "
          f"{actual_p[2]*1000:+.1f}) mm")
    print(f"  Position error: {pos_err:.2f} mm  "
          f"{'✅ < 5mm' if pos_err < 5 else '⚠️  > 5mm'}")

    return calculated_joint_angles


def run_challenge(actuator_controller):
    global stop_requested
    stop_requested = False

    joint_ids = [1, 2, 3]

    try:
        robot = Robot.from_config("robot_parameters.json")

        # ── POSITION mode setup ───────────────────────────────────────────────
        for jid in joint_ids:
            actuator_controller.disable_torque(jid)
        time.sleep(0.1)

        for jid in joint_ids:
            actuator_controller.change_operating_mode(jid, OperatingMode.POSITION)
        time.sleep(0.1)

        for jid in joint_ids:
            success = actuator_controller.enable_torque(jid)
            if not success:
                raise Exception(f"Failed to enable torque on joint {jid}")
        time.sleep(0.1)

        # ── Define IK targets — edit these ───────────────────────────────────
        # Position in metres [x, y, z]
        # Orientation as 3x3 rotation matrix
        ik_targets = [
            {
                "pos": [0.075, 0.2, 0],
                "ori": np.array([
                    [ 0, -1, 0],
                    [ 1,  0, 0],
                    [ 0,  0, 1]
                ])
            },
            # ── Add more targets here ─────────────────────────────────────────
            # {
            #     "pos": [0.1, 0.15, 0],
            #     "ori": np.array([
            #         [1, 0, 0],
            #         [0, 1, 0],
            #         [0, 0, 1]
            #     ])
            # },
        ]

        results = []
        for i, target in enumerate(ik_targets):
            if stop_requested:
                print("Stop requested.")
                break

            print(f"\n{'='*55}")
            print(f"IK TARGET {i+1}")
            print(f"{'='*55}")

            angles = move_to_ik(
                actuator_controller, robot, joint_ids,
                desired_pos         = target["pos"],
                desired_orientation = target["ori"],
                threshold_deg       = 3.0,
                timeout             = 8.0
            )
            results.append({
                "target": i + 1,
                "pos": target["pos"],
                "joint_angles_deg": [math.degrees(a) for a in angles]
            })

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print("SUMMARY")
        print(f"{'='*55}")
        for r in results:
            p = r["pos"]
            a = r["joint_angles_deg"]
            print(f"  Target {r['target']} "
                  f"pos=({p[0]*1000:.0f},{p[1]*1000:.0f},{p[2]*1000:.0f})mm → "
                  f"J1={a[0]:+.1f}° J2={a[1]:+.1f}° J3={a[2]:+.1f}°")

        return {"completed_successfully": True, "num_targets": len(results)}

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        try:
            time.sleep(1.0)
            for jid in joint_ids:
                actuator_controller.disable_torque(jid)
            print("\nCleanup done — torque disabled.")
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    actuator_controller = ActuatorController("actuator_config.json")
    try:
        run_challenge(actuator_controller)
    finally:
        actuator_controller.close()