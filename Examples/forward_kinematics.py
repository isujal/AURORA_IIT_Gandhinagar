""" 
example code for getting forward kinematics of any position

"""

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


def move_and_fk(actuator_controller, robot, joint_ids, target_angles_deg,
                settle_time=2.0):
    """
    Moves all joints to target angles and computes forward kinematics.

    Args:
        actuator_controller : ActuatorController instance
        robot               : Robot instance from robot_parameters.json
        joint_ids           : list of joint IDs e.g. [1, 2, 3]
        target_angles_deg   : list of target angles in DEGREES e.g. [90, 0, 0]
        settle_time         : seconds to wait for motors to reach targets

    Returns:
        T (4x4 transformation matrix) — end-effector pose
    """
    targets_rad = [math.radians(a) for a in target_angles_deg]

    # ── Send position commands ────────────────────────────────────────────────
    print("\n── Sending position commands ──")
    for jid, angle_rad, angle_deg in zip(joint_ids, targets_rad, target_angles_deg):
        raw = actuator_controller.relative_joint_angle_to_raw(jid, angle_rad)
        actuator_controller.set_position(jid, int(raw))
        print(f"  J{jid}: {angle_deg:+.1f}°  raw={int(raw)}")

    # ── Wait for motors to settle ─────────────────────────────────────────────
    print(f"Waiting {settle_time}s for motors to settle...")
    time.sleep(settle_time)

    # ── Read back actual angles ───────────────────────────────────────────────
    actual_angles_rad = []
    print("\n── Actual vs Target ──")
    for jid, target_rad, target_deg in zip(joint_ids, targets_rad, target_angles_deg):
        actual = actuator_controller.relative_joint_angle(jid)
        actual_angles_rad.append(actual)
        error  = math.degrees(actual - target_rad)
        print(f"  J{jid}: target={target_deg:+6.1f}°  "
              f"actual={math.degrees(actual):+6.1f}°  "
              f"error={error:+5.2f}°")

    # ── Forward Kinematics on actual angles ───────────────────────────────────
    T = robot.forward_kinematics(actual_angles_rad)

    pos = T[:3, 3]
    rot = T[:3, :3]
    ee_yaw = math.atan2(rot[1, 0], rot[0, 0])

    print("\n── Forward Kinematics Result ──")
    print(f"  x = {pos[0]*1000:+7.2f} mm")
    print(f"  y = {pos[1]*1000:+7.2f} mm")
    print(f"  z = {pos[2]*1000:+7.2f} mm")
    print(f"  yaw (XY plane) = {math.degrees(ee_yaw):+.2f}°")
    print(f"  dist from origin = {np.linalg.norm(pos[:2])*1000:.2f} mm")
    print(f"\nFull T matrix:\n{np.round(T, 4)}")

    return T


def run_challenge(actuator_controller):
    global stop_requested
    stop_requested = False

    joint_ids = [1, 2, 3]

    try:
        robot = Robot.from_config("robot_parameters.json")

        # ── POSITION mode setup — all joints ──────────────────────────────────
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

        # ── Define poses — add as many as you want ────────────────────────────
        # Each tuple is (J1_deg, J2_deg, J3_deg)
        poses = [
            (90,  0,   0),    # Pose 1 — upright
            (90,  30,  20),   # Pose 2 — original from image
            (45,  -30, 10),   # Pose 3 — custom
            (0,   0,   0),    # Pose 4 — home
        ]

        # ── Move through each pose and compute FK ─────────────────────────────
        results = []
        for i, pose in enumerate(poses):
            if stop_requested:
                print("Stop requested.")
                break

            print(f"\n{'='*50}")
            print(f"POSE {i+1}: J1={pose[0]}°  J2={pose[1]}°  J3={pose[2]}°")
            print(f"{'='*50}")

            T = move_and_fk(
                actuator_controller, robot, joint_ids,
                target_angles_deg=list(pose),
                settle_time=2.0
            )
            results.append({"pose": i+1, "angles_deg": pose, "T": T})

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for r in results:
            pos = r["T"][:3, 3] * 1000
            print(f"Pose {r['pose']} {r['angles_deg']} → "
                  f"EE=({pos[0]:+.1f}, {pos[1]:+.1f}, {pos[2]:+.1f}) mm")

        return {"completed_successfully": True, "num_poses": len(results)}

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