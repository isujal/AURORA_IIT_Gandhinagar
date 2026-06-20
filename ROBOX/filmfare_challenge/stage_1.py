import numpy as np
import time
import math
from robot_core.robot_model import Robot
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

stop_requested = False
_robot = None
_actuator_controller = None

TARGET_X = 0.0
TARGET_Y = 0.23

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested — halting arc motion.")

def run_challenge(params):
    global stop_requested, _robot, _actuator_controller
    stop_requested = False

    joint1_id = 1
    joint2_id = 2
    joint3_id = 3

    # ── Arc parameters ─────────────────────────────────────────────────────────
    # End-effector orbits the target at this radius.
    # Sweep 135° → 45° (upper-left to upper-right above target) = 90° total arc.
    ARC_RADIUS  = 0.15        # metres — tune to your robot's workspace
    N_STEPS     = 90          # one waypoint per degree
    STEP_DT     = 0.08        # seconds between waypoints
    THETA_START = math.radians(135)
    THETA_END   = math.radians(45)

    thetas = np.linspace(THETA_START, THETA_END, N_STEPS)

    try:
        _robot = Robot.from_config("robot_parameters.json")
        _actuator_controller = ActuatorController("actuator_config.json")

        # ── Set all joints to POSITION mode ────────────────────────────────────
        for jid in [joint1_id, joint2_id, joint3_id]:
            _actuator_controller.disable_torque(jid)
            time.sleep(0.2)
            _actuator_controller.change_operating_mode(jid, OperatingMode.POSITION)
            time.sleep(0.2)
            _actuator_controller.enable_torque(jid)
            time.sleep(0.2)

        print(f"Starting 90° arc around target ({TARGET_X}, {TARGET_Y})")
        print(f"Radius: {ARC_RADIUS} m | Steps: {N_STEPS} | dt: {STEP_DT}s")
        print(f"Arc: {math.degrees(THETA_START):.0f}° → {math.degrees(THETA_END):.0f}°")
        print("=" * 60)

        for i, theta in enumerate(thetas):
            if stop_requested:
                print("Stop requested — halting arc.")
                break

            # ── End-effector position on the arc ───────────────────────────────
            ee_x = TARGET_X + ARC_RADIUS * math.cos(theta)
            ee_y = TARGET_Y + ARC_RADIUS * math.sin(theta)

            # ── phi: angle pointing FROM end-effector TOWARD target ────────────
            phi = math.atan2(TARGET_Y - ee_y, TARGET_X - ee_x)

            # ── Build 4x4 transformation matrix for IK ─────────────────────────
            T = np.eye(4)
            T[0, 3] = ee_x
            T[1, 3] = ee_y
            T[0, 0] =  math.cos(phi)
            T[0, 1] = -math.sin(phi)
            T[1, 0] =  math.sin(phi)
            T[1, 1] =  math.cos(phi)

            position    = T[:3, 3]
            orientation = T[:3, :3]

            # Initial guess: point joints toward the target direction
            initial_guess = [phi, 0.0, 0.0]

            # ── Solve IK ────────────────────────────────────────────────────────
            try:
                q = _robot.inverse_kinematics(
                    position,
                    orientation,
                    initial_guess=initial_guess
                )
            except Exception as ik_err:
                print(f"  [step {i+1}] IK failed: {ik_err} — skipping")
                continue

            if q is None:
                print(f"  [step {i+1}] IK returned None at "
                      f"({ee_x:.4f}, {ee_y:.4f}) phi={math.degrees(phi):.1f}° — skipping")
                continue

            theta1, theta2, theta3 = q

            # ── Send to motors ──────────────────────────────────────────────────
            raw1 = _actuator_controller.relative_joint_angle_to_raw(joint1_id, theta1)
            raw2 = _actuator_controller.relative_joint_angle_to_raw(joint2_id, theta2)
            raw3 = _actuator_controller.relative_joint_angle_to_raw(joint3_id, theta3)

            _actuator_controller.set_position(joint1_id, int(raw1))
            _actuator_controller.set_position(joint2_id, int(raw2))
            _actuator_controller.set_position(joint3_id, int(raw3))

            print(f"  [{i+1:02d}/{N_STEPS}] "
                  f"θ_arc={math.degrees(theta):+7.1f}° | "
                  f"EE=({ee_x:+.4f}, {ee_y:+.4f}) | "
                  f"phi={math.degrees(phi):+6.1f}° | "
                  f"J1={math.degrees(theta1):+7.2f}° "
                  f"J2={math.degrees(theta2):+7.2f}° "
                  f"J3={math.degrees(theta3):+7.2f}°")

            time.sleep(STEP_DT)

        # ── Read final actual pose ──────────────────────────────────────────────
        j1_final = _actuator_controller.relative_joint_angle(joint1_id)
        j2_final = _actuator_controller.relative_joint_angle(joint2_id)
        j3_final = _actuator_controller.relative_joint_angle(joint3_id)

        T_final = _robot.forward_kinematics([j1_final, j2_final, j3_final])
        final_pos = T_final[:3, 3]

        print("\n" + "=" * 60)
        print("ARC COMPLETE — Final state:")
        print(f"  J1={math.degrees(j1_final):+7.2f}°  "
              f"J2={math.degrees(j2_final):+7.2f}°  "
              f"J3={math.degrees(j3_final):+7.2f}°")
        print(f"  Final EE position: x={final_pos[0]:.4f}  y={final_pos[1]:.4f}")
        print(f"  Target was:        x={TARGET_X:.4f}  y={TARGET_Y:.4f}")
        print("=" * 60)

        return {
            "completed_successfully": not stop_requested,
            "final_j1_deg": math.degrees(j1_final),
            "final_j2_deg": math.degrees(j2_final),
            "final_j3_deg": math.degrees(j3_final),
            "steps_executed": N_STEPS,
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

    finally:
        if _actuator_controller is not None:
            try:
                for jid in [joint1_id, joint2_id, joint3_id]:
                    _actuator_controller.disable_torque(jid)
                print("Cleanup done — torque disabled on all joints.")
            except:
                pass

if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Challenge results: {results}")