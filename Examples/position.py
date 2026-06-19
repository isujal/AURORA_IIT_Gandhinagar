import numpy as np
import time
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode


def run_challenge(actuator_controller):
    joint1_id = 1

    try:
        actuator_controller.disable_torque(joint1_id)
        time.sleep(0.1)
        actuator_controller.change_operating_mode(joint1_id, OperatingMode.POSITION)
        time.sleep(0.1)

        success = actuator_controller.enable_torque(joint1_id)
        time.sleep(0.1)
        if not success:
            raise Exception("Failed to enable torque on joint 1")

        # ── Send position command ─────────────────────────────────────────────
        angle     = np.radians(135)
        angle_raw = actuator_controller.relative_joint_angle_to_raw(joint1_id, angle)
        actuator_controller.set_position(joint1_id, int(angle_raw))
        print(f"Command sent: {np.degrees(angle):.2f}°  raw={int(angle_raw)}")

        # ── Wait for motor to reach target ────────────────────────────────────
        time.sleep(2.0)

        # ── Read back actual position ─────────────────────────────────────────
        a1   = actuator_controller.relative_joint_angle(joint1_id)
        pos1 = actuator_controller.get_position(joint1_id)

        print(f"Actual angle : {a1:.4f} rad  ({np.degrees(a1):.2f}°)")
        print(f"Raw position : {pos1}")
        print(f"Error        : {np.degrees(abs(angle - a1)):.2f}°")

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        try:
            actuator_controller.disable_torque(joint1_id)
            print("Torque disabled.")
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    from esp_bridge import ESPActuatorController as ActuatorController

    actuator_controller = ActuatorController("actuator_config.json")
    try:
        run_challenge(actuator_controller)
    finally:
        actuator_controller.close()