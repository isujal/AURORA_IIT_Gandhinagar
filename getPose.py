import math
import time
from esp_bridge import ESPActuatorController as ActuatorController

def run_challenge(params):
    try:
        actuator_controller = ActuatorController("actuator_config.json")

        print("Live monitoring — Ctrl+C to stop\n")
        while True:
            positions = []
            for jid in [1, 2, 3]:
                angle_deg = math.degrees(actuator_controller.relative_joint_angle(jid))
                positions.append(angle_deg)
            print(f"\r  J1={positions[0]:+7.2f}°  "
                f"J2={positions[1]:+7.2f}°  "
                f"J3={positions[2]:+7.2f}°", end="", flush=True)
            time.sleep(0.1)

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        if 'actuator_controller' in locals():
            actuator_controller.close()
            print("Done.")


if __name__ == "__main__":
    print(run_challenge(params={}))