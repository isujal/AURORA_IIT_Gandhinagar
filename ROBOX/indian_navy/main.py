import numpy as np
import math
import time
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode


"""
just tell me the sample code, where my robot's joint 3 is running on the velocity mode whose sample is given by me in the uploaded 
code, then the joint 2 should move in the position mode whose code is given as well, i want the joint 3 stable at 0 degrees. i just want you to 
write a code such that the joints behave in their modes which i gave you in such a way that the joint 2 will be 50 degrees at the initialised state and joint 3 will start from 133 degrees and go to 116 degrees and then joint 2 will increment from 50 to 60
after it is at 60 the joint 3 will move from 124 degrees to 107 degrees, its just that the minimum and maximum limits are changing when joint 2 is increemnted, ihave pasted the challenge as well, i want you to develeop the cpode accordingly
the difference between the minimum and maximum limit of the joint 3 should be an average of 19 or 20 like that. just tell me whether my approach is right for implementing this challenge, is not then suggest me code, and generate me a code for my approch as well
""" 
stop_requested = False

def stop_code():
    """Called by GUI when user presses Stop."""
    global stop_requested
    stop_requested = True


def run_challenge(actuator_controller):
    global stop_requested
    stop_requested = False

    joint3_id = 4  # Update this to your actual Joint 3 servo ID
    
    start_angle_deg = 72.0
    end_angle_deg   = 54.0
    joint3_speed    = 2   # rad/s — going negative (decreasing angle)
    update_rate     = 0.01  # seconds

    start_angle = math.radians(start_angle_deg)
    end_angle   = math.radians(end_angle_deg)

    try:
        # ── Set VELOCITY mode ──────────────────────────────────────────
        actuator_controller.disable_torque(joint3_id)
        time.sleep(0.1)
        actuator_controller.change_operating_mode(joint3_id, OperatingMode.VELOCITY)
        time.sleep(0.1)
        success = actuator_controller.enable_torque(joint3_id)
        time.sleep(0.1)
        if not success:
            raise Exception("Failed to enable torque on joint 3")

        loop_count = 0

        while not stop_requested:
            loop_count += 1
            print(f"\n── Loop {loop_count} ──────────────────────────────")

            # ── PHASE 1: Move from 72° → 54° (negative direction) ─────
            current_pos = actuator_controller.relative_joint_angle(joint3_id)
            print(f"[Phase 1] Current: {np.degrees(current_pos):.2f}° → Target: {end_angle_deg}°")

            if current_pos > end_angle:
                velocity_raw = actuator_controller.convert_velocity_to_raw(joint3_id, -joint3_speed)
                actuator_controller.set_velocity(joint3_id, velocity_raw)

                while not stop_requested:
                    current_pos = actuator_controller.relative_joint_angle(joint3_id)
                    if current_pos <= end_angle:
                        print(f"[Phase 1] Reached {np.degrees(current_pos):.2f}°")
                        break
                    time.sleep(update_rate)

            if stop_requested:
                break

            actuator_controller.set_velocity(joint3_id, 0)
            time.sleep(0.05)

            # ── PHASE 2: Move from 54° → 72° (positive direction) ─────
            current_pos = actuator_controller.relative_joint_angle(joint3_id)
            print(f"[Phase 2] Current: {np.degrees(current_pos):.2f}° → Target: {start_angle_deg}°")

            if current_pos < start_angle:
                velocity_raw = actuator_controller.convert_velocity_to_raw(joint3_id, +joint3_speed)
                actuator_controller.set_velocity(joint3_id, velocity_raw)

                while not stop_requested:
                    current_pos = actuator_controller.relative_joint_angle(joint3_id)
                    if current_pos >= start_angle:
                        print(f"[Phase 2] Reached {np.degrees(current_pos):.2f}°")
                        break
                    time.sleep(update_rate)

            if stop_requested:
                break

            actuator_controller.set_velocity(joint3_id, 0)
            time.sleep(0.05)

        print(f"\nStop requested — exiting after {loop_count} loop(s).")

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        try:
            actuator_controller.set_velocity(joint3_id, 0)
            time.sleep(0.02)
            actuator_controller.disable_torque(joint3_id)
            print("Cleanup done — torque disabled.")
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    from esp_bridge import ESPActuatorController as ActuatorController

    actuator_controller = ActuatorController("actuator_config.json")
    try:
        run_challenge(actuator_controller)
    finally:
        actuator_controller.close()