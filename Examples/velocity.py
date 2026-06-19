import time
import math
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

stop_requested = False


def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested - execution will terminate at next loop iteration.")


def run_challenge(params):
    global stop_requested
    stop_requested = False

    duration = 20.0      # total runtime in seconds
    update_rate = 0.02   # control loop update rate in seconds

    try:
        actuator_controller = ActuatorController("actuator_config.json")
        joint_id = 3

        # ── Set VELOCITY mode ─────────────────────────────────────────────
        actuator_controller.disable_torque(joint_id)
        time.sleep(0.3)
        actuator_controller.change_operating_mode(joint_id, OperatingMode.VELOCITY)
        time.sleep(0.3)
        actuator_controller.enable_torque(joint_id)
        time.sleep(0.3)

        # ── Sweep settings (same units/values as your working J3 logic) ────
        max_velocity = 45.0                     # RPM units — ~1.5 rad/s actual
        lower_limit = math.radians(0.0)        # below current start pos of -42°
        upper_limit = math.radians(120.0)         # enough travel to trigger reversal

        # Start moving the correct direction based on current position
        init_pos = actuator_controller.relative_joint_angle(joint_id)
        print(f"[Joint] Init position: {math.degrees(init_pos):.1f}°  "
              f"limits: {math.degrees(lower_limit):.1f}° to {math.degrees(upper_limit):.1f}°")

        if init_pos <= lower_limit:
            current_vel = max_velocity
            print("[Joint] Started below lower limit — forcing upward.")
        elif init_pos >= upper_limit:
            current_vel = -max_velocity
            print("[Joint] Started above upper limit — forcing downward.")
        else:
            current_vel = max_velocity

        print(f"Sweeping between {math.degrees(lower_limit):.0f}° and "
              f"{math.degrees(upper_limit):.0f}° for {duration}s...")

        start_time = time.time()

        while (time.time() - start_time) < duration:
            if stop_requested:
                print("Stop requested - terminating execution")
                break

            pos = actuator_controller.relative_joint_angle(joint_id)

            if pos >= upper_limit and current_vel > 0:
                current_vel = -max_velocity
                print(f"  Upper boundary hit at {math.degrees(pos):.1f}° — reversing.")
            elif pos <= lower_limit and current_vel < 0:
                current_vel = max_velocity
                print(f"  Lower boundary hit at {math.degrees(pos):.1f}° — reversing.")

            # Safety clamp, same as your working code
            current_vel = max(-max_velocity, min(max_velocity, current_vel))
            actuator_controller.set_velocity(joint_id, current_vel)

            time.sleep(update_rate)

        if stop_requested:
            print("Execution was terminated by user request.")

        final_pos = actuator_controller.relative_joint_angle(joint_id)
        return {
            "completed_successfully": not stop_requested,
            "final_position_deg": math.degrees(final_pos),
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

    finally:
        if 'actuator_controller' in locals():
            try:
                actuator_controller.set_velocity(joint_id, 0.0)
                time.sleep(0.2)
                actuator_controller.disable_torque(joint_id)
                print("Cleanup done — torque disabled.")
            except:
                pass


if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Challenge results: {results}")