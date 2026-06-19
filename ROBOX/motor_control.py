import numpy as np
import time
import math
from robot_core.robot_model import Robot
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

    update_rate = 0.05

    try:
        actuator_controller = ActuatorController("actuator_config.json")
        joint1_id = 1


        target_positions_deg = [30.0, 70.0, 110.0, 150.0, 30.0]
        target_positions_rad = [math.radians(d) for d in target_positions_deg]

        print("=== Stage 1.1: Position Control Drill ===")

        actuator_controller.disable_torque(joint1_id)
        time.sleep(0.3) 
        actuator_controller.change_operating_mode(joint1_id, OperatingMode.POSITION)
        time.sleep(0.3) 
        actuator_controller.enable_torque(joint1_id)
        time.sleep(0.3)  

        for i, angle_rad in enumerate(target_positions_rad):
            if stop_requested:
                break

            raw = actuator_controller.relative_joint_angle_to_raw(joint1_id, angle_rad)
            # raw = actuator_controller.relative_joint_angle_to_raw(2, 0)
            # raw = actuator_controller.relative_joint_angle_to_raw(3, 0)
            actuator_controller.set_position(joint1_id, int(raw))
            # actuator_controller.set_position(2, int(raw))
            # actuator_controller.set_position(3, int(raw))
            print(f"Moving to {target_positions_deg[i]:.1f}° ({angle_rad:.4f} rad)...")

            wait_start = time.time()
            while time.time() - wait_start < 5.0:
                if stop_requested:
                    print("Stop requested during position hold.")
                    break
                time.sleep(update_rate)

            actual = actuator_controller.relative_joint_angle(joint1_id)
            print(f"  Target: {math.degrees(angle_rad):.2f}°  |  Actual: {math.degrees(actual):.2f}°  |  Error: {math.degrees(actual - angle_rad):.2f}°")


        print("\n=== Stage 1.2: Velocity Control Drill ===")

        actuator_controller.disable_torque(joint1_id)
        time.sleep(0.5)  
        actuator_controller.change_operating_mode(joint1_id, OperatingMode.VELOCITY)
        time.sleep(0.5)  
        actuator_controller.enable_torque(joint1_id)
        time.sleep(0.3)

        actuator_controller.set_velocity(joint1_id, 0.0)
        time.sleep(0.5)
        print("Mode switched to VELOCITY — starting motion...")

        max_velocity = 40.0   
        current_vel  = max_velocity
        lower_limit  = math.radians(30.0)
        upper_limit  = math.radians(90.0)
        vel_duration = 30.0

        print(f"Commanding {max_velocity} rad/s within [{math.degrees(lower_limit):.0f}°, {math.degrees(upper_limit):.0f}°]")

        vel_start = time.time()
        while time.time() - vel_start < vel_duration:
            if stop_requested:
                print("Stop requested during velocity control.")
                break

            current_pos = actuator_controller.relative_joint_angle(joint1_id)

            if current_pos >= upper_limit and current_vel > 0:
                current_vel = -max_velocity
                print(f"  Upper boundary hit at {math.degrees(current_pos):.1f}° — reversing.")
            elif current_pos <= lower_limit and current_vel < 0:
                current_vel = max_velocity
                print(f"  Lower boundary hit at {math.degrees(current_pos):.1f}° — reversing.")

            result = actuator_controller.set_velocity(joint1_id, current_vel)
            print(f"  pos={math.degrees(current_pos):.1f}°  vel_cmd={current_vel:.2f}  result={result}")

            time.sleep(update_rate)

        # fOR STOPPING IT CLEANLY
        actuator_controller.set_velocity(joint1_id, 0.0)
        time.sleep(0.3)
        print("Velocity phase complete — motor stopped.")

        final_pos = actuator_controller.relative_joint_angle(joint1_id)
        print(f"\nFinal Joint 1 position: {math.degrees(final_pos):.2f}°")

        return {
            "completed_successfully": not stop_requested,
            "final_position_deg": math.degrees(final_pos),
            "positions_visited_deg": target_positions_deg
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

    finally:
        if 'actuator_controller' in locals():
            try:
                actuator_controller.set_velocity(joint1_id, 0.0)
                time.sleep(0.2)
                actuator_controller.disable_torque(joint1_id)
                print("Cleanup done — torque disabled.")
            except:
                pass


if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Challenge results: {results}")