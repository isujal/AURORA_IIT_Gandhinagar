import numpy as np
import time
import math
from robot_core.robot_model import Robot
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

"""
CHALLENGE OVERVIEW:
------------------
Stage 2: Combined Control — Multi-Mode Coordination

1. Joint 1 — VELOCITY mode: oscillate between 60° and 120° at 0.5 rad/s
2. Joint 2 — POSITION mode: sinusoidal motion of ±30° (±0.52 rad)
3. Joint 3 — VELOCITY mode: capped at ±0.3 rad/s safety limit
"""

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
        joint1_id = 1
        joint2_id = 2
        joint3_id = 3


        actuator_controller.disable_torque(joint1_id)
        time.sleep(0.3)
        actuator_controller.change_operating_mode(joint1_id, OperatingMode.VELOCITY)
        time.sleep(0.3)
        actuator_controller.enable_torque(joint1_id)
        time.sleep(0.3)



        actuator_controller.set_velocity(joint1_id, 0.0)
        time.sleep(0.3)    

        j1_max_velocity = 40.0       
        j1_current_vel = j1_max_velocity
        j1_lower_limit = math.radians(0.0)
        j1_upper_limit = math.radians(60.0)

        j1_init_pos = actuator_controller.relative_joint_angle(joint1_id)
        print(f"[J1] Init position: {math.degrees(j1_init_pos):.1f}°")
        if j1_init_pos >= j1_upper_limit:
            j1_current_vel = -j1_max_velocity
            print(f"[J1] Started above upper limit — forcing downward.")
        elif j1_init_pos <= j1_lower_limit:
            j1_current_vel = j1_max_velocity
            print(f"[J1] Started below lower limit — forcing upward.")


        actuator_controller.disable_torque(joint2_id)
        time.sleep(0.3)
        actuator_controller.change_operating_mode(joint2_id, OperatingMode.POSITION)
        time.sleep(0.3)
        actuator_controller.enable_torque(joint2_id)
        time.sleep(0.3)

        j2_amplitude_rad = math.radians(30.0)  # ±0.52 rad
        j2_center_rad = 0.0                    # adjust if your zero isn't centered
        j2_angular_freq = 1.0                  # rad/s — controls oscillation speed

        actuator_controller.disable_torque(joint3_id)
        time.sleep(0.3)
        actuator_controller.change_operating_mode(joint3_id, OperatingMode.VELOCITY)
        time.sleep(0.3)
        actuator_controller.enable_torque(joint3_id)
        time.sleep(0.3)

        j3_max_velocity = 15.0       # RPM units — ~1.5 rad/s actual
        j3_lower_limit = math.radians(-50.0)   # below current start pos of -42°
        j3_upper_limit = math.radians(10.0)    # enough travel to trigger reversal

        j3_init_pos = actuator_controller.relative_joint_angle(joint3_id)
        print(f"[J3] Init position: {math.degrees(j3_init_pos):.1f}°  limits: {math.degrees(j3_lower_limit):.1f}° to {math.degrees(j3_upper_limit):.1f}°")
        if j3_init_pos <= j3_lower_limit:
            j3_current_vel = j3_max_velocity  
            print(f"[J3] Started below lower limit — forcing upward.")
        elif j3_init_pos >= j3_upper_limit:
            j3_current_vel = -j3_max_velocity  
            print(f"[J3] Started above upper limit — forcing downward.")
        else:
            j3_current_vel = j3_max_velocity  

        print("[J1] Running direction test...")
        actuator_controller.set_velocity(joint1_id, 40.0)
        time.sleep(1.0)
        j1_after_pos = actuator_controller.relative_joint_angle(joint1_id)
        print(f"[J1] After +40 for 1s: {math.degrees(j1_after_pos):.1f}°")
        
        actuator_controller.set_velocity(joint1_id, -40.0)
        time.sleep(1.0)
        j1_after_neg = actuator_controller.relative_joint_angle(joint1_id)
        print(f"[J1] After -40 for 1s: {math.degrees(j1_after_neg):.1f}°")

        print("[J3] Running isolation test at 1.0 rad/s for 2s...")
        actuator_controller.set_velocity(joint3_id, 1.0)
        time.sleep(2.0)
        actuator_controller.set_velocity(joint3_id, 0.0)
        j3_test_pos = actuator_controller.relative_joint_angle(joint3_id)
        print(f"[J3] Position after isolation test: {math.degrees(j3_test_pos):.1f}°")
        time.sleep(0.5)

        print("=== Stage 2: Combined Control — Multi-Mode Coordination ===")
        print(f"Running for {duration}s ...")

        start_time = time.time()

        while (time.time() - start_time) < duration:
            if stop_requested:
                print("Stop requested - terminating execution")
                break

            elapsed = time.time() - start_time

            j1_pos = actuator_controller.relative_joint_angle(joint1_id)
            if j1_pos >= j1_upper_limit and j1_current_vel > 0:
                j1_current_vel = -j1_max_velocity
                print(f"  [J1] Upper boundary hit at {math.degrees(j1_pos):.1f}° — reversing.")
            elif j1_pos <= j1_lower_limit and j1_current_vel < 0:
                j1_current_vel = j1_max_velocity
                print(f"  [J1] Lower boundary hit at {math.degrees(j1_pos):.1f}° — reversing.")
            actuator_controller.set_velocity(joint1_id, j1_current_vel)

            j2_target_rad = j2_center_rad + j2_amplitude_rad * math.sin(j2_angular_freq * elapsed)
            raw2 = actuator_controller.relative_joint_angle_to_raw(joint2_id, j2_target_rad)
            actuator_controller.set_position(joint2_id, int(raw2))

            j3_pos = actuator_controller.relative_joint_angle(joint3_id)
            if j3_pos >= j3_upper_limit and j3_current_vel > 0:
                j3_current_vel = -j3_max_velocity
                print(f"  [J3] Upper boundary hit at {math.degrees(j3_pos):.1f}° — reversing.")
            elif j3_pos <= j3_lower_limit and j3_current_vel < 0:
                j3_current_vel = j3_max_velocity
                print(f"  [J3] Lower boundary hit at {math.degrees(j3_pos):.1f}° — reversing.")
            j3_current_vel = max(-j3_max_velocity, min(j3_max_velocity, j3_current_vel))
            actuator_controller.set_velocity(joint3_id, j3_current_vel)

            # ---- Status print (throttled) ----
            if int(elapsed * 10) % 5 == 0:
                j2_actual = actuator_controller.relative_joint_angle(joint2_id)
                print(f"  t={elapsed:5.2f}s | J1 pos={math.degrees(j1_pos):6.1f}° vel={j1_current_vel:+.2f} "
                      f"| J2 target={math.degrees(j2_target_rad):+6.1f}° actual={math.degrees(j2_actual):+6.1f}° "
                      f"| J3 pos={math.degrees(j3_pos):6.1f}° vel={j3_current_vel:+.2f}")

            time.sleep(update_rate)

        if stop_requested:
            print("Execution was terminated by user request.")

        final_j1 = actuator_controller.relative_joint_angle(joint1_id)
        final_j2 = actuator_controller.relative_joint_angle(joint2_id)
        final_j3 = actuator_controller.relative_joint_angle(joint3_id)

        return {
            "completed_successfully": not stop_requested,
            "final_joint1_deg": math.degrees(final_j1),
            "final_joint2_deg": math.degrees(final_j2),
            "final_joint3_deg": math.degrees(final_j3),
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

    finally:
        if 'actuator_controller' in locals():
            try:
                actuator_controller.set_velocity(joint1_id, 0.0)
                actuator_controller.set_velocity(joint3_id, 0.0)
                time.sleep(0.2)
                actuator_controller.disable_torque(joint1_id)
                actuator_controller.disable_torque(joint2_id)
                actuator_controller.disable_torque(joint3_id)
                print("Cleanup done — torque disabled on all joints.")
            except:
                pass


if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Challenge results: {results}")