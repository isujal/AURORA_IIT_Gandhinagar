import numpy as np
import time
import math
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")

def run_challenge(params):
    global stop_requested
    stop_requested = False

    duration    = 30
    update_rate = 0.02  # 20ms = smooth sine tracking

    try:
        actuator_controller = ActuatorController("actuator_config.json")
        joint2_id = 2

        print("=== Stage 2: Joint 2 Sinusoidal Position Control ===")
        print("Centre: 50°  |  Amplitude: ±80°  |  Range: -30° to 130°")

        # Setup Joint 2 in POSITION mode
        actuator_controller.disable_torque(joint2_id)
        actuator_controller.change_operating_mode(joint2_id, OperatingMode.POSITION)
        actuator_controller.enable_torque(joint2_id)

        # Move to start position (-30°) before beginning sine
        print("Moving to start position -30°...")
        raw = actuator_controller.relative_joint_angle_to_raw(
            joint2_id, math.radians(-30.0)
        )
        actuator_controller.set_position(joint2_id, int(raw))
        time.sleep(2.0)

        actual = actuator_controller.relative_joint_angle(joint2_id)
        print(f"At start: {math.degrees(actual):.2f}°")


        j2_centre    = math.radians(50.0)
        j2_amplitude = math.radians(80.0)
        j2_period    = 2.0   

        lower_clamp  = math.radians(-30.0)
        upper_clamp  = math.radians(130.0)

        print(f"Period: {j2_period}s  |  Update rate: {update_rate*1000:.0f}ms")
        print(f"Running for {duration}s...\n")

        start_time = time.time()

        while (time.time() - start_time) < duration:
            if stop_requested:
                print("Stop requested - terminating.")
                break

            elapsed = time.time() - start_time

            # Smooth -cos sine wave
            j2_target = j2_centre - j2_amplitude * math.cos(
                2 * math.pi * elapsed / j2_period
            )

            # Hard clamp — safety net
            j2_target = max(lower_clamp, min(upper_clamp, j2_target))

            # Send position command
            raw = actuator_controller.relative_joint_angle_to_raw(joint2_id, j2_target)
            actuator_controller.set_position(joint2_id, int(raw))

            # Print every second
            if int(elapsed) != int(elapsed - update_rate):
                actual = actuator_controller.relative_joint_angle(joint2_id)
                error  = math.degrees(actual - j2_target)
                print(
                    f"  t={elapsed:.1f}s | "
                    f"target={math.degrees(j2_target):6.1f}° | "
                    f"actual={math.degrees(actual):6.1f}° | "
                    f"error={error:5.1f}°"
                )

            time.sleep(update_rate)

        final = actuator_controller.relative_joint_angle(joint2_id)
        print(f"\nDone. Final Joint 2 position: {math.degrees(final):.2f}°")

        return {
            "completed_successfully": not stop_requested,
            "final_j2_deg":  math.degrees(final),
            "centre_deg":    50.0,
            "amplitude_deg": 80.0,
            "range":         "-30° to 130°"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        if 'actuator_controller' in locals():
            try:
                actuator_controller.disable_torque(joint2_id)
                print("Cleanup done.")
            except:
                pass

if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Results: {results}")