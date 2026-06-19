import numpy as np
import time
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested - execution terminating.")

def run_challenge(params):
    global stop_requested
    stop_requested = False

    update_rate = 0.01
    actuator_controller = None

    # ==========================================================
    # SET BASELINE CONFIGURATION (Straight up along Y-axis)
    # ==========================================================
    J1_UPRIGHT = 90.0    # Base joint points straight down the board's Y-axis
    J2_UPRIGHT = 0.0     # Middle joint aligned straight with link 1
    J3_UPRIGHT = 0.0     # Head joint aligned straight with link 2

    # ==========================================================
    # SNAKE MOVEMENT PARAMETERS (In Degrees)
    # ==========================================================
    J1_SWING = 60.0      # Base wiggles clock & anti-clockwise by 30°
    J2_SWING = 60.0      # Middle joint wiggles clock & anti-clockwise by 60°
    
    DURATION = 15.0      # Total run time
    # ==========================================================

    try:
        actuator_controller = ActuatorController('actuator_config.json')
        print("Initializing Actuators...")

        # Dynamic API mapping to handle backend variations seamlessly
        def get_method(names):
            for name in names:
                if hasattr(actuator_controller, name):
                    return getattr(actuator_controller, name)
            raise AttributeError(f"Could not find any of these methods: {names}")

        set_position = get_method(['set_position', 'set_target_position', 'write_position'])
        get_position = get_method(['relative_joint_angle', 'get_present_position', 'get_position'])
        angle_to_raw = get_method(['relative_joint_angle_to_raw', 'angle_to_raw'])
        set_mode = get_method(['change_operating_mode', 'set_mode', 'set_operating_mode'])

        for jid in [1, 2, 3]:
            actuator_controller.disable_torque(jid)
            set_mode(jid, OperatingMode.POSITION)
            actuator_controller.enable_torque(jid)

        # ----------------------------------------------------------
        # PHASE 1: Smoothly align to the Y-axis center line
        # ----------------------------------------------------------
        print("Aligning robot to Y-axis centerline...")
        j1_up = np.radians(J1_UPRIGHT)
        j2_up = np.radians(J2_UPRIGHT)
        j3_up = np.radians(J3_UPRIGHT)

        start_j1 = get_position(1)
        start_j2 = get_position(2)
        start_j3 = get_position(3)
        ramp_start = time.time()
        ramp_time = 2.0

        while (time.time() - ramp_start) < ramp_time:
            if stop_requested: break
            loop_start = time.time()
            alpha = (loop_start - ramp_start) / ramp_time
            smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0

            a1 = start_j1 + (j1_up - start_j1) * smooth_alpha
            a2 = start_j2 + (j2_up - start_j2) * smooth_alpha
            a3 = start_j3 + (j3_up - start_j3) * smooth_alpha

            set_position(1, int(angle_to_raw(1, a1)))
            set_position(2, int(angle_to_raw(2, a2)))
            set_position(3, int(angle_to_raw(3, a3)))
            time.sleep(max(0.001, update_rate - (time.time() - loop_start)))

        print("Robot centered on Y-axis. Commencing stabilized snake motion...")
        time.sleep(1.0)

        # ----------------------------------------------------------
        # PHASE 2: Snake Body Slither + Head Stabilization
        # ----------------------------------------------------------
        j1_amp = np.radians(J1_SWING)
        j2_amp = np.radians(J2_SWING)

        omega = 2.0 * np.pi * 0.6   # Speed of the snake slither (0.6 Hz)
        phase_delay = np.pi / 2.0   # 90-degree phase shift between J1 and J2 for organic ripple

        start_time = time.time()

        while True:
            if stop_requested: break
            elapsed = time.time() - start_time
            if elapsed >= DURATION: break

            loop_start = time.time()

            # 1. Drive Joint 1 into a ±30° clock/anti-clockwise wave around its 90° center
            angle_j1 = j1_up + j1_amp * np.sin(omega * elapsed)
            
            # 2. Drive Joint 2 into a ±60° ripple, slightly delayed behind Joint 1
            angle_j2 = j2_up + j2_amp * np.sin(omega * elapsed - phase_delay)

            # 3. HEAD STABILIZATION MATH:
            # To keep the head pointing perfectly straight along the global Y-axis:
            # Global Orientation = (angle_j1 - 90°) + angle_j2 + angle_j3 = 0°
            # Therefore, Joint 3 must exactly cancel out the accumulated relative angles of the body:
            body_deviation = (angle_j1 - j1_up) + angle_j2
            angle_j3 = j3_up - body_deviation

            # Commit the raw positions to the actuators
            set_position(1, int(angle_to_raw(1, angle_j1)))
            set_position(2, int(angle_to_raw(2, angle_j2)))
            set_position(3, int(angle_to_raw(3, angle_j3)))

            # Maintain steady loop timing
            time.sleep(max(0.001, update_rate - (time.time() - loop_start)))

        print("Snake challenge successfully concluded.")
        return {"completed_successfully": not stop_requested}

    except Exception as e:
        print(f"Error during execution: {e}")
        return {"completed_successfully": False, "error": str(e)}

    finally:
        print("Cleaning up torque fields...")
        if actuator_controller is not None:
            for jid in [1, 2, 3]:
                try:
                    actuator_controller.disable_torque(jid)
                except: pass
        print("Cleanup complete.")

if __name__ == "__main__":
    run_challenge(params={})