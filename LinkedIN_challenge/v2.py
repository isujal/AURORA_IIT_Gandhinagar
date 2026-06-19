import numpy as np
import time
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

"""
CUSTOM CHALLENGE: Shape-Shifting Alphabet (CHIG)
------------------
The robot uses extreme mechanical folds to morph into letter shapes.
"""

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested - execution terminating.")

# ==========================================================
# THE EXACT ALPHABET POSES (J1, J2, J3)
# ==========================================================
LETTER_POSES = {
    "C": [90.0, -90.0, 90.0],    # PERFECTED: Open 90-degree step shape from photo
    "H": [90.0, -90.0, -90.0],   # Downward hook shape
    "I": [90.0, 0.0, 0.0],       # Perfectly straight line
    "G": [90.0, -150.0, 150.0]   # Tight Z-fold
}

def morph_to_shape(actuator, target_j1, target_j2, target_j3, transition_time=3.0):
    """Smoothly glides the joints to the target shape angles."""
    global stop_requested
    update_rate = 0.01

    # Get current positions
    try:
        start_j1 = np.degrees(actuator.relative_joint_angle(1))
        start_j2 = np.degrees(actuator.relative_joint_angle(2))
        start_j3 = np.degrees(actuator.relative_joint_angle(3))
    except:
        start_j1, start_j2, start_j3 = 90.0, 0.0, 0.0

    ramp_start = time.time()
    while (time.time() - ramp_start) < transition_time:
        if stop_requested:
            break
            
        loop_start = time.time()
        alpha = (loop_start - ramp_start) / transition_time
        smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0 

        # Interpolate angles smoothly
        curr_j1 = start_j1 + (target_j1 - start_j1) * smooth_alpha
        curr_j2 = start_j2 + (target_j2 - start_j2) * smooth_alpha
        curr_j3 = start_j3 + (target_j3 - start_j3) * smooth_alpha

        # Send radians to motor
        actuator.set_position(1, int(actuator.relative_joint_angle_to_raw(1, np.radians(curr_j1))))
        actuator.set_position(2, int(actuator.relative_joint_angle_to_raw(2, np.radians(curr_j2))))
        actuator.set_position(3, int(actuator.relative_joint_angle_to_raw(3, np.radians(curr_j3))))
        
        time.sleep(max(0.001, update_rate - (time.time() - loop_start)))

def run_challenge(params):
    global stop_requested
    stop_requested = False
    actuator_controller = None

    try:
        actuator_controller = ActuatorController('actuator_config.json')
        print("Initializing Hardware for Extreme Shape-Shifting...")

        for jid in [1, 2, 3]:
            actuator_controller.disable_torque(jid)
            time.sleep(0.05)
            actuator_controller.change_operating_mode(jid, OperatingMode.POSITION)
            time.sleep(0.05)
            actuator_controller.enable_torque(jid)

        print("Moving to safe straight position...")
        morph_to_shape(actuator_controller, 90.0, 0.0, 0.0, transition_time=4.0)
        time.sleep(2.0)

        # ----------------------------------------------------------
        # PHASE 2: MORPH INTO EACH LETTER OF "CHIG"
        # ----------------------------------------------------------
        name_to_spell = "CHIG"
        
        for letter in name_to_spell:
            if stop_requested:
                break
                
            print(f"\n>>> Morphing into Letter: {letter}")
            target_angles = LETTER_POSES.get(letter)
            
            if target_angles:
                # Slowly fold into the shape (takes 3.5 seconds)
                morph_to_shape(actuator_controller, target_angles[0], target_angles[1], target_angles[2], transition_time=3.5) 
                
                # Hold the shape so you can see the letter perfectly!
                print(f"    [Holding shape '{letter}' for 4 seconds...]")
                time.sleep(4.0)

        print("\nShape-shifting complete!")
        print("Returning to safe home position...")
        morph_to_shape(actuator_controller, 90.0, 0.0, 0.0, transition_time=4.0)

        return {"completed_successfully": not stop_requested}

    except Exception as e:
        print(f"Error: {e}")
        return {"completed_successfully": False, "error": str(e)}

    finally:
        print("Cleaning up...")
        if actuator_controller is not None:
            for jid in [1, 2, 3]:
                try:
                    actuator_controller.set_velocity(jid, 0)
                    time.sleep(0.02)
                    actuator_controller.disable_torque(jid)
                except:
                    pass
        print("Cleanup complete. Robot safe.")

if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Results: {results}")