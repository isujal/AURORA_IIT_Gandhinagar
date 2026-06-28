import math
import time
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

# ── Poses ──────────────────────────────────────────────────────────────────

# point 2
# START_DEG = [94.0,  6.0, -14.0]
# END_DEG   = [94.0,  -30.0, -55.0]

# point 1 
# START_DEG = [40.0,  19.50, -68.0]
# END_DEG   = [ 40.0,  14.85, -68.0]

# point 3
# START_DEG = [40,  6, -68.0]
# END_DEG   = [ 40,  -15, -68.0]

# point (-7.5, 10)
# START_DEG = [185,  -77.69, 13.0]
# END_DEG   = [ 80,  -46.41, 14.0]

# point (-5, 15) (-7.5, 15) (0,17.5) (-2.5,17.5)  (-5, 17.5) 
# START_DEG = [130,  17.69, -31.0]
# END_DEG   = [ 20,  -70.41, -31.0]

# point (-10,12.5) 
# START_DEG = [130,  17.69, -41.0]
# END_DEG   = [ 20,  -70.41, -31.0]

# point 2
# START_DEG = [27.0,  99.0, 2.0]
# END_DEG   = [27.0,  30.0, 2.0]

START_DEG = [-4,  45.69, -56.0]
END_DEG   = [ 30,  45.41, -56.0]

END_RAD   = [math.radians(d) for d in END_DEG]

JOINT_IDS = [1, 2, 3]

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True


def run_challenge(params):
    global stop_requested
    stop_requested = False

    try:
        ctrl = ActuatorController("actuator_config.json")

        # Enable all joints in POSITION mode
        for jid in JOINT_IDS:
            ctrl.disable_torque(jid)
            # time.sleep(0.1)
            ctrl.change_operating_mode(jid, OperatingMode.POSITION)
            # time.sleep(0.1)
            ctrl.enable_torque(jid)
            # time.sleep(0.1)

        print("Sending target positions...")

        # Fire all joints simultaneously in one shot
        for i, jid in enumerate(JOINT_IDS):
            raw = ctrl.relative_joint_angle_to_raw(jid, END_RAD[i])
            ctrl.set_position(jid, int(raw))
            print(f"  J{jid}: {START_DEG[i]:+.2f}° → {END_DEG[i]:+.2f}°")

        print("Commands sent — waiting for motion to complete...")
        time.sleep(3.0)

        # Readback
        print("\nFinal positions:")
        for i, jid in enumerate(JOINT_IDS):
            actual_deg = math.degrees(ctrl.relative_joint_angle(jid))
            print(f"  J{jid}: target={END_DEG[i]:+.2f}°  actual={actual_deg:+.2f}°  error={actual_deg - END_DEG[i]:+.2f}°")

        return {"completed_successfully": True}

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        if 'ctrl' in locals():
            try:
                for jid in JOINT_IDS:
                    ctrl.disable_torque(jid)
                print("Torque disabled.")
            except Exception:
                pass


if __name__ == "__main__":
    run_challenge(params={})