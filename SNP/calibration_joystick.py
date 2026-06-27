import time
import pygame
from src.robot_core.snp_model import RobotModel

# ─── Robot Configuration ───────────────────────────────────────────────────────
ROBOT_IP   = "10.82.113.199"
ROBOT_PORT = 5000
LOCAL_PORT = 5000
MAX_SPEED  = 0.8

# ─── Stop Flag ────────────────────────────────────────────────────────────────
stop_requested = False

def stop_code():
    """Called by the GUI Stop button — do not modify."""
    global stop_requested
    stop_requested = True

# ─── Main Challenge Function ───────────────────────────────────────────────────
def run_challenge(params):
    global stop_requested
    stop_requested = False

    # ── Initialize pygame and controller ──────────────────────────────────────
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No controller found! Pair your Xbox controller via Bluetooth first.")
        return {'success': False, 'error': 'No controller detected'}

    controller = pygame.joystick.Joystick(0)
    controller.init()
    print(f"Controller connected: {controller.get_name()}")

    try:
        # ── Connect to Robot ───────────────────────────────────────────────────
        robot = RobotModel(
            mode       = "wifi",
            robot_ip   = ROBOT_IP,
            robot_port = ROBOT_PORT,
            local_port = LOCAL_PORT
        )
        print("Robot connected!")
        print("Controls:")
        print("  Left stick  Y → Forward / Backward  (vx)")
        print("  Right stick X → Rotate left / right (vw)")
        print("  B button      → Emergency stop")
        print("  GUI Stop      → Exit")

        # ── Deadzone Helper ────────────────────────────────────────────────────
        def deadzone(value, threshold=0.1):
            """Ignore small stick drift below threshold."""
            return value if abs(value) > threshold else 0.0

        # ── Control Loop ───────────────────────────────────────────────────────
        while True:

            # Check GUI stop button
            if stop_requested:
                print("Stop requested via GUI.")
                break

            # Refresh pygame event queue (required for axis values to update)
            pygame.event.pump()

            # ── Read Controller Axes ───────────────────────────────────────────
            # Axis 1 = Left stick Y  → vx (negated so up = forward)
            # Axis 0 = Left stick X  → vy (strafe, only for holonomic robots)
            # Axis 3 = Right stick X → vw (rotation)
            left_y  = -controller.get_axis(1)   # forward / backward
            right_x =  controller.get_axis(3)   # rotation

            # Apply deadzone and scale to max speed
            vx = deadzone(left_y)  * MAX_SPEED
            vy = 0.0                             # set to deadzone(controller.get_axis(0)) * MAX_SPEED if holonomic
            vw = deadzone(right_x) * MAX_SPEED

            # ── Read Buttons ───────────────────────────────────────────────────
            # B button (index 1) → emergency stop
            if controller.get_button(1):
                print("B button pressed — stopping robot.")
                break

            # ── Update Robot ───────────────────────────────────────────────────
            robot.update()
            robot.send_velocity(vx, vy, vw)

            # ── Read and Print Sensors ─────────────────────────────────────────
            sensors = robot.get_sensors()
            print(f"IR Distance: {sensors.sharp_ir_distance:.3f} m | vx={vx:.2f}  vy={vy:.2f}  vw={vw:.2f}")

            # 10 Hz control loop
            time.sleep(0.1)

        # ── Safe Stop ──────────────────────────────────────────────────────────
        print("Sending zero velocity...")
        robot.send_velocity(0, 0, 0)
        robot.close()
        pygame.quit()
        print("Connection closed. Goodbye!")
        return {'success': True}

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        return {'success': False, 'error': str(e)}


# ─── Axis Debugger (run this first to verify your axis mapping) ───────────────
def debug_controller():
    """
    Run this function standalone to check which axis index corresponds
    to which stick on YOUR controller. Move each stick and observe output.
    """
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No controller found!")
        return

    j = pygame.joystick.Joystick(0)
    j.init()
    print(f"Controller: {j.get_name()}")
    print("Move sticks to identify axis indices. Ctrl+C to stop.\n")

    try:
        while True:
            pygame.event.pump()
            axes = [round(j.get_axis(i), 2) for i in range(j.get_numaxes())]
            print(f"Axes: {axes}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDebug stopped.")
        pygame.quit()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Uncomment the line below FIRST to verify your axis mapping,
    # then comment it out and run run_challenge() for actual driving.

    # debug_controller()

    result = run_challenge(params={})
    print(f"Result: {result}")