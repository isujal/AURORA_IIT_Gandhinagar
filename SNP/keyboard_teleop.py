import time
from src.robot_core.snp_model import RobotModel, Teleop

# ─── Robot Configuration ───────────────────────────────────────────────────────
# TODO: Replace with your robot's IP address (check the Phone screen/ In Connected Devices)
ROBOT_IP   = "10.82.113.199"
ROBOT_PORT = 5000
LOCAL_PORT = 5000

# ─── Stop Flag ────────────────────────────────────────────────────────────────
stop_requested = False

def stop_code():
    """Called by the GUI Stop button — do not modify."""
    global stop_requested
    stop_requested = True

# ─── Main Challenge Function ───────────────────────────────────────────────────
def run_challenge(params):
    """
    Your task: Complete this function to control the SNP robot using keyboard input.

    What this function should do:
      1. Connect to the robot over WiFi
      2. Start a control loop that:
           - Reads keyboard input (arrow keys / WASD)
           - Converts key press → velocity command (vx, vy, vw)
           - Sends velocity to the robot
           - Reads and prints sensor data
      3. Stop the robot when Stop is pressed or a key condition is met

    Key concepts to study:
      - RobotModel: manages WiFi connection and sensor data
      - Teleop: reads keyboard and computes velocity
      - robot.update(): fetches latest sensor data from the robot
      - robot.send_velocity(vx, vy, vw): sends motion command
          vx = forward/backward  (m/s)
          vy = left/right strafe (m/s) — 0 for differential drive
          vw = rotation          (rad/s)
      - robot.get_sensors(): returns all sensor readings
    """
    global stop_requested
    stop_requested = False

    try:
        # ── Step 1: Initialize Robot ───────────────────────────────────────────
        # TODO: Create a RobotModel in WiFi mode using the IP above
        # Hint: RobotModel(mode="wifi", robot_ip=..., robot_port=..., local_port=...)
        robot = RobotModel(mode = "wifi", robot_ip = ROBOT_IP, robot_port = ROBOT_PORT, local_port = LOCAL_PORT)  # replace this line


        # ── Step 2: Initialize Teleop ──────────────────────────────────────────
        # Teleop reads keyboard and computes vx, vy, vw
        # The second argument is max speed (0.0 to 1.0)
        # TODO: Create a Teleop object
        # Hint: Teleop(robot, max_speed)
        teleop = Teleop(robot,0.8)  # replace this line


        print("Robot connected! Use arrow keys or WASD to drive. Press Stop to exit.")

        # ── Step 3: Control Loop ───────────────────────────────────────────────
        while True:

            # Check if user pressed Stop button
            if stop_requested:
                print("Stop requested.")
                break

            # TODO: Update sensor data from robot
            # Hint: robot.update()
            robot.update()

            # TODO: Update teleop (reads keyboard input)
            # Hint: teleop.update()
            teleop.update()


            # TODO: Send velocity to robot
            # Hint: robot.send_velocity(teleop.vx, teleop.vy, teleop.vw)
            robot.send_velocity(teleop.vx, teleop.vy, teleop.vw)


            # TODO: Read sensors and print at least one value
            # Hint: sensors = robot.get_sensors()
            #        print(sensors.sharp_ir_distance)

            sensors = robot.get_sensors()
            print(sensors.sharp_ir_distance)


            # Control loop runs at 10Hz (100ms per cycle)
            time.sleep(0.1)

        # ── Step 4: Stop Robot ─────────────────────────────────────────────────
        # TODO: Send zero velocity to stop the robot safely
        # Hint: robot.send_velocity(0, 0, 0)
        robot.send_velocity(0, 0, 0)


        # TODO: Close the robot connection
        # Hint: robot.close()
        robot.close()


        return {'success': True}

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    result = run_challenge(params={})
    print(f"Result: {result}")