# Inverse Kinematics Challenge
import numpy as np
import time
import math
from robot_core.robot_model import Robot
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

"""
CHALLENGE OVERVIEW:
------------------
In this challenge, students will:
1. Specify a target end-effector position and orientation
2. Solve inverse kinematics using the built-in solver
3. Command the robot to the IK solution
4. Verify accuracy using forward kinematics
"""

# Global flag to signal code termination
stop_requested = False


def stop_code():
    """
    Sets the stop_requested flag to True to safely terminate execution.
    """
    global stop_requested
    stop_requested = True
    print("Stop requested - execution will terminate safely.")


def run_challenge(params):
    """
    Inverse kinematics challenge using built-in IK.
    """
    global stop_requested
    stop_requested = False

    try:
        # Initialize robot and controller
        robot = Robot.from_config("robot_parameters.json")
        actuator_controller = ActuatorController("actuator_config.json")

        joint_ids = [1,2,3]  # TODO: confirm joint IDs for your robot

        # -------------------------------------------------
        # Target end-effector pose (edit here)
        # -------------------------------------------------
        target_x = -0.15 # meters
        target_y = 0.15    # meters
        phi = math.pi/4        # radians (0 = right)

        print("== TARGET END-EFFECTOR POSE ==")
        print(f"x = {target_x:.4f} m")
        print(f"y = {target_y:.4f} m")
        print(f"phi = {phi:.4f} rad ({np.degrees(phi):.2f} deg)")
        print("==============================================")

        # -------------------------------------------------
        # Build target transformation matrix
        # -------------------------------------------------
        T = np.eye(4)
        T[0, 3] = target_x
        T[1, 3] = target_y
        T[0, 0] = math.cos(phi) 
        T[0, 1] = -math.sin(phi) 
        T[1, 0] = math.sin(phi) 
        T[1, 1] = math.cos(phi) 

        position = T[:3,3] 
        orientation = T[:3, :3]

        # Initial guess to help IK convergence
        target_angle = math.atan2(target_y, target_x)
        initial_guess = [target_angle, 0.0, 0.0]

        # -------------------------------------------------
        # Run Inverse Kinematics
        # -------------------------------------------------
        print("Running Inverse Kinematics...")

        q = robot.inverse_kinematics(
            position,
            orientation,
            initial_guess=initial_guess
        )

        theta1, theta2, theta3 = q

        print("IK Solution:")
        print(f"theta1 = {theta1:.4f} rad ({np.degrees(theta1):.2f} deg)")
        print(f"theta2 = {theta2:.4f} rad ({np.degrees(theta2):.2f} deg)")
        print(f"theta3 = {theta3:.4f} rad ({np.degrees(theta3):.2f} deg)")

        # -------------------------------------------------
        # Send joint commands
        # -------------------------------------------------
        print("Sending commands to motors...")

        for jid in joint_ids:
            actuator_controller.disable_torque(jid)
            actuator_controller.change_operating_mode(jid, OperatingMode.POSITION)
            actuator_controller.enable_torque(jid)

        raw1 = actuator_controller.relative_joint_angle_to_raw(1, theta1 )
        raw2 = actuator_controller.relative_joint_angle_to_raw(2, theta2 )
        raw3 = actuator_controller.relative_joint_angle_to_raw(3, theta3 )

        actuator_controller.set_position(1, int(raw1))
        actuator_controller.set_position(2, int(raw2))
        actuator_controller.set_position(3, int(raw3))

        print("Robot moving to IK solution...")
        move_start = time.time()

        while time.time() - move_start < 5.0:
            if stop_requested:
                print("Stop requested during motion.")
                break
            time.sleep(0.05)

        # -------------------------------------------------
        # Read back actual joint angles
        # -------------------------------------------------
        print("===== ACTUAL JOINT ANGLES =====")
        actual_angles = []

        for jid, commanded in zip(
            joint_ids,
            [theta1, theta2, theta3]
        ):
            actual = actuator_controller.relative_joint_angle(jid)
            actual_angles.append(actual)
            error = actual - commanded

            print(f"Joint {jid}:")
            print(f"  commanded = {commanded:.4f} rad ({np.degrees(commanded):.2f} deg)")
            print(f"  actual    = {actual:.4f} rad ({np.degrees(actual):.2f} deg)")
            print(f"  error     = {error:.4f} rad ({np.degrees(error):.2f} deg)")

        # -------------------------------------------------
        # Verify end-effector position using FK
        # -------------------------------------------------
        print("===== ACTUAL END-EFFECTOR POSITION =====")

        T_actual = robot.forward_kinematics(actual_angles)
        actual_pos = T_actual[:3, 3]

        error_x = abs(target_x - actual_pos[0])
        error_y = abs(target_y - actual_pos[1])
        error_total = math.sqrt(error_x**2 + error_y**2)

        print(f"Target position = [{target_x:.4f}, {target_y:.4f}, 0.0000]")
        print(f"Actual position = [{actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f}]")

        print("Position errors:")
        print(f"x error = {error_x*1000:.2f} mm")
        print(f"y error = {error_y*1000:.2f} mm")
        print(f"total error = {error_total*1000:.2f} mm")

        if error_total < 0.005:
            print("Excellent accuracy (< 5 mm)")
        elif error_total < 0.010:
            print("Good accuracy (< 10 mm)")
        elif error_total < 0.020:
            print("Moderate accuracy (< 20 mm)")
        else:
            print("Poor accuracy (> 20 mm)")

        return {
            "completed_successfully": not stop_requested,
            "position_error_mm": error_total * 1000
        }

    except Exception as e:
        print(f"Error occurred: {e}")
        return {
            "completed_successfully": False,
            "error": str(e)
        }

    finally:
        if 'actuator_controller' in locals():
            for jid in (joint_ids if 'joint_ids' in dir() else []):
                try:
                    actuator_controller.disable_torque(jid)
                except:
                    pass
            print("Motors stopped and torque disabled.")


if __name__ == "__main__":
    results = run_challenge(params={})
    print(f"Challenge results: {results}")
