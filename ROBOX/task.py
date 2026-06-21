# TODO Challenge 5

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
1. Specify target joint angles for two poses
2. Observe forward kinematics predictions for each
3. Watch the robot move to Pose 1, then Pose 2
4. Compare predicted vs actual end-effector position for each pose
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


def move_to_pose(robot, actuator_controller, joint_ids, theta_targets, pose_name, hold_time=5.0):
    """
    Runs FK prediction, sends joint commands, waits, then reads back
    actual angles and computes end-effector error for one pose.
    Returns (actual_angles, position_error_mm, orientation_error_deg).
    """
    global stop_requested

    theta1_target, theta2_target, theta3_target = theta_targets

    print(f"\n========== {pose_name} ==========")
    print("== TARGET JOINT ANGLES ==")
    print(f"theta1 = {theta1_target:.4f} rad ({np.degrees(theta1_target):.2f} deg)")
    print(f"theta2 = {theta2_target:.4f} rad ({np.degrees(theta2_target):.2f} deg)")
    print(f"theta3 = {theta3_target:.4f} rad ({np.degrees(theta3_target):.2f} deg)")
    print("=========================================")

    # Forward Kinematics (Prediction)
    print("Running Forward Kinematics...")
    T_target = robot.forward_kinematics([theta1_target, theta2_target, theta3_target])

    target_pos = T_target[:3, 3]
    target_rot = T_target[:3, :3]
    ee_direction = math.atan2(target_rot[1, 0], target_rot[0, 0])

    print("Predicted End-Effector Pose:")
    print(f"x = {target_pos[0]:.4f} m")
    print(f"y = {target_pos[1]:.4f} m")
    print(f"z = {target_pos[2]:.4f} m")
    print(f"orientation (XY plane) = {np.degrees(ee_direction):.2f} deg")
    print(f"distance from origin = {np.sqrt(target_pos[0]**2 + target_pos[1]**2)*1000:.1f} mm")

    # Send joint commands
    print("Sending commands to motors...")
    raw1 = actuator_controller.relative_joint_angle_to_raw(1, theta1_target)
    raw2 = actuator_controller.relative_joint_angle_to_raw(2, theta2_target)
    raw3 = actuator_controller.relative_joint_angle_to_raw(3, theta3_target)

    actuator_controller.set_position(1, int(raw1))
    actuator_controller.set_position(2, int(raw2))
    actuator_controller.set_position(3, int(raw3))

    print(f"Robot moving to {pose_name}...")
    move_start = time.time()
    while time.time() - move_start < hold_time:
        if stop_requested:
            print("Stop requested during motion.")
            break
        time.sleep(0.05)

    # Read back actual joint angles
    print("===== ACTUAL JOINT ANGLES =====")
    actual_angles = []
    for jid, commanded in zip(joint_ids, theta_targets):
        actual = actuator_controller.relative_joint_angle(jid)
        actual_angles.append(actual)
        error = actual - commanded

        print(f"Joint {jid}:")
        print(f"  commanded = {commanded:.4f} rad ({np.degrees(commanded):.2f} deg)")
        print(f"  actual    = {actual:.4f} rad ({np.degrees(actual):.2f} deg)")
        print(f"  error     = {error:.4f} rad ({np.degrees(error):.2f} deg)")

    # FK using actual joint angles
    print("===== ACTUAL END-EFFECTOR POSITION =====")
    T_actual = robot.forward_kinematics(actual_angles)
    actual_pos = T_actual[:3, 3]
    actual_rot = T_actual[:3, :3]
    actual_dir = math.atan2(actual_rot[1, 0], actual_rot[0, 0])

    error_xyz = target_pos - actual_pos
    error_total = np.linalg.norm(error_xyz)
    error_orientation = abs(ee_direction - actual_dir)
    if error_orientation > math.pi:
        error_orientation = 2 * math.pi - error_orientation

    print(f"Target EE position = {target_pos}")
    print(f"Actual EE position = {actual_pos}")

    print("Errors:")
    print(f"x error = {abs(error_xyz[0])*1000:.2f} mm")
    print(f"y error = {abs(error_xyz[1])*1000:.2f} mm")
    print(f"z error = {abs(error_xyz[2])*1000:.2f} mm")
    print(f"total position error = {error_total*1000:.2f} mm")
    print(f"orientation error = {np.degrees(error_orientation):.2f} deg")

    if error_total < 0.005:
        print("Excellent accuracy (< 5 mm)")
    elif error_total < 0.010:
        print("Good accuracy (< 10 mm)")
    elif error_total < 0.020:
        print("Moderate accuracy (< 20 mm)")
    else:
        print("Poor accuracy (> 20 mm)")

    return actual_angles, error_total * 1000, np.degrees(error_orientation)


def run_challenge(params):
    """
    Forward kinematics challenge using built-in FK.
    Moves the arm through two poses, one after the other.
    """
    global stop_requested
    stop_requested = False

    try:
        # Initialize robot and controller
        robot = Robot.from_config("robot_parameters.json")
        actuator_controller = ActuatorController("actuator_config.json")

        joint_ids = [1, 2, 3]  # TODO: confirm joint IDs for your robot

        # -------------------------------------------------
        # Define both poses (in radians)
        # -------------------------------------------------
        pose1 = (
            math.radians(60),    # theta1  //60  <->  120
            math.radians(70),   # theta2  //130 <-> -130
            math.radians(-40),   # theta3  //-90 <->  90
        )
        pose2 = (
            math.radians(120),
            math.radians(-70),
            math.radians(40),
        )

        # Enable position mode on all joints once
        for jid in joint_ids:
            actuator_controller.disable_torque(jid)
            actuator_controller.change_operating_mode(jid, OperatingMode.POSITION)
            actuator_controller.enable_torque(jid)

        results_per_pose = []

        # ---- Move to Pose 1 ----
        if not stop_requested:
            actual1, pos_err1, ori_err1 = move_to_pose(
                robot, actuator_controller, joint_ids, pose1, "POSE 1", hold_time=0.50
            )
            results_per_pose.append({
                "pose": "pose1",
                "position_error_mm": pos_err1,
                "orientation_error_deg": ori_err1,
            })

        # ---- Move to Pose 2 ----
        if not stop_requested:
            actual2, pos_err2, ori_err2 = move_to_pose(
                robot, actuator_controller, joint_ids, pose2, "POSE 2", hold_time=0.50
            )
            results_per_pose.append({
                "pose": "pose2",
                "position_error_mm": pos_err2,
                "orientation_error_deg": ori_err2,
            })

        return {
            "completed_successfully": not stop_requested,
            "results": results_per_pose,
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