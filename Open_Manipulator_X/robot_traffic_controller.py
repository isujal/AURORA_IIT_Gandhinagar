#!/usr/bin/env python3
# AB6_KIN_OMX_HW_004 — Robot Traffic Controller (OpenManipulator-X)
#
# Complete the gesture poses (radians) and optional keyboard teleop.

#!/usr/bin/env python3
# Traffic Teleoperation (TEMPLATE)
import time
import numpy as np
import sys
import math

from src.robot_core.open_manipulator_x import OpenManipulatorX

# Global flag used to stop the program safely
stop_requested = False

def stop_code():
    """Called by GUI when user presses stop."""
    global stop_requested
    stop_requested = True
    print("Stop requested.")


# ============================================================
# HELPER FUNCTION
# ============================================================
def move_robot_pose(ctrl, joint_ids, pose):
    """
    Move robot to a target pose.

    A pose contains one angle for each joint.

    Example:
    pose = [joint1, joint2, joint3, joint4]

    Hint:
    1. Loop through all joints.
    2. Get the matching angle.
    3. Convert angle to raw motor value.
    4. Send position command.

    Useful functions:
    - ctrl.convert_position_to_raw(...)
    - ctrl.set_position(...)
    """

    # Move every joint to its target angle
    for i, jid in enumerate(joint_ids):
        current_angle = pose[i]
        raw_val = ctrl.relative_joint_angle_to_raw(jid, current_angle)
        ctrl.set_position(jid, raw_val)

# ============================================================
# MAIN CHALLENGE
# ============================================================
def run_challenge(params):

    global stop_requested
    stop_requested = False

    robot = None

    try:

        # ----------------------------------------------------
        # Import keyboard package
        # ----------------------------------------------------
        try:
            import keyboard
        except ImportError:
            print("Install keyboard package first:")
            print("pip install keyboard")
            return {"success": False}

        print("Initializing OpenManipulator-X...")

        # Create robot object
        robot = OpenManipulatorX()

        # Verify robot connection
        if not robot.is_connected:
            print("Robot not connected.")
            return {"success": False}

        robot.ping_motors()

        ctrl = robot.controller
        joint_ids = robot.joint_ids

        # ----------------------------------------------------
        # TORQUE
        # ----------------------------------------------------
        # Torque must be enabled before motors can move.
        # Without torque the robot will not hold position.

        for jid in joint_ids:
            ctrl.enable_torque(jid)

        # ----------------------------------------------------
        # OPERATING MODE
        # ----------------------------------------------------
        # Mode 3 = Position Control Mode
        # This mode allows motors to move to target angles.

        for jid in joint_ids:
            ctrl.change_operating_mode(jid, 3)

        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------
        # Choose a safe speed between 5 and 30.
        # Lower value = slower movement.
        # Higher value = faster movement.

        SPEED = 10.0

        # ----------------------------------------------------
        # POSES (in Radians)
        # ----------------------------------------------------

        # TODO 1
        # Create STOP pose.
        # Goal: Arm points upward.
        STOP = [
            0.0, 0.0, -0.174533, -1.5708  
        ]

        HOME = [
            0.0, 0.0, -0.174533, 0.0  
        ]        

        # TODO 2
        # Create GO pose.
        # Goal: Arm points forward.
        GO = [
            0.0, 0.0,-0.174533, 0.0  
        ]

        # TODO 3
        # Create LEFT pose.
        # Goal: Robot points left.
        LEFT = [
            1.5708, 0.0, -0.174533, 0.0  
        ]

        # TODO 4
        # Create RIGHT pose.
        # Goal: Robot points right.
        RIGHT = [
            -1.5708, -0.0, -0.174533, 0.0  
        ]

        QUITING = [
            0.0, 0.0, 0.0, 0.0
        ]

        print("\n=== Robot Traffic Controller ===")
        print("G -> GO")
        print("S -> STOP")
        print("L -> LEFT")
        print("R -> RIGHT")
        print("H -> HOME")
        print("Q -> QUIT\n")

        current_command = None

        robot.move_to_rest_position(
                    wait_time=4.0,
                    speed=15
                )

        # ====================================================
        # MAIN CONTROL LOOP
        # ====================================================
        while not stop_requested:

            # ------------------------------------------------
            # GO COMMAND
            # ------------------------------------------------
            if keyboard.is_pressed('g') and current_command != 'g':
                print("GO")
                move_robot_pose(ctrl, joint_ids, GO)
                current_command = 'g'
                time.sleep(0.3)

            # ------------------------------------------------
            # STOP COMMAND
            # ------------------------------------------------
            elif keyboard.is_pressed('s') and current_command != 's':
                print("STOP")
                move_robot_pose(ctrl, joint_ids, STOP)
                current_command = 's'
                time.sleep(0.3)

            # ------------------------------------------------
            elif keyboard.is_pressed('h') and current_command != 'h':
                print("HOME")
                move_robot_pose(ctrl, joint_ids, HOME)
                current_command = 'h'
                time.sleep(0.3)                

            # ------------------------------------------------
            # LEFT COMMAND
            # ------------------------------------------------
            elif keyboard.is_pressed('l') and current_command != 'l':
                print("LEFT")
                move_robot_pose(ctrl, joint_ids, LEFT)
                current_command = 'l'
                time.sleep(0.3)

            # ------------------------------------------------
            # RIGHT COMMAND
            # ------------------------------------------------
            elif keyboard.is_pressed('r') and current_command != 'r':
                print("RIGHT")
                move_robot_pose(ctrl, joint_ids, RIGHT)
                current_command = 'r'
                time.sleep(0.3)

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------
            elif keyboard.is_pressed('q'):
                print("Quitting...")
                break

            # Reset command after key release
            if not any(
                keyboard.is_pressed(k)
                for k in ('g', 's', 'l', 'r')
            ):
                current_command = None

            time.sleep(0.02)

        print("Control loop exited.")
        return {"success": True}

    except Exception as e:

        print("Error:", e)

        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        print("Cleaning up...")

        if robot is not None:

            try:
                # Return robot to safe rest position
                robot.move_to_rest_position(
                    wait_time=8.0,
                    speed=15
                )
            except:
                pass

            try:
                # Disable torque before exiting
                for jid in robot.joint_ids:
                    ctrl.disable_torque(jid)
            except:
                pass

        print("Done.")