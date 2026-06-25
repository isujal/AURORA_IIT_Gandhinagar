#!/usr/bin/env python3
import time
import numpy as np
import sys

from src.robot_core.open_manipulator_x import OpenManipulatorX

stop_requested = False


def deg_to_rad(deg_list):
    return [np.deg2rad(d) for d in deg_list]


def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")


def move_robot_pose(ctrl, joint_ids, pose):
    """Send each joint to its target angle."""
    for jid, angle in zip(joint_ids, pose):
        raw = ctrl.convert_position_to_raw(angle)
        ctrl.set_position(jid, raw)


def run_challenge(params):

    global stop_requested
    stop_requested = False
    robot = None

    try:
        try:
            import keyboard
        except ImportError:
            print("Install keyboard package first: pip install keyboard")
            return {"success": False}

        print("Initializing OpenManipulator-X...")
        robot = OpenManipulatorX()

        if not robot.is_connected:
            print("Robot not connected.")
            return {"success": False}

        robot.ping_motors()

        ctrl = robot.controller
        joint_ids = robot.joint_ids

        # Enable torque
        for jid in joint_ids:
            ctrl.enable_torque(jid)

        # Position control mode
        for jid in joint_ids:
            ctrl.change_operating_mode(jid, 3)

        # Safe speed
        SPEED = 20

        # ── Poses ──────────────────────────────────────────
        STOP  = deg_to_rad([  0,  0,  0, -90])  # arm up
        GO    = deg_to_rad([  0,  0,  0,   0])  # arm forward
        LEFT  = deg_to_rad([ 90,  0,  0,   0])  # base +90°
        RIGHT = deg_to_rad([-90,  0,  0,   0])  # base -90°
        # ───────────────────────────────────────────────────

        print("\n=== Robot Traffic Controller ===")
        print("G -> GO | S -> STOP | L -> LEFT | R -> RIGHT | Q -> QUIT\n")

        current_command = None

        while not stop_requested:

            if keyboard.is_pressed('g') and current_command != 'g':
                print("GO")
                move_robot_pose(ctrl, joint_ids, GO)
                current_command = 'g'
                time.sleep(0.3)

            elif keyboard.is_pressed('s') and current_command != 's':
                print("STOP")
                move_robot_pose(ctrl, joint_ids, STOP)
                current_command = 's'
                time.sleep(0.3)

            elif keyboard.is_pressed('l') and current_command != 'l':
                print("LEFT")
                move_robot_pose(ctrl, joint_ids, LEFT)
                current_command = 'l'
                time.sleep(0.3)

            elif keyboard.is_pressed('r') and current_command != 'r':
                print("RIGHT")
                move_robot_pose(ctrl, joint_ids, RIGHT)
                current_command = 'r'
                time.sleep(0.3)

            elif keyboard.is_pressed('q'):
                print("Quitting...")
                break

            if not any(keyboard.is_pressed(k) for k in ('g', 's', 'l', 'r')):
                current_command = None

            time.sleep(0.02)

        print("Control loop exited.")
        return {"success": True}

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    finally:
        print("Cleaning up...")
        if robot is not None:
            try:
                robot.move_to_rest_position(wait_time=3.0, speed=15)
            except:
                pass
            try:
                for jid in robot.joint_ids:
                    ctrl.disable_torque(jid)
            except:
                pass
        print("Done.")