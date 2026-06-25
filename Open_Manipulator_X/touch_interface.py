#!/usr/bin/env python3
# AB6_KIN_OMX_HW_005 — Touch Interface

import time
import numpy as np

# Note: Keeping your template's import, though the manual references OMXRobot [cite: 140]
from src.robot_core.open_manipulator_x import OpenManipulatorX

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True

# ============================================================
# VISIT SEQUENCE
# ============================================================

# Type your desired sequence here. 
# The numbers correspond to the target point (1 = first point, 2 = second point, etc.)
VISIT_SEQUENCE = [9,8,7,6,5,4,3,2,1]

# ============================================================
# TARGET COORDINATES
# ============================================================

TARGET_POINTS = [
    [0.25, 0.0802, 0.035],
    [0.235, 0.0489, 0.003],
    [0.242, -0.0120, 0.001],
    [0.255,-0.0760, 0.028],
    [0.233, 0.0903, 0.009],
    [0.262, 0.051, 0.0157],
    [0.2485, -0.0022, 0.017],
    [0.235, -0.048,0.012],
    [0.245, -0.087, 0.007]
]

TARGET_POINTS = [
    np.array(p, dtype=float)
    for p in TARGET_POINTS
]

# ============================================================
# WRIST ORIENTATION (JOINT 4) PER POINT
# ============================================================

# WHY NOT A FIXED "180 DEG" / "270 DEG" VALUE:
# Joint 4's hardware limit (manual, Table 1.4) is -1.79 rad to
# +2.04 rad (~ -102 deg to +117 deg). There is no joint-4 value
# that equals literal 180 deg or 270 deg -- the motor physically
# cannot get there. Commanding either one would just get clamped
# to the same +2.04 rad ceiling, which is why forward and downward
# ended up looking identical.
#
# WHY A FIXED j4 NUMBER DOESN'T WORK EITHER:
# IK is position-only [cite: 537] and returns whatever j2/j3/j4
# combination the DLS solver converged to from the *current* pose
# [cite: 529]. The same j4 value means a different *gripper*
# orientation depending on what j2 and j3 ended up at. That's the
# "multiple solutions" problem you're seeing -- it's not a bug in
# the solver, it's because j4's effect on the world is relative to
# j2+j3, not absolute.
#
# THE FIX:
# In this 4-DOF arm, links 2, 3, and 4 all tilt in the same plane,
# so the gripper's pitch relative to the ground is the *sum*:
#     gripper_pitch = j2 + j3 + j4
# To force the gripper to a specific real-world pitch, solve IK
# for position as before, then OVERWRITE j4 with whatever value
# makes that sum equal the pitch you want:
#     j4 = target_pitch - j2 - j3
# This guarantees the tool points where you want regardless of
# which j2/j3 IK happened to land on.
#
#   FORWARD_PITCH  = 0      -> gripper horizontal (wall points 1-4)
#   DOWNWARD_PITCH = -pi/2  -> gripper straight down (ground points 5-9)
#
# If the arm comes out pointing the opposite way on a test run,
# flip the sign on DOWNWARD_PITCH (try +pi/2 instead of -pi/2) --
# that's a one-line change here, nothing else needs to move.

FORWARD_PITCH = 0.0                 # gripper horizontal, points 1-4
DOWNWARD_PITCH = np.pi / 4         # gripper straight down, points 5-9

J4_MIN = 0
J4_MAX = np.pi/2

# 1-based point index -> desired gripper pitch (radians)
GRIPPER_PITCH = {}
for _i in range(1, 5):
    GRIPPER_PITCH[_i] = FORWARD_PITCH
for _i in range(5, 10):
   GRIPPER_PITCH[_i] = DOWNWARD_PITCH


def get_gripper_pitch(seq_number):
    """Look up the desired gripper pitch (radians) for a 1-based point number."""
    return GRIPPER_PITCH.get(seq_number, FORWARD_PITCH)


def apply_gripper_orientation(joint_angles, target_pitch):
    """
    Overwrite joint 4 (index 3) so the gripper's resulting pitch
    (j2 + j3 + j4) equals target_pitch, regardless of what j2/j3
    the IK solver landed on.

    Returns (joint_angles_with_fixed_j4, was_clamped). was_clamped
    is True if the required j4 fell outside the hardware limit and
    had to be capped -- meaning that exact pitch isn't reachable
    from this j2/j3 combination, and the actual orientation will
    be off from what was requested. This is surfaced to the caller
    instead of silently swallowed, since the previous version's
    silent clamp is exactly what made forward/down look identical.
    """
    j = list(joint_angles)
    required_j4 = target_pitch - j[1] - j[2]

    was_clamped = not (J4_MIN <= required_j4 <= J4_MAX)
    clamped_j4 = max(J4_MIN, min(J4_MAX, required_j4))

    j[3] = clamped_j4
    return j, was_clamped


# ============================================================
# HOME POSITION
# ============================================================

# Using the manual's pre-defined READY_POSITION as a safe home/base pose [cite: 43]
# Arm raised, ready for tasks: [0.0, -1.05, 0.35, 0.70] [cite: 43]
HOME_JOINT = [0.0, -1.05, 0.35, 0.70]


# ============================================================
# CONSTANTS
# ============================================================

APPROACH_LIFT = 0.05
WALL_RETREAT = 0.02
GROUND_Z_THRESH = 0.01

# ============================================================
# IK HELPER
# ============================================================

def solve_ik(robot, xyz):
    """
    Convert Cartesian coordinate into robot joint angles.
    """
    try:
        # Uses Damped Least Squares (DLS) numerical IK [cite: 529]
        # Returns np.ndarray of 4 joint angles in radians [cite: 538]
        q = robot.inverse_kinematics(xyz)
        return q.tolist()
    except ValueError as e:
        # Raises ValueError if IK does not converge [cite: 539]
        print("IK failed:", e)
        return None

# ============================================================
# TOUCH ONE TARGET
# ============================================================

def touch_target(robot, target_xyz, target_pitch):
    """
    Goal:
    1. Create safe approach coordinate.
    2. Solve IK for approach point.
    3. Move to approach point.
    4. Move to target point.
    5. Return to approach point.

    target_pitch: desired gripper pitch in radians (j2+j3+j4),
    applied to both the approach and touch poses by overwriting
    joint 4 after IK, since IK itself does not control orientation
    [cite: 537].
    """
    ground_target = (target_xyz[2] <= GROUND_Z_THRESH)

    if ground_target:
        # Create approach point above the target (Z-axis offset).
        approach_xyz = np.array([
            target_xyz[0], 
            target_xyz[1], 
            target_xyz[2] + APPROACH_LIFT
        ])
    else:
        # Create approach point away from the wall (X-axis offset).
        approach_xyz = np.array([
            target_xyz[0] - WALL_RETREAT, 
            target_xyz[1], 
            target_xyz[2]
        ])

    # Solve IK for approach and touch points
    approach_joint = solve_ik(robot, approach_xyz)
    touch_joint = solve_ik(robot, target_xyz)

    # Abort if either point is unreachable
    if approach_joint is None or touch_joint is None:
        print("Target is out of reach or IK failed.")
        return False

    # Fix the wrist (joint 4) so the gripper's actual pitch matches
    # target_pitch, for both the approach and touch poses, so the
    # wrist doesn't flip mid-motion.
    approach_joint, approach_clamped = apply_gripper_orientation(approach_joint, target_pitch)
    touch_joint, touch_clamped = apply_gripper_orientation(touch_joint, target_pitch)

    if approach_clamped or touch_clamped:
        print(
            "  WARNING: requested gripper pitch "
            f"({np.degrees(target_pitch):.0f} deg) is not fully "
            "reachable from this j2/j3 configuration -- joint 4 "
            "was clamped to its hardware limit. Actual orientation "
            "will be off from what was requested."
        )

    # Move to approach point 
    robot.move_to_joint(approach_joint, wait_time=2.0, speed=20)
    
    if stop_requested: return False
    
    # Move to touch point (slower speed for precision) 
    robot.move_to_joint(touch_joint, wait_time=2.0, speed=15)
    
    if stop_requested: return False
    
    # Return to approach point 
    robot.move_to_joint(approach_joint, wait_time=2.0, speed=20)

    return True

# ============================================================
# MAIN CHALLENGE
# ============================================================

def run_challenge(params):
    global stop_requested
    stop_requested = False

    robot = None

    try:
        print("Initializing OMX...")
        robot = OpenManipulatorX()

        if not robot.is_connected:
            return {"success": False, "error": "Robot not connected"}

        # Pings all four arm joints to verify they are responding [cite: 176]
        robot.ping_motors()

        # Velocity mode setup from your original template
        try:
            _set_velocity_mode(robot)
        except NameError:
            pass # Failsafe in case _set_velocity_mode is handled externally

        # ----------------------------------------------------
        # Move to Home Position (Base)
        # ----------------------------------------------------
        print("Moving to home pose...")
        robot.move_to_joint(HOME_JOINT, wait_time=3.0, speed=20)

        success_count = 0

        # ----------------------------------------------------
        # Visit Targets Based on VISIT_SEQUENCE
        # ----------------------------------------------------
        
        for seq_number in VISIT_SEQUENCE:
            if stop_requested:
                print("\nStop requested by user! Halting sequence.")
                break
                
            # Convert 1-based sequence number to 0-based array index
            index = seq_number - 1 
            
            # Make sure the requested index actually exists in our list
            if index < 0 or index >= len(TARGET_POINTS):
                print(f"Skipping invalid sequence number: {seq_number}")
                continue

            target = TARGET_POINTS[index]
            target_pitch = get_gripper_pitch(seq_number)

            print(
                f"\nSequence {seq_number} -> Target: "
                f"x={target[0]:.3f}, "
                f"y={target[1]:.3f}, "
                f"z={target[2]:.3f}, "
                f"gripper pitch={np.degrees(target_pitch):.0f} deg"
            )

            # Touch this target
            if touch_target(robot, target, target_pitch):
                success_count += 1

            # ----------------------------------------------------
            # Retract to Base Functionality
            # ----------------------------------------------------
            if not stop_requested:
                print(f"Retracting back to base position after Sequence {seq_number}...")
                robot.move_to_joint(HOME_JOINT, wait_time=2.5, speed=20)

        print(f"Completed {success_count} targets.")
        return {"success": True}

    except Exception as e:
        print("Error:", e)
        return {"success": False, "error": str(e)}

    finally:
        if robot is not None:
            
            # ----------------------------------------------------
            # ANTI-FALL SAFE SHUTDOWN
            # ----------------------------------------------------
            print("\nExecuting safe shutdown to prevent arm from dropping...")
            try:
                # Always fold into the low-energy rest pose before cutting torque.
                robot.move_to_rest_position(wait_time=6.0, speed=20)
            except Exception as safe_err:
                print("Could not reach rest position:", safe_err)

            # Stop all joints safely
            try:
                _stop_joints(robot)
            except NameError:
                pass
                
            # Disable torque only AFTER the arm is safely resting [cite: 1095]
            print("Disabling torque safely...")
            try:
                robot.disable_torque() # Use standard command to disable torque [cite: 218]
            except:
                pass