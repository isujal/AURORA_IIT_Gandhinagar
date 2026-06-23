

# Challenge: Cartesian Teleoperation - Sim-to-Real
import numpy as np
import time
from pynput import keyboard

"""
CHALLENGE OVERVIEW:
------------------
In this challenge you will use your KEYBOARD to teleoperate the robot
end-effector in Cartesian space in REAL TIME. When you press a key,
the robot moves one step in that direction - simultaneously in MuJoCo
simulation AND on the real hardware.

AVAILABLE OBJECTS (provided automatically):
  sim2real  - SimToReal instance (controls sim + hardware)
  robot     - Robot model with forward_kinematics() and inverse_kinematics()
  model     - MuJoCo model
  data      - MuJoCo data

KEYBOARD CONTROLS:
  W : Move +Y (forward)     S : Move -Y (backward)
  A : Move -X (left)        D : Move +X (right)
  R : Move +Z (up)          F : Move -Z (down)
  H : Return to home position
  Q : Quit teleoperation

The pynput library captures keyboard presses even when the GUI
window is in focus. Each key press triggers one step of motion.
"""

# -------------------------------------------------
# TODO 1: Choose whether to also move the real hardware
# -------------------------------------------------
USE_HARDWARE = False  # True or False

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")


# =================================================
# IK helper (provided - do NOT modify)
# =================================================

def solve_ik(robot, target_pos, q_init):
    """Solve IK for a Cartesian target position."""
    q_sol = robot.inverse_kinematics(
        position=target_pos,
        orientation=np.eye(3),
        initial_guess=q_init,
        tolerance=1e-4,
        max_iter=100
    )
    return q_sol


def run_challenge(sim2real, robot, model, data):
    global stop_requested
    stop_requested = False

    if hasattr(sim2real, 'hardware_enabled'):
        sim2real.hardware_enabled = USE_HARDWARE
    print(f"Hardware enabled: {USE_HARDWARE}")

    # Get current EE position as starting point
    T_home = robot.forward_kinematics(np.zeros(3))
    ee_start = T_home[:3, 3].copy()

    print(f"\\nInitial EE position: "
          f"[{ee_start[0]*100:.1f}, {ee_start[1]*100:.1f}, {ee_start[2]*100:.1f}] cm")

    # -------------------------------------------------
    # TODO 2: Set the step size (in meters)
    # Each key press moves the EE by this amount.
    # -------------------------------------------------
    step_size = 0.05
    joint_step = 0.05  # ~3 deg per key press (in radians)


    # -------------------------------------------------
    # TODO 3: Define the direction map
    # Map each key character to a unit direction vector [x, y, z].
    # Keys: 'w'/'s' for Y axis, 'a'/'d' for X axis, 'r'/'f' for Z axis.
    # -------------------------------------------------
    direction_map = {
        'w': np.array([ 0,  1,  0]),   # +Y
        's': np.array([ 0, -1,  0]),   # -Y
        'a': np.array([-1,  0,  0]),   # -X
        'd': np.array([ 1,  0,  0]),   # +X
        'r': np.array([ 0,  0,  1]),   # +Z
        'f': np.array([ 0,  0, -1]),   # -Z
    }

    # Move to home first
    print("\\nMoving to home position...")
    q_current = np.zeros(3)
    sim2real.set_joint_positions(q_current)
    time.sleep(0.5)

    target_pos = ee_start.copy()
    positions_visited = [target_pos.copy()]
    commands_executed = []
    step_count = 0
    quit_flag = False

    print("\\n" + "=" * 55)
    print("  CARTESIAN TELEOPERATION - KEYBOARD CONTROL")
    print("=" * 55)
    print(f"  Step size: {step_size*100:.1f} cm")
    print("  Controls: W/A/S/D = XY plane | R/F = Z axis")
    print("  H = Home | Q = Quit")
    print("=" * 55)
    print("\\n  Press keys on your keyboard to move the robot...\\n")

    # -------------------------------------------------
    # TODO 4: Implement the keyboard callback
    # This function is called every time a key is pressed.
    # -------------------------------------------------
    def on_key_press(key):
    nonlocal target_pos, q_current, step_count, quit_flag

    try:
        k = key.char.lower()
    except AttributeError:
        return

    if k in direction_map:
        # Cartesian step
        target_pos = target_pos + direction_map[k] * step_size
        q_current  = solve_ik(robot, target_pos, q_current)
        sim2real.set_joint_positions(q_current)
        T   = robot.forward_kinematics(q_current)
        ee  = T[:3, 3]
        err = np.linalg.norm(ee - target_pos)
        step_count += 1
        positions_visited.append(ee.copy())
        commands_executed.append(k)
        print(f"  Step {step_count:03d}: [{k.upper()}] "
              f"EE=({ee[0]*100:.1f}, {ee[1]*100:.1f}, {ee[2]*100:.1f})cm  "
              f"IK_err={err*1000:.2f}mm  "
              f"joints={np.rad2deg(q_current).round(1)}°")

    elif k == 'x':
        # X key — toggle between W/S and A/D axis lock (just a nudge in +X+Y)
        # Per your request: pressing X moves EE diagonally +X +Y one step
        target_pos = target_pos + np.array([1, 1, 0]) * step_size / np.sqrt(2)
        q_current  = solve_ik(robot, target_pos, q_current)
        sim2real.set_joint_positions(q_current)
        T   = robot.forward_kinematics(q_current)
        ee  = T[:3, 3]
        step_count += 1
        positions_visited.append(ee.copy())
        commands_executed.append(k)
        print(f"  Step {step_count:03d}: [X] diagonal +X+Y "
              f"EE=({ee[0]*100:.1f}, {ee[1]*100:.1f}, {ee[2]*100:.1f})cm  "
              f"joints={np.rad2deg(q_current).round(1)}°")

    elif k == 'z':
        # Z key — INCREMENT all joint positions by joint_step
        # q_new = q_old + [joint_step, joint_step, joint_step]
        q_current = q_current + np.array([joint_step, joint_step, joint_step])
        sim2real.set_joint_positions(q_current)
        T  = robot.forward_kinematics(q_current)
        ee = T[:3, 3]
        target_pos = ee.copy()   # sync target to actual EE
        step_count += 1
        positions_visited.append(ee.copy())
        commands_executed.append(k)
        print(f"  Step {step_count:03d}: [Z] joints++ "
              f"joints={np.rad2deg(q_current).round(1)}°  "
              f"EE=({ee[0]*100:.1f}, {ee[1]*100:.1f}, {ee[2]*100:.1f})cm")

    elif k == 'v':
        # V key — DECREMENT all joint positions by joint_step
        # q_new = q_old - [joint_step, joint_step, joint_step]
        q_current = q_current - np.array([joint_step, joint_step, joint_step])
        sim2real.set_joint_positions(q_current)
        T  = robot.forward_kinematics(q_current)
        ee = T[:3, 3]
        target_pos = ee.copy()   # sync target to actual EE
        step_count += 1
        positions_visited.append(ee.copy())
        commands_executed.append(k)
        print(f"  Step {step_count:03d}: [V] joints-- "
              f"joints={np.rad2deg(q_current).round(1)}°  "
              f"EE=({ee[0]*100:.1f}, {ee[1]*100:.1f}, {ee[2]*100:.1f})cm")

    elif k == 'c':
        # C key — go to HOME (same as H)
        target_pos = ee_start.copy()
        q_current  = np.zeros(3)
        sim2real.set_joint_positions(q_current)
        T  = robot.forward_kinematics(q_current)
        ee = T[:3, 3]
        print(f"  [C] HOME — EE: ({ee[0]*100:.1f}, {ee[1]*100:.1f}, {ee[2]*100:.1f})cm  "
              f"joints={np.rad2deg(q_current).round(1)}°")

    elif k == 'h':
        target_pos = ee_start.copy()
        q_current  = np.zeros(3)
        sim2real.set_joint_positions(q_current)
        T  = robot.forward_kinematics(q_current)
        ee = T[:3, 3]
        print(f"  [H] HOME — EE: ({ee[0]*100:.1f}, {ee[1]*100:.1f}, {ee[2]*100:.1f})cm")

    elif k == 'q':
        quit_flag = True
        print("  Quit requested.")

    # -------------------------------------------------
    # TODO 5: Start the keyboard listener and wait
    # Use pynput.keyboard.Listener to listen for key presses.
    # Keep the loop running until quit_flag or stop_requested.
    # -------------------------------------------------
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()
    while not quit_flag and not stop_requested:
        time.sleep(0.05)
    listener.stop()

    # -------------------------------------------------
    # Final results
    # -------------------------------------------------
    T_final = robot.forward_kinematics(q_current)
    final_ee = T_final[:3, 3]

    total_dist = sum(
        np.linalg.norm(positions_visited[j] - positions_visited[j-1])
        for j in range(1, len(positions_visited))
    )

    print("\\n" + "=" * 55)
    print("           FINAL RESULTS")
    print("=" * 55)
    print(f"  Steps executed:  {step_count}")
    print(f"  Commands:        {''.join(commands_executed)}")
    print(f"  Start EE (cm):   [{ee_start[0]*100:.1f}, {ee_start[1]*100:.1f}, {ee_start[2]*100:.1f}]")
    print(f"  Final EE (cm):   [{final_ee[0]*100:.1f}, {final_ee[1]*100:.1f}, {final_ee[2]*100:.1f}]")
    print(f"  Final joints:    {np.rad2deg(q_current).round(1)} deg")
    print(f"  Total distance:  {total_dist*100:.1f} cm")
    print("=" * 55)

    return {
        "completed_successfully": not stop_requested,
        "commands": ''.join(commands_executed),
        "num_steps": step_count,
        "step_size_m": float(step_size),
        "start_ee": ee_start.tolist(),
        "final_ee": final_ee.tolist(),
        "total_distance_m": float(total_dist),
    }

