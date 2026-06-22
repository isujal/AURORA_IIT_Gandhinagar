# -----------------------------------------------------------------------------
# Part 1 - Inverse Kinematics Solver
# -----------------------------------------------------------------------------
# Provided automatically:
#   robot  - Robot model with inverse_kinematics() and forward_kinematics()
#
# API:
#   robot.inverse_kinematics(position, orientation, initial_guess,
#                             tolerance=1e-6, max_iter=100)
#     -> np.array [q1, q2, q3]  (radians)
#
#   robot.forward_kinematics(joint_angles)
#     -> 4x4 homogeneous transform T  (T[:3,3] = EE position)
#
# Return a dict with at least "q_solution" so Part 2 can reuse your answer.
# -----------------------------------------------------------------------------
import numpy as np

stop_requested = False


def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")


def run_challenge(robot):
    global stop_requested
    stop_requested = False

    # -- TODO: Set target end-effector POSITION (metres) ----------------------
    # Choose a point [x, y, z] within the robot's reachable workspace.
    target_position = np.array([0.15, 0.08, 0])

    # -- TODO: Set target ORIENTATION (3x3 rotation matrix) -------------------
    # A 3x3 rotation matrix describes the desired orientation of the gripper.
    target_orientation = np.eye(3)

    # -- TODO: Initial guess for IK solver (radians) --------------------------
    # Provide a starting joint configuration for the IK solver.
    initial_guess = np.array([0.5, 0.3, 0.2])

    # -- Solve IK -------------------------------------------------------------
    print("===== INVERSE KINEMATICS =====")
    print(f"Target position    : {target_position}")
    print(f"Initial guess (deg): {np.rad2deg(initial_guess).round(2)}")

    # TODO: call robot.inverse_kinematics() with the correct arguments
    q_solution = robot.inverse_kinematics(
        target_position,
        target_orientation,
        initial_guess
    )
    print(f"\n===== IK Solution =====")
    print(f"Joint angles (rad) : {q_solution}")
    print(f"Joint angles (deg) : {np.rad2deg(q_solution).round(2)}")

    # -- Verify with FK --------------------------------------------------------
    T_verify      = robot.forward_kinematics(q_solution)
    ee_pos_verify = T_verify[:3, 3]
    pos_error     = np.linalg.norm(target_position - ee_pos_verify)

    print(f"\n===== FK Verification =====")
    print(f"Achieved position  : {ee_pos_verify}")
    print(f"Position error     : {pos_error:.8f} m")
    if pos_error < 0.01:
        print("IK solution within 1 cm tolerance - ready for Part 2!")
    else:
        print("IK solution outside tolerance - refine your inputs.")

    return {
        "q_solution":        q_solution.tolist(),
        "target_position":   target_position.tolist(),
        "achieved_position": ee_pos_verify.tolist(),
        "position_error":    float(pos_error),
        "success":           bool(pos_error < 0.01),
    }
