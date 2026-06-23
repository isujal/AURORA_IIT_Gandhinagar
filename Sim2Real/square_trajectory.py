# Challenge: Quintic Polynomial Trajectory - Square
import numpy as np
import time

"""
CHALLENGE OVERVIEW:
------------------
In this challenge you will:
1. Define 4 CORNERS of a square in Cartesian space
2. Generate waypoints along each edge
3. Build QuinticPolynomial trajectories for smooth motion
4. Use IK at each timestep to convert Cartesian targets to joint angles
5. Animate the robot tracing a square in MuJoCo + real hardware

AVAILABLE OBJECTS (provided automatically):
  sim2real  - SimToReal instance (controls sim + hardware)
  robot     - Robot model with forward_kinematics() and inverse_kinematics()
  model     - MuJoCo model
  data      - MuJoCo data

KEY CONCEPT - Square Trajectory:
  The square is made of 4 edges. Each edge is divided into waypoints.
  Between consecutive waypoints we use a quintic polynomial so the
  robot moves smoothly without jerk.
"""

# -------------------------------------------------
# TODO 1: Choose whether to also move the real hardware
# -------------------------------------------------
USE_HARDWARE = True  # True or False

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")


# =================================================
# QuinticPolynomial class (provided - do NOT modify)
# =================================================

class QuinticPolynomial:
    """5th-order polynomial: smooth position, velocity, acceleration."""

    def __init__(self, p0, pf, v0, vf, a0, af, T):
        self.T = T
        self.a0_coeff = p0
        self.a1_coeff = v0
        self.a2_coeff = a0 / 2.0

        A = np.array([
            [T**3,   T**4,    T**5],
            [3*T**2, 4*T**3,  5*T**4],
            [6*T,    12*T**2, 20*T**3]
        ])
        b = np.array([
            pf - p0 - v0*T - (a0/2.0)*T**2,
            vf - v0 - a0*T,
            af - a0
        ])
        x = np.linalg.solve(A, b)
        self.a3_coeff = x[0]
        self.a4_coeff = x[1]
        self.a5_coeff = x[2]

    def position(self, t):
        return (self.a0_coeff +
                self.a1_coeff * t +
                self.a2_coeff * t**2 +
                self.a3_coeff * t**3 +
                self.a4_coeff * t**4 +
                self.a5_coeff * t**5)


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

    # Get Z reference from home position FK
    ee_body_id = model.body("gripper_link").id
    T_home = robot.forward_kinematics(np.zeros(3))
    z_ref = T_home[2, 3]

    # -------------------------------------------------
    # TODO 2: Define the 4 corners of the square (in meters)
    # The robot will trace edges: C1 -> C2 -> C3 -> C4 -> C1
    # All corners must be within the reachable workspace.
    # -------------------------------------------------
    corners = [
        np.array([0.1, 0.075, z_ref]),   # Corner 1 (Top-Left based on your line coordinate)
        np.array([0, 0.075, z_ref]),  # Corner 2 (Top-Right)
        np.array([0.0, 0.125, z_ref]), # Corner 3 (Bottom-Right)
        np.array([0.1, 0.125, z_ref]),  # Corner 4 (Bottom-Left)
    ]

    # corners = [
        # np.array([0.15, 0.125, z_ref]),   # Corner 1 (Top-Left based on your line coordinate)
        # np.array([0, 0.125, z_ref]),  # Corner 2 (Top-Right)
        # np.array([0.0, 0.175, z_ref]), # Corner 3 (Bottom-Right)
        # np.array([0.15, 0.175, z_ref]),  # Corner 4 (Bottom-Left)
    # ]
    # Close the square by returning to Corner 1
    corners.append(corners[0].copy())

    # -------------------------------------------------
    # TODO 3: Set points per edge and segment time
    # points_per_edge controls smoothness; segment_time controls speed.
    # -------------------------------------------------
    points_per_edge = 1  # Generates fine discrete step segments per edge
    segment_time = 2.5   # Controls how long it takes to traverse between consecutive waypoints

    dt = 0.02  # simulation timestep (50 Hz)

    print("\n" + "=" * 50)
    print("  QUINTIC TRAJECTORY - SQUARE")
    print("=" * 50)
    print("Corners:")
    for i, c in enumerate(corners[:-1]):
        print(f"  C{i+1}: ({c[0]*100:.1f}cm, {c[1]*100:.1f}cm)")

    # Generate waypoints along each edge
    waypoints = []
    for i in range(len(corners) - 1):
        edge_pts = np.linspace(corners[i], corners[i+1], points_per_edge, endpoint=False)
        waypoints.extend(edge_pts)
    waypoints.append(corners[-1])
    waypoints = np.array(waypoints)
    print(f"Total waypoints: {len(waypoints)}")

    # -------------------------------------------------
    # Build quintic trajectories for each segment
    # -------------------------------------------------
    trajectories = []
    for i in range(len(waypoints) - 1):
        p_start = waypoints[i]
        p_end = waypoints[i + 1]

        # TODO 4: Create a QuinticPolynomial for each axis
        # The robot should start and stop smoothly at each waypoint.
        traj_x = QuinticPolynomial(p_start[0], p_end[0], 0, 0, 0, 0, segment_time)
        traj_y = QuinticPolynomial(p_start[1], p_end[1], 0, 0, 0, 0, segment_time)
        traj_z = QuinticPolynomial(p_start[2], p_end[2], 0, 0, 0, 0, segment_time)

        trajectories.append((traj_x, traj_y, traj_z))

    print(f"Generated {len(trajectories)} trajectory segments")

    # -------------------------------------------------
    # Move to the first waypoint
    # -------------------------------------------------
    print("\nMoving to start position...")
    q_current = np.array([0.15, 0.125, z_ref])
    sim2real.set_joint_positions(q_current)
    time.sleep(1.5)

    q_start = solve_ik(robot, waypoints[0], q_current)

    move_steps = 100
    for i in range(move_steps):
        if stop_requested:
            return {"completed_successfully": False}
        alpha = i / (move_steps - 1)
        q = (1 - alpha) * q_current + alpha * q_start
        sim2real.set_joint_positions(q)
        time.sleep(0.01)

    q_current = q_start
    T_pos = robot.forward_kinematics(q_current)
    print(f"At start - EE: [{T_pos[0,3]:.4f}, {T_pos[1,3]:.4f}, {T_pos[2,3]:.4f}]")

    # -------------------------------------------------
    # Execute trajectory
    # -------------------------------------------------
    print(f"\n===== Tracing square ({len(trajectories)} segments) =====")

    corner_indices = [points_per_edge * i - 1 for i in range(1, 5)]

    for seg_idx, (traj_x, traj_y, traj_z) in enumerate(trajectories):
        t = 0
        while t <= segment_time:
            if stop_requested:
                print("Stopped by user.")
                return {"completed_successfully": False}

            target_pos = np.array([
                traj_x.position(t),
                traj_y.position(t),
                traj_z.position(t)
            ])

            q_sol = solve_ik(robot, target_pos, q_current)
            q_current = q_sol
            sim2real.set_joint_positions(q_current)

            time.sleep(dt)
            t += dt

        # Progress - report at corners
        T_pos = robot.forward_kinematics(q_current)
        ee = T_pos[:3, 3]

        if seg_idx in corner_indices:
            corner_num = corner_indices.index(seg_idx) + 2
            if corner_num <= 4:
                print(f"  Corner {corner_num} reached - EE: [{ee[0]:.4f}, {ee[1]:.4f}]")
            else:
                print(f"  Back to Corner 1 - EE: [{ee[0]:.4f}, {ee[1]:.4f}]")
        elif (seg_idx + 1) % 5 == 0:
            print(f"  Segment {seg_idx+1}/{len(trajectories)} - EE: [{ee[0]:.4f}, {ee[1]:.4f}]")

    # -------------------------------------------------
    # Final results
    # -------------------------------------------------
    print("\n" + "=" * 50)
    print("           FINAL RESULTS")
    print("=" * 50)

    print("\n--- MuJoCo Simulation ---")
    T_final = robot.forward_kinematics(q_current)
    fk_ee = T_final[:3, 3]
    print(f"  Final joint angles (deg): {np.rad2deg(q_current)}")
    print(f"  EE position (m):          [{fk_ee[0]:.4f}, {fk_ee[1]:.4f}, {fk_ee[2]:.4f}]")
    print(f"  Target (Corner 1) (m):     [{corners[0][0]:.4f}, {corners[0][1]:.4f}, {corners[0][2]:.4f}]")
    print(f"  Closure error (m):         {np.linalg.norm(corners[0] - fk_ee):.6f}")

    if USE_HARDWARE:
        print("\n--- Hardware ---")
        hw_pos = sim2real.get_hardware_positions()
        if hw_pos is not None:
            print(f"  Joint angles (deg): {np.rad2deg(hw_pos)}")
            T_hw = robot.forward_kinematics(hw_pos)
            hw_ee = T_hw[:3, 3]
            print(f"  EE position (m):    [{hw_ee[0]:.4f}, {hw_ee[1]:.4f}, {hw_ee[2]:.4f}]")

    print("\n" + "=" * 50)
    print("           Trajectory complete")
    print("=" * 50)

    return {
        "completed_successfully": not stop_requested,
        "corners": [c.tolist() for c in corners[:-1]],
        "final_ee": fk_ee.tolist(),
        "closure_error": float(np.linalg.norm(corners[0] - fk_ee)),
    }