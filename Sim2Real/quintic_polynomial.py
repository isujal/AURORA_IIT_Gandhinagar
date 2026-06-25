# Challenge: Quintic Polynomial Trajectory - Straight Line
import numpy as np
import time

"""
CHALLENGE OVERVIEW:
------------------
In this challenge you will:
1. Define START and END points for a straight line in Cartesian space
2. Build a QuinticPolynomial trajectory for smooth motion
3. Use Inverse Kinematics at each timestep to convert
   Cartesian targets into joint angles
4. Animate the robot tracing the line in MuJoCo + real hardware

AVAILABLE OBJECTS (provided automatically):
  sim2real  - SimToReal instance (controls sim + hardware)
  robot     - Robot model with forward_kinematics() and inverse_kinematics()
  model     - MuJoCo model
  data      - MuJoCo data

KEY CONCEPT - Quintic Polynomial:
  p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
  By setting initial/final velocity and acceleration to zero,
  the motion is perfectly smooth with no jerk at start/end.
"""

# -------------------------------------------------
# Choose whether to also move the real hardware
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

    # Get current EE position (for Z reference)
    ee_body_id = model.body("gripper_link").id
    T_home = robot.forward_kinematics(np.zeros(3))
    z_ref = T_home[2, 3]

    # -------------------------------------------------
    # TODO 2: Define start and end points (in meters)
    # The robot will trace a straight line between them.
    # Both points must be within the reachable workspace.
    # -------------------------------------------------
    start_xy = [0.10, 0.1]   # 10 cm forward, 5 cm left
    end_xy   = [-0.10, 0.1]   # 10 cm forward, 15 cm left — straight line along Y axis

    start_point = np.array([start_xy[0], start_xy[1], z_ref])
    end_point   = np.array([end_xy[0],   end_xy[1],   z_ref])

    # -------------------------------------------------
    # TODO 3: Set the number of waypoints along the line
    # More waypoints = smoother path but more IK calls.
    # -------------------------------------------------
    num_waypoints = 2   # 10 waypoints → 9 segments, good balance of smoothness vs speed

    # -------------------------------------------------
    # TODO 4: Set the time for each trajectory segment (seconds)
    # Controls how fast the robot moves between waypoints.
    # -------------------------------------------------
    segment_time = 3.0   # 0.5s per segment → total motion ~4.5s, smooth and observable

    dt = 0.02  # simulation timestep (50 Hz)

    print("\n" + "=" * 50)
    print("  QUINTIC TRAJECTORY - STRAIGHT LINE")
    print("=" * 50)
    print(f"Start: {start_point}")
    print(f"End:   {end_point}")
    print(f"Waypoints: {num_waypoints}")
    print(f"Segment time: {segment_time}s")

    # Generate evenly-spaced waypoints
    waypoints = np.linspace(start_point, end_point, num_waypoints)

    # -------------------------------------------------
    # Build quintic trajectories for each segment
    # -------------------------------------------------
    trajectories = []
    for i in range(len(waypoints) - 1):
        p_start = waypoints[i]
        p_end = waypoints[i + 1]

        # TODO 5: Create a QuinticPolynomial for each axis (x, y, z)
        # The robot should start and stop smoothly (zero velocity and acceleration
        # at both endpoints).
        #
        # QuinticPolynomial(p0, pf, v0, vf, a0, af, T)
        #   p0, pf = start and end position for this axis
        #   v0, vf = start and end velocity = 0 (smooth stop/start)
        #   a0, af = start and end acceleration = 0 (smooth stop/start)
        #   T      = time duration of this segment
        traj_x = QuinticPolynomial(p_start[0], p_end[0], 0, 0, 0, 0, segment_time)
        traj_y = QuinticPolynomial(p_start[1], p_end[1], 0, 0, 0, 0, segment_time)
        traj_z = QuinticPolynomial(p_start[2], p_end[2], 0, 0, 0, 0, segment_time)

        trajectories.append((traj_x, traj_y, traj_z))

    print(f"Generated {len(trajectories)} trajectory segments")

    # -------------------------------------------------
    # Move to the first waypoint
    # -------------------------------------------------
    print("\nMoving to start position...")
    q_current = np.array([np.deg2rad(45), np.deg2rad(-30), 0.0])
    sim2real.set_joint_positions(q_current)
    time.sleep(0.3)

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

    # Wait 1 second at start position before beginning trajectory
    print("Reached start point — waiting 1 second before starting trajectory...")
    time.sleep(1.0)
    print("Starting trajectory now.")

    # -------------------------------------------------
    # Execute trajectory
    # -------------------------------------------------
    print(f"\n===== Tracing line ({len(trajectories)} segments) =====")

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

        # Progress
        if (seg_idx + 1) % 5 == 0 or seg_idx == 0 or seg_idx == len(trajectories) - 1:
            T_pos = robot.forward_kinematics(q_current)
            ee = T_pos[:3, 3]
            print(f"  Segment {seg_idx+1}/{len(trajectories)} - EE: [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]")

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
    print(f"  Target end point (m):      [{end_point[0]:.4f}, {end_point[1]:.4f}, {end_point[2]:.4f}]")
    print(f"  Position error (m):        {np.linalg.norm(end_point - fk_ee):.6f}")

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
        "start_point": start_point.tolist(),
        "end_point": end_point.tolist(),
        "final_ee": fk_ee.tolist(),
    }