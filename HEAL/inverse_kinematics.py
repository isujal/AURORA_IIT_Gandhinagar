#!/usr/bin/env python3
"""
Task 02 – Inverse Kinematics Cartesian Reaching Challenge
this is the task which is needed to be analyzed 

Overview:
---------
This script evaluates a user's ability to understand and specify a 3D Cartesian
target point for the HEAL robot end-effector. The robot automatically moves to
the target using IK (Levenberg–Marquardt) solved from live joint state feedback,
followed by a smooth quintic velocity trajectory.

Key Features:
-------------
1. User provides (or a random one is generated) a Cartesian target [x, y, z].
2. IK is solved using KDL's ChainIkSolverPos_LMA from the current joint state.
3. Robot executes a smooth quintic velocity profile to reach the target.
4. Forward kinematics verifies the final achieved end-effector position.
5. A position error score is computed and displayed.
6. Robot returns to home (all joints at 0°) after evaluation.
low_limits = np.array([limit[0] for limit in JOINT_LIMITS_DEG])
    high_limits = np.array([limit[1] for limit in JOINT_LIMITS_DEG]) + 1
    return np.random.randint(low_limits, high_limits)
Usage:
------
    python3 02_ik_cartesian_reach_solution.py

Dependencies:
-------------
- ROS Noetic
- PyKDL, kdl_parser_py, urdf_parser_py
- utils/TRAJECTORY_PLANNERS/trajectory_planners.py (quintic planner)
- /robot_description param must be loaded
- /joint_states topic must be publishing
- /velocity_controller/command topic must have an active controller
"""

import os
import sys
import time
import rospy
import numpy as np
import PyKDL as kdl
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam

# Import trajectory planner from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.TRAJECTORY_PLANNERS.trajectory_planners import TrajectoryPlanner

# ----------------------------- Configuration -----------------------------
NUM_JOINTS    = 6
BASE_LINK     = "base_link"
TIP_LINK      = "tool_ff"
TRAJ_TIME     = 4.0       # Duration of motion in seconds
DT            = 0.001     # Time step for velocity streaming (s)
POS_TOLERANCE = 0.01      # Acceptable position error in meters (1 cm)

# Reachable workspace bounds for random target generation (in meters)
# Tune these to match the actual HEAL workspace
WORKSPACE_BOUNDS = {
    "x": (-0.40,  0.40),
    "y": ( 0.25,  0.60),
    "z": ( 0.10,  0.60),
}

# ----------------------------- Global State ------------------------------
current_joint_rad = None   # Latest joint positions from /joint_states


# --------------------------- Callback -----------------------------------------
def joint_state_cb(msg):
    """Store the latest joint angles (rad) from the first NUM_JOINTS entries."""
    global current_joint_rad
    current_joint_rad = np.array(msg.position[:NUM_JOINTS])


# --------------------------- Joint State Utilities ----------------------------
def wait_for_joint_state(timeout=5.0):
    """Block until a joint state message arrives or timeout expires."""
    start = rospy.Time.now()
    rate  = rospy.Rate(200)
    while current_joint_rad is None and (rospy.Time.now() - start).to_sec() < timeout:
        rate.sleep()
    return current_joint_rad


def get_current_joints():
    """Return current joint angles (rad). Falls back to zeros on timeout."""
    js = wait_for_joint_state()
    if js is None:
        rospy.logwarn("⚠️  No joint state received — defaulting to zeros.")
        return np.zeros(NUM_JOINTS)
    return js.copy()


# --------------------------- KDL Setup ----------------------------------------
def build_kdl_chain():
    """
    Load URDF from parameter server and build the KDL chain from
    BASE_LINK → TIP_LINK. Returns (chain, fk_solver, ik_solver).
    """
    if not rospy.has_param("/robot_description"):
        rospy.logerr("❌ '/robot_description' param not found. Is the URDF loaded?")
        sys.exit(1)

    success, tree = treeFromParam("/robot_description")
    if not success:
        rospy.logerr("❌ Failed to parse KDL tree from URDF.")
        sys.exit(1)

    chain      = tree.getChain(BASE_LINK, TIP_LINK)
    fk_solver  = kdl.ChainFkSolverPos_recursive(chain)
    # TODO 1 : Create the IK solver using Levenberg-Marquardt method
    # Hint: kdl.ChainIkSolverPos_LMA(chain)
    ik_solver  = kdl.ChainIkSolverPos_LMA(chain)

    rospy.loginfo("✅ KDL chain loaded: %s → %s  (%d joints)",
                  BASE_LINK, TIP_LINK, chain.getNrOfJoints())
    return chain, fk_solver, ik_solver


# --------------------------- FK Helper ----------------------------------------
def forward_kinematics(fk_solver, q_np):
    """
    Run forward kinematics for joint array q_np (numpy).
    Returns a kdl.Frame representing the end-effector pose.
    """
    n  = len(q_np)
    q  = kdl.JntArray(n)
    for i in range(n):
        q[i] = q_np[i]
    frame = kdl.Frame()
    fk_solver.JntToCart(q, frame)
    return frame


def frame_to_xyz(frame):
    """Extract (x, y, z) from a kdl.Frame as a numpy array."""
    return np.array([frame.p.x(), frame.p.y(), frame.p.z()])


# --------------------------- IK Helper ----------------------------------------
def solve_ik(ik_solver, n_joints, q_seed_np, target_frame):
    """
    Attempt IK from q_seed_np towards target_frame.
    Returns joint angles (numpy array) on success, or None on failure.
    """
    q_seed = kdl.JntArray(n_joints)
    for i in range(n_joints):
        q_seed[i] = q_seed_np[i]

    # TODO 2: Create an output kdl.JntArray
    q_out  = kdl.JntArray(n_joints)
    result = ik_solver.CartToJnt(q_seed, target_frame, q_out)

    if result >= 0:
        return np.array([q_out[i] for i in range(n_joints)])
    return None


# --------------------------- Target Generation --------------------------------
def generate_random_target():
    """Sample a random reachable Cartesian target within WORKSPACE_BOUNDS."""
    x = np.random.uniform(*WORKSPACE_BOUNDS["x"])
    y = np.random.uniform(*WORKSPACE_BOUNDS["y"])
    z = np.random.uniform(*WORKSPACE_BOUNDS["z"])
    return np.array([x, y, z])


def make_target_frame(pos_xyz, quat_xyzw=None):
    """
    Build a kdl.Frame from a position [x,y,z] and optional quaternion [x,y,z,w].
    If no quaternion is given the orientation is set to identity (tool pointing up).
    """
    if quat_xyzw is None:
        rot = kdl.Rotation.Identity()
    else:
        qx, qy, qz, qw = quat_xyzw
        rot = kdl.Rotation.Quaternion(qx, qy, qz, qw)
    return kdl.Frame(rot, kdl.Vector(*pos_xyz))


# --------------------------- Motion Execution ---------------------------------
def move_to_joint_config(target_joints_rad):
    """
    Plan a quintic velocity trajectory from the current joint state to
    target_joints_rad and stream it at 1/DT Hz.
    """
    # TODO 3 : Enter the ROS topic name to publish velocity commands
    pub      = rospy.Publisher("/velocity_controller/command",
                               Float64MultiArray, queue_size=10)
    q_start  = get_current_joints()
    planner  = TrajectoryPlanner()

    _, vel_traj, _ = planner.quintic_joint_trajectory(
        q_start, target_joints_rad, TRAJ_TIME, DT
    )

    rate = rospy.Rate(1.0 / DT)
    msg  = Float64MultiArray()
    rospy.loginfo("🚀 Executing trajectory (%d steps)...", len(vel_traj))

    for vel in vel_traj:
        if rospy.is_shutdown():
            return
        msg.data = vel.tolist()
        pub.publish(msg)
        rate.sleep()

    # Stop robot
    msg.data = [0.0] * NUM_JOINTS
    pub.publish(msg)
    rospy.loginfo("✅ Motion complete.")


def move_to_home():
    """Return all joints to 0° (home configuration)."""
    input("\n🏠 Press [Enter] to return robot to home position (0° all joints)...")
    rospy.loginfo("🔄 Moving to home...")
    rospy.sleep(0.5)
    move_to_joint_config(np.zeros(NUM_JOINTS))


# --------------------------- Evaluation ---------------------------------------
def evaluate_performance(target_xyz, achieved_xyz):
    """
    Compute position error and a score (0–100) based on closeness to target.

    Scoring:
        error ≤  1 cm  →  100 pts  (Excellent)
        error ≤  2 cm  →   80 pts  (Good)
        error ≤  5 cm  →   60 pts  (Acceptable)
        error ≤ 10 cm  →   40 pts  (Poor)
        error >  10 cm →    0 pts  (Failed)
    """
    error_vec  = achieved_xyz - target_xyz
    error_norm = np.linalg.norm(error_vec)
    success    = error_norm <= POS_TOLERANCE

    if   error_norm <= 0.01: score, grade = 100, "Excellent 🏆"
    elif error_norm <= 0.02: score, grade =  80, "Good ✅"
    elif error_norm <= 0.05: score, grade =  60, "Acceptable ⚠️"
    elif error_norm <= 0.10: score, grade =  40, "Poor ❗"
    else:                    score, grade =   0, "Failed ❌"

    return success, error_norm, error_vec, score, grade


# --------------------------- Display Utilities --------------------------------
def print_target(target_xyz):
    print("\n🎯 Target End-Effector Position:")
    print(f"   X : {target_xyz[0]:+.4f} m")
    print(f"   Y : {target_xyz[1]:+.4f} m")
    print(f"   Z : {target_xyz[2]:+.4f} m")


def print_current_ee(pos_xyz):
    print("\n📍 Current End-Effector Position (before move):")
    print(f"   X : {pos_xyz[0]:+.4f} m")
    print(f"   Y : {pos_xyz[1]:+.4f} m")
    print(f"   Z : {pos_xyz[2]:+.4f} m")


def print_eval(target_xyz, achieved_xyz, success, error_norm, error_vec, score, grade):
    print("\n📊 Performance Summary:")
    print(f"   Target   : ({target_xyz[0]:+.4f}, {target_xyz[1]:+.4f}, {target_xyz[2]:+.4f}) m")
    print(f"   Achieved : ({achieved_xyz[0]:+.4f}, {achieved_xyz[1]:+.4f}, {achieved_xyz[2]:+.4f}) m")
    print(f"   Error    : {error_norm*100:.2f} cm")
    print(f"   ΔX={error_vec[0]*100:+.2f} cm  "
          f"ΔY={error_vec[1]*100:+.2f} cm  "
          f"ΔZ={error_vec[2]*100:+.2f} cm")
    print(f"\n🏅 Score  : {score}/100  —  {grade}")
    if success:
        print("🏆 TARGET REACHED within tolerance!")
    else:
        print(f"❗ Outside tolerance (limit: {POS_TOLERANCE*100:.0f} cm)")


# ----------------------------- Main -------------------------------------------
def main():
    rospy.init_node("ik_cartesian_reach_task", anonymous=True)
    rospy.Subscriber("/joint_states", JointState, joint_state_cb)

    print("\n🦾 IK Cartesian Reaching Task Initialized")
    print("=" * 52)

    # TODO 4 : build kdl chain and unpack its 3 return values
    chain, fk_solver, ik_solver = build_kdl_chain()
    n_joints = chain.getNrOfJoints()

    # --- Wait for first joint state ---
    rospy.loginfo("⏳ Waiting for /joint_states...")
    q_init = get_current_joints()
    rospy.loginfo("✅ Joint state received.")

    # --- Show current end-effector position ---
    current_frame = forward_kinematics(fk_solver, q_init)
    current_xyz   = frame_to_xyz(current_frame)
    print_current_ee(current_xyz)
    
    # TODO 5 : generate random target to get the target position
    target_xyz = generate_random_target()

    print_target(target_xyz)

    # --- Build target frame (identity orientation — position only task) ---
    # Keep current orientation so only translation changes
    current_quat = current_frame.M.GetQuaternion()   # (x,y,z,w)
    target_frame  = make_target_frame(target_xyz, quat_xyzw=current_quat)

    # --- Solve IK ---
    rospy.loginfo("🔍 Solving IK...")
    q_solution = solve_ik(ik_solver, n_joints, q_init, target_frame)

    if q_solution is None:
        rospy.logerr("❌ IK failed — target may be outside the reachable workspace.")
        rospy.logerr("   Try a different target point.")
        return

    rospy.loginfo("✅ IK solution found:")
    for i, q in enumerate(q_solution):
        rospy.loginfo("   Joint %d: %.4f rad  (%.2f°)", i+1, q, np.degrees(q))

    # --- User confirmation before moving ---
    input("\n▶️  Press [Enter] to move the robot to the IK solution...")

    # --- Execute trajectory ---
    move_to_joint_config(q_solution)
    rospy.sleep(0.5)   # Settle

    # --- Evaluate achieved position via FK ---
    q_achieved    = get_current_joints()
    achieved_frame = forward_kinematics(fk_solver, q_achieved)
    achieved_xyz   = frame_to_xyz(achieved_frame)

    success, error_norm, error_vec, score, grade = evaluate_performance(
        target_xyz, achieved_xyz
    )
    print_eval(target_xyz, achieved_xyz, success, error_norm, error_vec, score, grade)

    # --- Retry option ---
    if not success:
        if input("\n🔁 Retry IK from achieved pose as seed? (y/n): ").strip().lower() in ("y", "yes"):
            rospy.loginfo("🔍 Re-solving IK from current pose...")
            q_retry = solve_ik(ik_solver, n_joints, q_achieved, target_frame)
            if q_retry is not None:
                move_to_joint_config(q_retry)
                rospy.sleep(0.5)
                q_final       = get_current_joints()
                final_frame    = forward_kinematics(fk_solver, q_final)
                final_xyz      = frame_to_xyz(final_frame)
                success, error_norm, error_vec, score, grade = evaluate_performance(
                    target_xyz, final_xyz
                )
                print("\n🔄 After retry:")
                print_eval(target_xyz, final_xyz, success, error_norm, error_vec, score, grade)
            else:
                rospy.logerr("❌ Retry IK also failed.")

    # --- Return home ---
    move_to_home()
    print("\n✅ Task complete. Robot is back at home. Exiting.\n")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
