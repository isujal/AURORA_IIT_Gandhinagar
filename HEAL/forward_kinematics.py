#!/usr/bin/env python3
"""
Task 01 – Relative Joint Angle Reaching Challenge


Overview:
---------
This script evaluates a user’s ability to position a 6-DOF robot arm to match
randomly generated joint targets using either manual teleoperation or automatic control.

Key Features:
-------------
1. Generates a random target joint configuration (in degrees).
2. User moves the robot to match the target within a defined tolerance.
3. Computes joint-wise signed angular error and a performance score.
4. Offers optional automatic movement to the target pose using velocity control.
5. Finally returns the robot to a predefined home pose (all joints at 0°).

Usage Note:
-----------
- Joint states are subscribed from the "/joint_states" topic.
- Velocity commands are published to "/velocity_controller/command".
- Manual control should be run in a separate terminal:
    `cd utils && python3 joint_teleop_heal.py`
"""

import os, sys, time
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Import trajectory planner from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.TRAJECTORY_PLANNERS.trajectory_planners import TrajectoryPlanner

# ----------------------------- Configuration -----------------------------
# TODO 1 : How many joints does HEAL have?
NUM_JOINTS =  6
JOINT_LIMITS_DEG = [(-90, 90), (-45, 45), (-30, 60), (-90, 90), (0, 90), (-60, 60)]
ANGLE_TOLERANCE = 5.0            # Allowed error per joint in degrees
DT = 0.001                       # Time step for velocity streaming
TRAJ_TIME = 5.0                  # Duration of automatic motion

# ----------------------------- Global State ------------------------------
current_joint_rad = None  # Buffer to store the latest joint state


# --------------------------- Callback Function ---------------------------
def joint_state_cb(msg):
    """Callback: Extract and store current joint angles (in radians) from the first NUM_JOINTS joints."""
    global current_joint_rad
    # TODO 2 : Store first NUM_JOINTS positions from msg as a numpy array
    # Hint: msg.position contains all joint angles in radians
    current_joint_rad = np.array(msg.position)


# --------------------------- Utility Functions ---------------------------
def wait_for_joint_state(timeout=5.0):
    """Blocks until joint state is received or timeout expires."""
    start = rospy.Time.now()
    rate = rospy.Rate(200)
    while current_joint_rad is None and (rospy.Time.now() - start).to_sec() < timeout:
        rate.sleep()
    return current_joint_rad


def get_current_joint_angles_deg():
    """Returns the current joint angles (in degrees). Falls back to zeros if timeout occurs."""
    js = wait_for_joint_state()
    if js is None:
        rospy.logwarn("⚠️ No joint state received — defaulting to zeros.")
        return np.zeros(NUM_JOINTS)
    return np.degrees(js)


def generate_target_angles():
    """Generates a random target configuration within joint limits."""
    low_limits = np.array([limit[0] for limit in JOINT_LIMITS_DEG])
    high_limits = np.array([limit[1] for limit in JOINT_LIMITS_DEG]) + 1
    return np.random.randint(low_limits, high_limits)

    # return np.array([
    #     # TODO 3 : Generate a random integer angle for each joint within its limits
    #     # Hint: np.random.randint(low, high + 1)
    #     np.random.randint([-90,-45,-30,-90,0,-60],[90,45,60,90,90,60])
    # ])


def evaluate_performance(target_deg, achieved_deg):
    """
    Compares achieved vs. target joint angles and calculates:
    - Success status (all joints within ANGLE_TOLERANCE)
    - Signed errors
    - Score per joint based on % error over full joint range
    """
    signed_err = achieved_deg - target_deg
    abs_err = np.abs(signed_err)

    score_per_joint = []
    success = True

    for i in range(NUM_JOINTS):
        joint_range = JOINT_LIMITS_DEG[i][1] - JOINT_LIMITS_DEG[i][0]
        percent_error = (abs_err[i] / joint_range) * 100

        if abs_err[i] > ANGLE_TOLERANCE:
            success = False

        # Scoring by % error
        if percent_error > 50:
            score = 0
        elif percent_error > 30:
            score = 3
        elif percent_error > 20:
            score = 5
        elif percent_error > 10:
            score = 7
        else:
            score = 10

        score_per_joint.append(score)

    return success, signed_err, score_per_joint


def move_robot_to_joint_angles(target_deg):
    """
    Plans and streams joint velocity commands to move the robot from its
    current configuration to the target joint angles.
    """
    # TODO 4 : Enter the ROS topic name to publish velocity commands
    pub = rospy.Publisher("/joint_states", Float64MultiArray, queue_size=10)

    # Get starting pose from joint state
    q_start_rad = wait_for_joint_state()
    if q_start_rad is None:
        rospy.logerr("❌ Joint state unavailable. Aborting motion.")
        return

    # Convert target to radians and plan
    # TODO 5 : Convert target_deg from degrees to radians
    q_goal_rad = np.deg2rad(target_deg)
    # TODO 6 : Create a TrajectoryPlanner instance
    planner = TrajectoryPlanner()
    _, vel_traj, _ = planner.quintic_joint_trajectory(q_start_rad, q_goal_rad, TRAJ_TIME,DT)

    # Stream velocity commands
    rate = rospy.Rate(1.0 / DT)
    msg = Float64MultiArray()
    rospy.loginfo("🚀 Executing velocity trajectory...")
    for vel in vel_traj:
        if rospy.is_shutdown():
            return
        msg.data = vel.tolist()
        pub.publish(msg)
        rate.sleep()

    # Send final zero velocity command
    msg.data = [0.0] * NUM_JOINTS
    pub.publish(msg)
    rospy.loginfo("✅ Target configuration reached.")


def move_to_home():
    """
    Moves the robot to the 'home' pose — all joints at 0° — after pressing Enter.
    """
    input("\n🏠 Press [Enter] to move the robot to home position (0° for all joints)...")
    rospy.loginfo("🔄 Returning to home position...")
    rospy.sleep(1)
    move_robot_to_joint_angles(np.zeros(NUM_JOINTS))


# -------------------------- Display Utilities --------------------------
def print_target_angles(target):
    print("\n🎯 Target Joint Angles:")
    for i, a in enumerate(target):
        print(f"   Joint {i+1}: {a:+.0f}°")


def print_eval(success, signed_err, score_per_joint):
    print("\n📊 Performance Summary:")
    total_score = 0
    for i, (e, s) in enumerate(zip(signed_err, score_per_joint)):
        desc = "Overshoot ➡️" if e > 0 else "Undershoot ⬅️" if e < 0 else "Perfect 🎯"
        print(f"   - Joint {i+1}: Error = {abs(e):.2f}° ({desc}), Score = {s}/10")
        total_score += s
    print(f"\n🏅 Total Score: {total_score}/{NUM_JOINTS * 10}")
    print("\n🏆 SUCCESS!" if success else "\n❗ One or more joints exceeded tolerance.")


# ------------------------------ Main Execution ------------------------------
def main():
    rospy.init_node("relative_angle_task", anonymous=True)
    # TODO 7 : Enter the ROS topic name for joint states
    rospy.Subscriber("/joint_states", JointState, joint_state_cb)

    print("\n🔧 Relative Joint Angle Reaching Task Initialized")

    # Step 1: Generate and display target
    target_deg = generate_target_angles()
    print_target_angles(target_deg)

    # Step 2: Ask user to move robot manually via teleop
    input("\n🕹️ Use teleoperation to move robot to target, then press [Enter]...")

    # Step 3: Capture current pose and evaluate
    achieved_deg = get_current_joint_angles_deg()
    success, signed_err, scores = evaluate_performance(target_deg, achieved_deg)
    print_eval(success, signed_err, scores)

    # Step 4: Offer auto-move to target
    if input("\n🤖 Auto-move robot to target? (y/n): ").strip().lower() in ("y", "yes"):
        move_robot_to_joint_angles(target_deg)

    # Step 5: Return to home
    move_to_home()
    print("\n✅ Task complete. Robot is back at home. Exiting.\n")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
