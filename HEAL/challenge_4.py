#!/usr/bin/env python3
"""
Pick and Place Solution Script using Velocity Control and Grasping Actions



Description:
------------
This script implements a simple **pick-and-place** pipeline for the HEAL robotic arm. The robot follows a
sequence of predefined Cartesian poses, including optional grasp or release actions using ROS action topics.

Key Features:
-------------
- Uses inverse kinematics (KDL) to compute joint configurations from Cartesian targets.
- Trajectories are executed in joint velocity space using a quintic interpolator.
- Grasping and releasing actions are handled via ROS topics compatible with Addverb HEAL.

Dependencies:
-------------
- ROS Noetic
- PyKDL
- TrajectoryPlanner class (user-defined)
- addverb_cobot_msgs (custom HEAL message package)
"""

import os
import sys
import rospy
import numpy as np
import PyKDL as kdl
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromParam
from addverb_cobot_msgs.msg import GraspActionGoal, ReleaseActionGoal

# Add custom utils directory to import trajectory planner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.TRAJECTORY_PLANNERS.trajectory_planners import TrajectoryPlanner


# ---------------------------------------------------
# Action Publishers: Grasp and Release
# ---------------------------------------------------

def publish_grasp_action_goal(grasp_force_value=100
                              ):
    """
    Publish a grasp goal to the robot's action server.
    """
    pub = rospy.Publisher('/robotA/grasp_action/goal', GraspActionGoal, queue_size=10)
    rospy.sleep(0.5)  # Allow publisher to initialize
    msg = GraspActionGoal()
    msg.goal.grasp_force = grasp_force_value
    pub.publish(msg)
    rospy.loginfo("Grasp command sent with force: %s", grasp_force_value)


def publish_release_action_goal():
    """
    Publish a release goal to the robot's action server.
    """
    pub = rospy.Publisher('/robotA/release_action/goal', ReleaseActionGoal, queue_size=10)
    rospy.sleep(0.5)  # Allow publisher to initialize
    msg = ReleaseActionGoal()
    pub.publish(msg)
    rospy.loginfo("Release command sent.")


# ---------------------------------------------------
# Main Velocity-Based Pick-and-Place Commander
# ---------------------------------------------------

class VelocityCommander:
    def __init__(self):
        rospy.init_node("velocity_commander", anonymous=True)

        # Publisher for velocity control commands
        self.pub = rospy.Publisher("/velocity_controller/command", Float64MultiArray, queue_size=10)

        # Subscriber to joint state feedback
        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)

        # Trajectory planner (quintic polynomial)
        self.traj_planner = TrajectoryPlanner()

        # State Variables
        self.current_joint_state = None
        self.trajectory_generated = False
        self.velocity_traj = None
        self.current_index = 0
        self.dt = 0.001  # 1ms control loop

        # Load URDF and KDL chain for IK
        if not rospy.has_param("robot_description"):
            rospy.logerr("robot_description param not found.")
            exit(1)
        self.robot = URDF.from_parameter_server()
        success, tree = treeFromParam("robot_description")
        if not success:
            rospy.logerr("Failed to parse KDL tree from URDF.")
            exit(1)

        self.base_link = "base_link"
        self.tip_link = "tool_ff"
        self.chain = tree.getChain(self.base_link, self.tip_link)
        self.n_joints = self.chain.getNrOfJoints()
        rospy.loginfo("KDL chain created with %d joints", self.n_joints)

        # IK solver (Levenberg–Marquardt)
        self.ik_solver = kdl.ChainIkSolverPos_LMA(self.chain)

        # Define pick-and-place poses and associated actions
        # Define pick-and-place poses and associated actions
        # ------------------ TODO ------------------
        # Fill this list with poses in the following format:
        # Each pose should be a dictionary with:
        # - 'pos': kdl.Vector(x, y, z)
        # - 'quat': kdl.Rotation.Quaternion(x, y, z, w)
        # - 'action': 'open', 'close', or '' (for no action)
        # These should represent:
        # [1] At home position
        # [2] Move to grasp pose
        # [3] Lifts the object at a certain small height
        # [4] Moves to home position
        # [5] Moves to place position and places the object
        # [6] Return to start/home position
        
        # Quaternions (quat) decides the orientation of the gripper, the way it hold the object and it will remain same throughout the task
        
        # self.poses = [
        #     #Example entry:
        #     # Pickup Position 
        #     {
        #         'pos': kdl.Vector(0.24104107914929673, 0.5771097627404838, 0.24493448321162367),
        #         'quat': kdl.Rotation.Quaternion(0, 0, 0, 1),
        #         'action': "close"
        #     }
        # ]
        self.poses = [
            {
                # At home position
                'pos': kdl.Vector(-0.001020239304655961, 0.3678096086036452, 0.5540068593860171),
                'quat':kdl.Rotation.Quaternion(0, 0, 0, 1),
                'action': "open"
            },
            {
                # Move to grasp pose
                'pos': kdl.Vector(0.24104107914929673, 0.5771097627404838, 0.24493448321162367),
                'quat': kdl.Rotation.Quaternion(0, 0, 0, 1),
                'action': "close"
            },
            {
                # Lifts the object at a certain small height
                'pos': kdl.Vector(0.24104107914929673, 0.5771097627404838, 0.40493448321162367),
                'quat': kdl.Rotation.Quaternion(0, 0, 0, 1),
                'action': "close"
            },
            {
                # Moves to home position
                'pos': kdl.Vector(-0.001020239304655961, 0.3678096086036452, 0.5540068593860171),
                'quat': kdl.Rotation.Quaternion(0, 0, 0, 1),
                'action': "close"
            },
            {
                # Moves to place position and places the object
                'pos': kdl.Vector(-0.24104107914929673, 0.5771097627404838, 0.3593448321162367),
                'quat': kdl.Rotation.Quaternion(0, 0, 0, 1),
                'action': "open"
            },
            {
                # Return to start/home position
                'pos': kdl.Vector(-0.001020239304655961, 0.3678096086036452, 0.5540068593860171),
                'quat': kdl.Rotation.Quaternion(0, 0, 0, 1),
                'action': "open"
            }
        ]

        self.current_pose_index = 0

    def compute_ik(self, q_init, pose_index):
        """
        Solve inverse kinematics for the given target pose.
        Returns joint angles (numpy array) if successful, else None.
        """
        target = self.poses[pose_index]
        target_frame = kdl.Frame(target['quat'], target['pos'])
        q_out = kdl.JntArray(self.n_joints)
        result = self.ik_solver.CartToJnt(q_init, target_frame, q_out)
        if result >= 0:
            return np.array([q_out[i] for i in range(self.n_joints)])
        else:
            rospy.logerr("IK failed for pose %d", pose_index)
            return None

    def joint_state_callback(self, msg):
        """
        Callback to update current joint state and generate trajectory.
        """
        self.current_joint_state = np.array(msg.position)

        if not self.trajectory_generated:
            q_init = kdl.JntArray(self.n_joints)
            for i in range(self.n_joints):
                q_init[i] = self.current_joint_state[i] if self.current_joint_state is not None else 0.0

            target_joint_state = self.compute_ik(q_init, self.current_pose_index)
            if target_joint_state is None:
                return

            T = 2.8  # seconds
            _, self.velocity_traj, _ = self.traj_planner.quintic_joint_trajectory(
                self.current_joint_state, target_joint_state, T, self.dt
            )

            self.trajectory_generated = True
            self.current_index = 0

    def run(self):
        """
        Main control loop to publish joint velocity commands and trigger grasp/release.
        """
        rate = rospy.Rate(1.0 / self.dt)
        while not rospy.is_shutdown():
            if self.trajectory_generated and self.current_index < len(self.velocity_traj):
                msg = Float64MultiArray()
                msg.data = self.velocity_traj[self.current_index].tolist()
                self.pub.publish(msg)
                self.current_index += 1

            elif self.trajectory_generated and self.current_index >= len(self.velocity_traj):
                # Trajectory complete – trigger any associated action
                action = self.poses[self.current_pose_index].get('action', '').lower().strip()
                if action == "close":
                    publish_grasp_action_goal()
                elif action == "open":
                    publish_release_action_goal()
                rospy.sleep(3)

                if self.current_pose_index < len(self.poses) - 1:
                    self.current_pose_index += 1
                    self.trajectory_generated = False
                else:
                    rospy.loginfo("Pick-and-place sequence complete.")
                    rospy.signal_shutdown("Finished all poses.")
            rate.sleep()


# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------

if __name__ == '__main__':
    try:
        commander = VelocityCommander()
        commander.run()
    except rospy.ROSInterruptException:
        pass
