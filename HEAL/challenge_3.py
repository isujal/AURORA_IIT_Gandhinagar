#!/usr/bin/env python3

"""
Trajectory Tracker Template using DLS Velocity Commander
Author: [Your Name Here]
Email: [your.email@example.com]

Overview:
---------
This is a student template to implement Cartesian trajectory tracking for the HEAL robot using velocity-based
Damped Least Squares (DLS) inverse kinematics. You will fill in your own trajectory functions and test them.

Instructions:
-------------
1. Fill in each trajectory generator function below:
   - `circular_trajectory()`
   - `sine_wave_bidirectional()`
   - `straight_line_trajectory()`

2. Use the `custom_trajectory()` function if you wish to define your own path.

3. You may test one at a time by uncommenting the respective function call inside the main control loop.

Safety Note:
------------
⚠️ Angular velocity control is disabled in DLS by default to prevent cable winding. You may modify
   `compute_dls_ik()` inside `dls_velocity_commander.py` to re-enable it for smooth cyclic paths.
"""

# ------------------ Imports ------------------

import rospy
import numpy as np
import PyKDL as kdl
import matplotlib.pyplot as plt
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float64MultiArray
from dls_velocity_commander import DLSVelocityCommander

# ------------------ Trajectory Functions (TO BE FILLED) ------------------

def circular_trajectory(t):
    """
    TODO: Implement a circular trajectory in the XY plane with constant Z height.
    - pos: [x, y, z] should trace a circle over time.
    - quat: should be tangential to the path (optional).
    """
    radius = 0.2
    omega = 1.0 
    center_x = 0.0
    center_y = 0.6
    constant_z = 0.45
    x= center_x + radius*np.cos(omega*t) 
    y= center_y + radius*np.sin(omega*t) 
    z= constant_z
    pos = [x,y,z]  # Replace with your circular position
    quat = quaternion_from_euler(0, 0, 0)  # Default downward
    return pos, quat

def sine_wave_bidirectional(t):
    """
    TODO: Implement a sine wave that goes back and forth using time.
    - pos: X moves sinusoidally. Y = sin(pi * x), Z is constant.
    - quat: fixed downward.
    """

    pos = [0.0, 0.0, 0.0]  # Replace with sine wave position
    quat = quaternion_from_euler(0, 0, 0)
    return pos, quat

def straight_line_trajectory(t):
    """
    TODO: Implement straight-line back-and-forth motion along the X axis.
    - pos: Move smoothly between two X points while Y and Z remain fixed.
    - quat: fixed downward.
    """
    amplitude = 0.25
    omega = 1.0
    centre_x =0.2
    centre_y = 0.5
    constant_z = 0.45
    x = centre_x + amplitude*np.sin(omega*t)
    y= centre_y
    z = constant_z
    0.0, 0.0, 0.0
    pos = [x,y,z]  # Replace with linear motion
    quat = quaternion_from_euler(0, 0, 0)
    return pos, quat


# ------------------ Plotting Function ------------------

def plot_trajectory(traj, title="End-Effector Trajectory"):
    traj = np.array(traj)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label='EE Path', lw=2)
    ax.scatter([0], [0], [0], color='r', label='Origin', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Main Execution ------------------

if __name__ == '__main__':
    # Initial pose to initialize IK solver (use your trajectory)
    init_pos, init_quat = straight_line_trajectory(0.0)
    commander = DLSVelocityCommander(target_pos=init_pos, target_quat=init_quat)
    pub_zero = rospy.Publisher("/velocity_controller/command", Float64MultiArray, queue_size=10)
    trajectory_log = []

    def shutdown_hook():
        rospy.loginfo("Shutting down. Sending zero velocity command.")
        zero_msg = Float64MultiArray()
        zero_msg.data = [0.0] * commander.n_joints
        pub_zero.publish(zero_msg)
        plot_trajectory(trajectory_log)

    rospy.on_shutdown(shutdown_hook)

    rate = rospy.Rate(100)  # 100 Hz loop
    t_start = rospy.get_time()

    try:
        while not rospy.is_shutdown():
            t = rospy.get_time() - t_start

            # ---- SELECT ONLY ONE TRAJECTORY FUNCTION AT A TIME ----
            #pos, quat = circular_trajectory(t)
            #pos, quat = sine_wave_bidirectional(t)
            pos, quat = straight_line_trajectory(t)

            commander.target_pos = kdl.Vector(*pos)
            commander.target_quat = kdl.Rotation.Quaternion(*quat)
            commander.target_frame = kdl.Frame(commander.target_quat, commander.target_pos)
            commander.run_once()

            trajectory_log.append(pos)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
