"""
LEAP HAND — DEGREE-BASED POSE WRITING
======================================

JOINT RANGES (allegro convention):
  MCP side (spread)  : -20° to +20°   (-0.35 to +0.35 rad)
  MCP bend (knuckle) :   0° to  90°   ( 0.00 to  1.57 rad)
  PIP  (mid knuckle) :   0° to  90°   ( 0.00 to  1.57 rad)
  DIP  (fingertip)   :   0° to  57°   ( 0.00 to  1.00 rad)

  0° always = straight/open
  Positive degrees = curling closed
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu


# =============================================================================
# DEGREE HELPER — write angles in degrees, auto-converts to radians
# =============================================================================
def d(degrees):
    """Convert degrees to radians. Use this everywhere instead of raw numbers."""
    return float(np.deg2rad(degrees))


# =============================================================================
# LEAP NODE
# =============================================================================
class LeapNode:
    def __init__(self):
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.motors = motors = list(range(16))

        ports_to_try = ['COM13', 'COM5', 'COM3', '/dev/ttyUSB0', '/dev/ttyUSB1']
        connected = False
        for port in ports_to_try:
            try:
                print(f"Trying {port}...")
                self.dxl_client = DynamixelClient(motors, port, 4000000)
                self.dxl_client.connect()
                connected = True
                print(f"Connected on {port}!")
                break
            except Exception:
                pass

        if not connected:
            raise RuntimeError("Could not connect to LEAP Hand on any port.")

        self.dxl_client.sync_write(motors, np.ones(16) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(16) * self.kP, 84, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(motors, np.ones(16) * self.kI, 82, 2)
        self.dxl_client.sync_write(motors, np.ones(16) * self.kD, 80, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(motors, np.ones(16) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_allegro(self, pose):
        leap_pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.curr_pos = np.array(leap_pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_leap(self, pose):
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_cur(self):
        return self.dxl_client.read_cur()


# =============================================================================
# POSE BUILDER — using degrees
# Each finger: [mcp_side°, mcp_bend°, pip°, dip°]
# make_pose applies same angles to all 4 fingers
# =============================================================================
def make_pose(mcp_side_deg=0, mcp_bend_deg=0, pip_deg=0, dip_deg=0):
    """
    Write angles in degrees. Internally converts to radians for LEAP.

    Ranges:
      mcp_side_deg : -20 to +20   (side spread)
      mcp_bend_deg :   0 to  90   (main knuckle bend)
      pip_deg      :   0 to  90   (middle knuckle)
      dip_deg      :   0 to  57   (fingertip knuckle)
    """
    one_finger = [d(mcp_side_deg), d(mcp_bend_deg), d(pip_deg), d(dip_deg)]
    return np.array(one_finger * 4, dtype=np.float64)


def make_pose_per_finger(index, middle, ring, thumb):
    """
    Set different angles per finger.
    Each argument is a list: [mcp_side°, mcp_bend°, pip°, dip°]

    Example:
      make_pose_per_finger(
          index  = [0, 0,  0,  0],   # pointing straight
          middle = [0, 80, 80, 50],  # curled
          ring   = [0, 80, 80, 50],  # curled
          thumb  = [0, 50, 50, 30],  # half curled
      )
    """
    def to_rad(finger_deg):
        return [d(a) for a in finger_deg]

    return np.array(
        to_rad(index) + to_rad(middle) + to_rad(ring) + to_rad(thumb),
        dtype=np.float64
    )


# =============================================================================
# SMOOTH MOTION
# =============================================================================
def move_to(hand, target_pose, steps=30, delay=0.03):
    current = np.array(hand.read_pos(), dtype=np.float64)
    target_leap = lhu.allegro_to_LEAPhand(target_pose, zeros=False)
    for i in range(1, steps + 1):
        alpha = i / steps
        hand.set_leap(current + alpha * (target_leap - current))
        time.sleep(delay)


# =============================================================================
# NAMED MOTION FUNCTIONS — all angles written in degrees
# =============================================================================

def open_hand(hand):
    """All fingers fully straight. Safe home position."""
    print(">> open_hand  (all joints 0°)")
    pose = make_pose(
        mcp_side_deg=0,
        mcp_bend_deg=0,
        pip_deg=0,
        dip_deg=0
    )
    move_to(hand, pose)


def half_curl(hand):
    """Fingers at ~40° bend — relaxed, ready-to-grasp position."""
    print(">> half_curl  (MCP 40°, PIP 40°, DIP 23°)")
    pose = make_pose(
        mcp_side_deg=0,
        mcp_bend_deg=40,   # 0.70 rad
        pip_deg=40,        # 0.70 rad
        dip_deg=23         # 0.40 rad
    )
    move_to(hand, pose)


def pinch_pose(hand):
    """Partial curl — pinching or holding small objects."""
    print(">> pinch_pose  (MCP 46°, PIP 46°, DIP 29°)")
    pose = make_pose(
        mcp_side_deg=0,
        mcp_bend_deg=46,   # 0.80 rad
        pip_deg=46,        # 0.80 rad
        dip_deg=29         # 0.50 rad
    )
    move_to(hand, pose)


def close_fist(hand):
    """Full fist — all joints at maximum curl."""
    print(">> close_fist  (MCP 86°, PIP 86°, DIP 57°)")
    pose = make_pose(
        mcp_side_deg=0,
        mcp_bend_deg=86,   # 1.50 rad
        pip_deg=86,        # 1.50 rad
        dip_deg=57         # 1.00 rad
    )
    move_to(hand, pose)


def spread_fingers(hand):
    """Open hand with fingers spread wide apart."""
    print(">> spread_fingers  (MCP side +17°)")
    pose = make_pose(
        mcp_side_deg=17,   # 0.30 rad spread
        mcp_bend_deg=0,
        pip_deg=0,
        dip_deg=0
    )
    move_to(hand, pose)


def point_index(hand):
    """Index finger straight, other fingers curled into fist."""
    print(">> point_index")
    pose = make_pose_per_finger(
        index  = [0,  0,  0,  0],    # all straight — pointing
        middle = [0, 80, 80, 46],    # fully curled
        ring   = [0, 80, 80, 46],    # fully curled
        thumb  = [0, 57, 57, 34],    # curled
    )
    move_to(hand, pose)


def peace_sign(hand):
    """Index and middle straight, ring and pinky curled."""
    print(">> peace_sign")
    pose = make_pose_per_finger(
        index  = [0,  0,  0,  0],    # straight
        middle = [0,  0,  0,  0],    # straight
        ring   = [0, 80, 80, 46],    # curled
        thumb  = [0, 57, 57, 34],    # curled
    )
    move_to(hand, pose)


def read_and_print_degrees(hand):
    """Print all 16 joint positions converted to degrees."""
    pos = hand.read_pos()
    cur = hand.read_cur()
    print("\n--- Current Joint Positions ---")
    fingers = ['Index', 'Middle', 'Ring ', 'Thumb ']
    joints  = ['MCP-side', 'MCP-bend', 'PIP    ', 'DIP    ']
    for f, fname in enumerate(fingers):
        print(f"  {fname}:")
        for j in range(4):
            idx = f * 4 + j
            # pos[idx] is in LEAP native (3.14 = open), subtract pi to get allegro, then to degrees
            allegro_rad = pos[idx] - np.pi
            deg = np.degrees(allegro_rad)
            mA  = cur[idx]
            print(f"    [{idx:2d}] {joints[j]}: {deg:+6.1f}°   ({pos[idx]:.3f} rad LEAP native)   {mA:.0f} mA")
    print()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    hand = LeapNode()
    print("\nLEAP Hand connected. Starting degree-based demo...\n")

    try:
        open_hand(hand)
        time.sleep(1.0)

        read_and_print_degrees(hand)   # see what 0° looks like

        close_fist(hand)
        time.sleep(1.0)

        read_and_print_degrees(hand)   # see what 86° looks like

        open_hand(hand)
        time.sleep(0.8)

        half_curl(hand)
        time.sleep(1.0)

        open_hand(hand)
        time.sleep(0.8)

        pinch_pose(hand)
        time.sleep(1.0)

        open_hand(hand)
        time.sleep(0.8)

        spread_fingers(hand)
        time.sleep(1.0)

        open_hand(hand)
        time.sleep(0.8)

        point_index(hand)
        time.sleep(1.5)

        open_hand(hand)
        time.sleep(0.8)

        peace_sign(hand)
        time.sleep(1.5)

        open_hand(hand)
        time.sleep(0.5)

        print("\nDemo complete!")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("Safe exit — returning to open hand...")
        hand.set_allegro(np.zeros(16))
        time.sleep(1.0)