"""
LEAP HAND — FULLY EXPLAINED DEMO
=================================

JOINT ANATOMY (per finger):
  Each finger has 4 controllable joints:
    MCP Side  = side-to-side spread (abduction/adduction)
    MCP Bend  = forward curl at the big knuckle
    PIP       = middle knuckle curl
    DIP       = fingertip knuckle curl

LEAP HAND — 16 JOINTS TOTAL (4 fingers × 4 joints):
  Index finger  → joints  0, 1, 2, 3
  Middle finger → joints  4, 5, 6, 7
  Ring finger   → joints  8, 9, 10, 11
  Thumb         → joints 12, 13, 14, 15

ANGLE CONVENTION (allegro mode — what set_allegro uses):
  0.0 rad = fully open / straight finger
  Positive values = curl inward (closing)
  ~1.5 rad = tightly curled

HOW A COMMAND REACHES THE MOTOR:
  1. You define a pose: np.array of 16 angle values
  2. You call hand.set_allegro(pose)
  3. Internally: lhu.allegro_to_LEAPhand() converts 0-based angles
     into LEAP's native 3.14-centered radian space
  4. DynamixelClient.write_desired_pos() sends those values
     over USB → U2D2 adapter → Dynamixel TTL bus → each motor
  5. Each motor's PID controller moves to that target angle

HOW POSITIONS ARE READ:
  1. hand.read_pos() sends a read request over the same USB bus
  2. Each motor responds with its current angle in radians
  3. Returns a numpy array of 16 values (LEAP native space)
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu


# =============================================================================
# LEAP NODE — the class that talks to the hardware over USB
# =============================================================================
class LeapNode:
    def __init__(self):
        # PID gains — these control how stiff/smooth the motors are
        # kP: proportional gain — higher = stiffer, lower = softer/jittery
        # kD: derivative gain  — higher = more damping, reduces oscillation
        self.kP = 600
        self.kI = 0
        self.kD = 200

        # Current limit in mA — prevents motors from overheating
        # 350 for lite hand, 550 for full hand
        self.curr_lim = 350

        # Starting position: all joints at 0 (open hand in allegro space)
        # allegro_to_LEAPhand converts 0→3.14 for LEAP's native coordinates
        self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        # All 16 motor IDs on the Dynamixel bus
        self.motors = motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # Try to connect — auto-scans common ports
        ports_to_try = ['COM13', 'COM5', 'COM3', '/dev/ttyUSB0', '/dev/ttyUSB1']
        connected = False
        for port in ports_to_try:
            try:
                print(f"Trying {port}...")
                # DynamixelClient opens the serial port at 4,000,000 baud
                self.dxl_client = DynamixelClient(motors, port, 4000000)
                self.dxl_client.connect()
                connected = True
                print(f"Connected on {port}!")
                break
            except Exception as e:
                print(f"  Failed: {e}")

        if not connected:
            raise RuntimeError("Could not connect to LEAP Hand on any port.")

        # Set all motors to position-current control mode (mode 5)
        # This means: "go to this angle, but don't exceed current_limit amps"
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)

        # Enable torque — motors are now active and will hold position
        self.dxl_client.set_torque_enabled(motors, True)

        # Write PID gains to all motors
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)  # side joints softer
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)  # side joints softer

        # Set current (torque) limit
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)

        # Send the initial position command (open hand)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # -------------------------------------------------------------------------
    # set_allegro: the MAIN command you'll use
    #   - Takes angles where 0.0 = open, positive = curling closed
    #   - Converts to LEAP native space internally (adds 3.14 offset)
    #   - THIS is where the movement command actually gets sent to the motors
    # -------------------------------------------------------------------------
    def set_allegro(self, pose):
        # Convert from allegro (0=open) to LEAP native (3.14=open)
        leap_pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.curr_pos = np.array(leap_pose)
        # THIS LINE sends the command over USB to all 16 Dynamixel motors
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # set_leap: use this if you already have LEAP-native angles (3.14 = open)
    def set_leap(self, pose):
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read_pos: asks each motor "what angle are you at right now?"
    # Returns 16 values in LEAP native space (radians, ~3.14 = open)
    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_vel(self):
        return self.dxl_client.read_vel()

    def read_cur(self):
        return self.dxl_client.read_cur()


# =============================================================================
# POSE BUILDER
# Build a 16-value pose array from per-joint angles.
# All 4 fingers get the same angles (you can customize per-finger below).
# =============================================================================
def make_pose(mcp_side=0.0, mcp_bend=0.0, pip=0.0, dip=0.0):
    """
    mcp_side : side spread (0 = neutral, positive = spread apart)
    mcp_bend : big knuckle curl (0 = straight, 1.5 = fully bent)
    pip      : middle knuckle curl
    dip      : fingertip knuckle curl
    """
    # One finger's 4 joints, repeated for all 4 fingers
    one_finger = [mcp_side, mcp_bend, pip, dip]
    return np.array(one_finger * 4, dtype=np.float64)


# =============================================================================
# SMOOTH MOTION HELPER
# Instead of jumping instantly to a target, this interpolates step-by-step.
# It reads current position, then moves gradually toward the target.
# =============================================================================
def move_to(hand, target_pose, steps=30, delay=0.03):
    """
    hand        : LeapNode instance
    target_pose : 16-value numpy array (allegro angles)
    steps       : how many intermediate steps (more = smoother)
    delay       : seconds between each step (controls speed)

    HOW IT WORKS:
    - Reads where the hand is NOW (in LEAP native space)
    - Converts target to LEAP native space
    - Interpolates linearly between current and target
    - Sends one command per step
    """
    # Read current positions (LEAP native radians)
    current = np.array(hand.read_pos(), dtype=np.float64)

    # Convert our allegro target to LEAP native for interpolation
    target_leap = lhu.allegro_to_LEAPhand(target_pose, zeros=False)

    print(f"  Moving over {steps} steps...")
    for i in range(1, steps + 1):
        alpha = i / steps                             # 0.033 → 0.066 → ... → 1.0
        interp = current + alpha * (target_leap - current)  # linear blend
        hand.set_leap(interp)                         # send to motors
        time.sleep(delay)


# =============================================================================
# NAMED MOTION FUNCTIONS
# Each function = one motion. Clear, reusable, easy to call in any order.
# =============================================================================

def open_hand(hand):
    """
    Fully open/flat hand — all joints at 0 (allegro convention).
    This is the safe home position.
    """
    print(">> open_hand")
    pose = make_pose(mcp_side=0.0, mcp_bend=0.0, pip=0.0, dip=0.0)
    move_to(hand, pose)


def close_fist(hand):
    """
    Full fist — MCP and PIP both curl strongly.
    mcp_bend=1.5 : big knuckle bends ~85 degrees
    pip=1.5      : middle knuckle bends ~85 degrees
    dip=1.0      : fingertip knuckle bends ~57 degrees
    """
    print(">> close_fist")
    pose = make_pose(mcp_side=0.0, mcp_bend=1.5, pip=1.5, dip=1.0)
    move_to(hand, pose)


def pinch_pose(hand):
    """
    Partial curl — good for pinching or holding small objects.
    All joints halfway closed.
    """
    print(">> pinch_pose")
    pose = make_pose(mcp_side=0.0, mcp_bend=0.8, pip=0.8, dip=0.5)
    move_to(hand, pose)


def spread_fingers(hand):
    """
    Open hand with fingers spread wide apart (abduction).
    mcp_side controls the side-to-side spread.
    Positive = spread out, negative = squeeze together.
    """
    print(">> spread_fingers")
    pose = make_pose(mcp_side=0.3, mcp_bend=0.0, pip=0.0, dip=0.0)
    move_to(hand, pose)


def point_index(hand):
    """
    Index finger pointing, other fingers curled.
    We build this manually per-finger instead of using make_pose.

    Joint layout reminder:
      Index  = [0, 1, 2, 3]
      Middle = [4, 5, 6, 7]
      Ring   = [8, 9, 10, 11]
      Thumb  = [12, 13, 14, 15]
    """
    print(">> point_index")
    pose = np.array([
        0.0, 0.0, 0.0, 0.0,    # Index:  open/straight
        0.0, 1.4, 1.4, 0.8,    # Middle: curled closed
        0.0, 1.4, 1.4, 0.8,    # Ring:   curled closed
        0.0, 1.0, 1.0, 0.6,    # Thumb:  partially curled
    ], dtype=np.float64)
    move_to(hand, pose)


def half_curl(hand):
    """
    All fingers at half-curl — useful as a neutral grasping ready position.
    """
    print(">> half_curl")
    pose = make_pose(mcp_side=0.0, mcp_bend=0.7, pip=0.7, dip=0.4)
    move_to(hand, pose)


def read_and_print(hand):
    """
    Read and display the current state of all 16 joints.
    Useful for debugging and understanding what angles the hand is actually at.
    """
    pos = hand.read_pos()
    cur = hand.read_cur()
    print("\n--- Current Hand State ---")
    print(f"Joint positions (LEAP native radians):")
    fingers = ['Index', 'Middle', 'Ring', 'Thumb']
    joint_names = ['MCP-side', 'MCP-bend', 'PIP    ', 'DIP    ']
    for f, finger in enumerate(fingers):
        print(f"  {finger}:")
        for j in range(4):
            idx = f * 4 + j
            print(f"    [{idx:2d}] {joint_names[j]}: {pos[idx]:.3f} rad  |  current: {cur[idx]:.1f} mA")
    print()


# =============================================================================
# MAIN — connect and run the demo sequence
# =============================================================================
if __name__ == "__main__":

    # STEP 1: Connect to the hand
    # LeapNode.__init__() opens the USB port, enables torque, sets PID gains
    hand = LeapNode()
    print("\nLEAP Hand connected. Starting demo...\n")

    try:
        # STEP 2: Run each motion function one by one
        # Each function builds a pose, then move_to() smoothly drives there

        open_hand(hand)        # flat/safe home position
        time.sleep(1.0)

        read_and_print(hand)   # see what angles look like when open

        close_fist(hand)       # curl all fingers into a fist
        time.sleep(1.0)

        read_and_print(hand)   # see what angles look like when closed

        open_hand(hand)
        time.sleep(0.8)

        pinch_pose(hand)       # partial curl / pinch
        time.sleep(1.0)

        open_hand(hand)
        time.sleep(0.8)

        spread_fingers(hand)   # fingers spread apart
        time.sleep(1.0)

        open_hand(hand)
        time.sleep(0.8)

        point_index(hand)      # index pointing, others curled
        time.sleep(1.5)

        open_hand(hand)
        time.sleep(0.8)

        half_curl(hand)        # ready-to-grasp position
        time.sleep(1.0)

        open_hand(hand)        # always end safe
        time.sleep(0.5)

        print("\nDemo complete!")

    except KeyboardInterrupt:
        print("\nStopped by user — returning to open hand...")

    finally:
        # ALWAYS return to safe open position before quitting
        # This prevents the hand from freezing in a curled state
        print("Returning to open hand (safe exit)...")
        hand.set_allegro(np.zeros(16))
        time.sleep(1.0)