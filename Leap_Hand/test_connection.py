import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu

# Direct Dynamixel register addresses
ADDR_HARDWARE_ERROR   = 70   # read — shows what error the motor has
ADDR_TORQUE_ENABLE    = 64   # write 1 = on, 0 = off
ADDR_OPERATING_MODE   = 11   # write 5 = position-current control
ADDR_GOAL_CURRENT     = 102
ADDR_PROFILE_VEL      = 112  # controls how fast motor moves to target

def clear_errors_and_enable(dxl_client, motors):
    """
    Dynamixel motors latch errors and refuse torque until cleared.
    Sequence: disable torque → reboot → re-enable torque.
    """
    print("  Clearing motor errors...")

    # Step 1 — disable torque on all motors (required before mode change)
    dxl_client.set_torque_enabled(motors, False)
    time.sleep(0.2)

    # Step 2 — reboot each motor to clear hardware error latch
    # Reboot packet: 0xFF 0xFF 0xFD 0x00 ID 0x02 0x00 0x08 CRC
    for motor_id in motors:
        try:
            dxl_client.reboot(motor_id)
        except Exception:
            pass
    time.sleep(1.0)   # motors need time to reboot and re-enumerate

    # Step 3 — set operating mode 5 (position-current control)
    dxl_client.sync_write(motors, np.ones(len(motors))*5, ADDR_OPERATING_MODE, 1)
    time.sleep(0.1)

    # Step 4 — set current limit (mA)
    dxl_client.sync_write(motors, np.ones(len(motors))*350, ADDR_GOAL_CURRENT, 2)
    time.sleep(0.1)

    # Step 5 — set profile velocity (how fast it moves — 0 = max speed)
    # Setting a moderate velocity prevents violent snapping to position
    dxl_client.sync_write(motors, np.ones(len(motors))*100, ADDR_PROFILE_VEL, 4)
    time.sleep(0.1)

    # Step 6 — enable torque
    dxl_client.set_torque_enabled(motors, True)
    time.sleep(0.2)

    print("  Done. Torque enabled.")

class LeapNode:
    def __init__(self):
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.motors = motors = list(range(16))
        self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        for port in ['COM13', 'COM5', '/dev/ttyUSB0']:
            try:
                print(f"Trying {port}...")
                self.dxl_client = DynamixelClient(motors, port, 4000000)
                self.dxl_client.connect()
                print(f"Connected on {port}!")
                break
            except Exception:
                pass

        # Clear any latched errors, reboot motors, enable torque
        clear_errors_and_enable(self.dxl_client, motors)

        # PID gains
        self.dxl_client.sync_write(motors, np.ones(16)*self.kP, 84, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3)*(self.kP*0.75), 84, 2)
        self.dxl_client.sync_write(motors, np.ones(16)*self.kI, 82, 2)
        self.dxl_client.sync_write(motors, np.ones(16)*self.kD, 80, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3)*(self.kD*0.75), 80, 2)

        # Command current position as first target (no movement on startup)
        actual_pos = self.dxl_client.read_pos()
        self.curr_pos = np.array(actual_pos)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        print(f"  Startup position: {np.round(self.curr_pos[:4], 2)}")

    def set_allegro(self, pose):
        leap_pose = lhu.allegro_to_LEAPhand(np.array(pose), zeros=False)
        self.curr_pos = leap_pose
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_cur(self):
        return self.dxl_client.read_cur()

def d(deg):
    return float(np.deg2rad(deg))

def move_to(hand, target_allegro, steps=30, delay=0.04):
    current = np.array(hand.read_pos(), dtype=np.float64)
    target_leap = lhu.allegro_to_LEAPhand(np.array(target_allegro), zeros=False)
    print(f"  current={np.round(current[:4],2)} → target={np.round(target_leap[:4],2)}")
    for i in range(1, steps+1):
        alpha = i / steps
        interp = current + alpha * (target_leap - current)
        hand.dxl_client.write_desired_pos(hand.motors, interp)
        time.sleep(delay)
    # After move, check if motors actually moved
    final = np.array(hand.read_pos())
    print(f"  actual after move={np.round(final[:4],2)}")
    cur = np.array(hand.read_cur())
    print(f"  current draw (mA)={np.round(cur[:4],1)}")

OPEN  = np.zeros(16)
FIST  = np.array([d(0),d(86),d(86),d(57)] * 4)
PINCH = np.array([d(0),d(46),d(46),d(29)] * 4)

# ── Run ──────────────────────────────────────────────────────────────────────
hand = LeapNode()
print("\nStarting demo...\n")

try:
    print("1. Open hand...")
    move_to(hand, OPEN)
    time.sleep(2.0)

    print("\n2. Close fist...")
    move_to(hand, FIST)
    time.sleep(2.0)

    print("\n3. Open hand...")
    move_to(hand, OPEN)
    time.sleep(2.0)

    print("\n4. Pinch...")
    move_to(hand, PINCH)
    time.sleep(2.0)

    print("\n5. Open...")
    move_to(hand, OPEN)
    time.sleep(1.0)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    hand.set_allegro(OPEN)
    time.sleep(1.0)
    print("Done.")