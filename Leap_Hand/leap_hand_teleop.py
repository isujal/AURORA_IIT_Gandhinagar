"""
LEAP HAND — VISION TELEOPERATION (raw position units)
======================================================
Uses Dynamixel raw position values (0-4095) directly.
No radian conversion, no allegro conversion — what you see
in Dynamixel Wizard is exactly what gets sent.

DYNAMIXEL POSITION UNITS:
  0 to 4095 = full 360 degrees
  1 unit = 0.088 degrees

CALIBRATED RANGES (from Dynamixel Wizard):
  Index  MCP-bend: 1800 (open) to 2800 (closed)
  Middle MCP-bend: 1800 (open) to 2800 (closed)
  Ring   MCP-bend:  800 (open) to 1800 (closed)  ← mounted opposite direction
  Others: fill in after checking Wizard

CONTROLS:
  S = pause / resume
  Q = quit
  R = reset smoother
"""

import sys
import os
import time
import numpy as np
import cv2

import mediapipe.python.solutions.hands as mp_hands_mod
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_styles

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu


# =============================================================================
# CONFIG
# =============================================================================
SMOOTHING_ALPHA = 0.4
CAMERA_INDEX    = 0
SEND_HZ         = 30

# =============================================================================
# RAW POSITION RANGES (Dynamixel units, from Wizard)
#
# Format: [open_position, close_position]
# open  = where the joint sits when finger is fully straight
# close = where the joint sits when finger is fully curled
#
# If open > close, the servo runs in reverse (like ring finger).
# The fraction math handles both directions automatically.
#
# FILL IN VALUES FROM WIZARD — use ??? as placeholder until confirmed
# =============================================================================
RAW_OPEN = np.array([
    2048,   # [ 0] Index  MCP-side   ← neutral, update from Wizard
    1800,   # [ 1] Index  MCP-bend   ← confirmed from Wizard //
    2048,   # [ 2] Index  PIP        ← update from Wizard //
    2048,   # [ 3] Index  DIP        ← update from Wizard //
    2048,   # [ 4] Middle MCP-side   ← neutral
    1800,   # [ 5] Middle MCP-bend   ← confirmed
    2048,   # [ 6] Middle PIP        ← update
    2048,   # [ 7] Middle DIP        ← update
    2048,   # [ 8] Ring   MCP-side   ← neutral
    800,   # [ 9] Ring   MCP-bend   ← confirmed (but inverted — open=1800)
    1800,   # [10] Ring   PIP        ← update
    2600,   # [11] Ring   DIP        ← update
    1909,   # [12] Thumb  MCP-side   ← update
    1531,   # [13] Thumb  MCP-bend   ← update
    1948,   # [14] Thumb  PIP        ← update
    2479,   # [15] Thumb  DIP        ← update
], dtype=np.float64)

RAW_CLOSE = np.array([
    2048,   # [ 0] Index  MCP-side   ← same as open (no side movement)
    2800,   # [ 1] Index  MCP-bend   ← confirmed
    2800,   # [ 2] Index  PIP        ← estimate, update from Wizard
    2600,   # [ 3] Index  DIP        ← estimate
    2048,   # [ 4] Middle MCP-side
    2800,   # [ 5] Middle MCP-bend   ← confirmed
    2800,   # [ 6] Middle PIP        ← estimate
    2600,   # [ 7] Middle DIP        ← estimate
    2048,   # [ 8] Ring   MCP-side
    1800,   # [ 9] Ring   MCP-bend   ← confirmed INVERTED (close=800, lower value)
    2800,   # [10] Ring   PIP        ← estimate inverted
    2600,   # [11] Ring   DIP        ← estimate
    1647,   # [12] Thumb  MCP-side
    763,   # [13] Thumb  MCP-bend   ← update
    2795,   # [14] Thumb  PIP        ← update
    3155,   # [15] Thumb  DIP        ← update
], dtype=np.float64)

# Range per joint (can be negative if servo is mounted in reverse)
RAW_RANGE = RAW_CLOSE - RAW_OPEN


# =============================================================================
# CONVERSION HELPERS
# =============================================================================
def raw_to_rad(raw_pos):
    """Convert Dynamixel raw position (0-4095) to radians (0 to 2pi)."""
    return float(raw_pos) * 2.0 * np.pi / 4095.0

def fractions_to_raw(fractions):
    """
    fractions: array of 16 values, 0.0=open, 1.0=closed
    Returns raw Dynamixel positions for each joint.
    """
    return RAW_OPEN + np.array(fractions) * RAW_RANGE

def raw_to_radians_array(raw_array):
    """Convert array of raw positions to radians for Dynamixel SDK."""
    return raw_array * 2.0 * np.pi / 4095.0


# =============================================================================
# LEAP NODE — sends raw positions directly
# =============================================================================
class LeapNode:
    def __init__(self):
        self.kP       = 600
        self.kI       = 0
        self.kD       = 200
        self.curr_lim = 350
        self.motors   = motors = list(range(16))

        for port in ['COM13', 'COM5', 'COM3', '/dev/ttyUSB0', '/dev/ttyUSB1']:
            try:
                print(f"Trying {port}...")
                self.dxl_client = DynamixelClient(motors, port, 4000000)
                self.dxl_client.connect()
                print(f"Connected on {port}!")
                break
            except Exception:
                pass

        self.dxl_client.set_torque_enabled(motors, False)
        time.sleep(0.2)
        for mid in motors:
            try: self.dxl_client.reboot(mid)
            except: pass
        time.sleep(1.0)

        self.dxl_client.sync_write(motors, np.ones(16)*5,             11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(16)*self.kP,       84, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3)*(self.kP*.75), 84, 2)
        self.dxl_client.sync_write(motors, np.ones(16)*self.kI,       82, 2)
        self.dxl_client.sync_write(motors, np.ones(16)*self.kD,       80, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3)*(self.kD*.75), 80, 2)
        self.dxl_client.sync_write(motors, np.ones(16)*self.curr_lim, 102, 2)

        # Go to open position on startup
        self.send_raw(RAW_OPEN)
        print("LEAP Hand ready — at open position.")
        print(f"  Index  MCP range: {RAW_OPEN[1]:.0f} → {RAW_CLOSE[1]:.0f}")
        print(f"  Middle MCP range: {RAW_OPEN[5]:.0f} → {RAW_CLOSE[5]:.0f}")
        print(f"  Ring   MCP range: {RAW_OPEN[9]:.0f} → {RAW_CLOSE[9]:.0f}  (inverted)")

    def send_raw(self, raw_positions):
        """Send raw Dynamixel position array (0-4095 per joint)."""
        rad_positions = raw_to_radians_array(np.array(raw_positions))
        self.dxl_client.write_desired_pos(self.motors, rad_positions)

    def send_fractions(self, fractions_16):
        """
        Send 16 fractions (0=open, 1=closed).
        Converts to raw positions using calibrated ranges.
        """
        raw = fractions_to_raw(fractions_16)
        # Clamp each joint to its valid range
        raw_min = np.minimum(RAW_OPEN, RAW_CLOSE)
        raw_max = np.maximum(RAW_OPEN, RAW_CLOSE)
        raw     = np.clip(raw, raw_min, raw_max)
        self.send_raw(raw)

    def open_hand(self):
        self.send_raw(RAW_OPEN)

    def close_hand(self):
        self.send_raw(RAW_CLOSE)

    def read_pos_raw(self):
        """Read positions and convert back to raw units for display."""
        try:
            rad = np.array(self.dxl_client.read_pos())
            return rad * 4095.0 / (2.0 * np.pi)
        except:
            return RAW_OPEN.copy()


# =============================================================================
# ANGLE MATH
# =============================================================================
def joint_angle(A, B, C):
    """Curl angle at joint B. 0=straight, increases as joint curls."""
    v1 = A - B
    v2 = C - B
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.pi - np.arccos(np.clip(cos_a, -1.0, 1.0)))


def lm(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y, p.z])


# MediaPipe max angles at full curl (used to normalise to 0-1 fraction)
MP_MAX_MCP  = 1.4
MP_MAX_PIP  = 1.4
MP_MAX_DIP  = 1.0
MP_MAX_SIDE = 0.35


# =============================================================================
# LANDMARKS → 16 FRACTIONS
#
# Each value: 0.0 = fully open, 1.0 = fully closed
# send_fractions() maps these to the correct raw Dynamixel positions,
# including handling inverted servos (ring finger) automatically.
#
# HAND MAPPING:
#   Your Index  → LEAP Index  (joints  0- 3)  landmarks 5-8
#   Your Middle → LEAP Middle (joints  4- 7)  landmarks 9-12
#   Your Ring   → LEAP Ring   (joints  8-11)  landmarks 13-16
#   Your Thumb  → LEAP Thumb  (joints 12-15)  landmarks 2-4
#   Your Pinky  → NOT MAPPED
# =============================================================================
def landmarks_to_fractions(landmarks):
    frac  = np.zeros(16)
    wrist = lm(landmarks, 0)

    # ── INDEX (0-3) ────────────────────────────────────────────────────────
    side_raw = lm(landmarks, 5) - lm(landmarks, 9)
    frac[0]  = float(np.clip( side_raw[0] / MP_MAX_SIDE, -1.0, 1.0))
    frac[1]  = float(np.clip( joint_angle(wrist,            lm(landmarks,5), lm(landmarks,6)) / MP_MAX_MCP, 0, 1))
    frac[2]  = float(np.clip( joint_angle(lm(landmarks,5),  lm(landmarks,6), lm(landmarks,7)) / MP_MAX_PIP, 0, 1))
    frac[3]  = float(np.clip( joint_angle(lm(landmarks,6),  lm(landmarks,7), lm(landmarks,8)) / MP_MAX_DIP, 0, 1))

    # ── MIDDLE (4-7) ───────────────────────────────────────────────────────
    frac[4]  = 0.0
    frac[5]  = float(np.clip( joint_angle(wrist,             lm(landmarks,9),  lm(landmarks,10)) / MP_MAX_MCP, 0, 1))
    frac[6]  = float(np.clip( joint_angle(lm(landmarks,9),   lm(landmarks,10), lm(landmarks,11)) / MP_MAX_PIP, 0, 1))
    frac[7]  = float(np.clip( joint_angle(lm(landmarks,10),  lm(landmarks,11), lm(landmarks,12)) / MP_MAX_DIP, 0, 1))

    # ── RING (8-11) ────────────────────────────────────────────────────────
    # Fraction 0=open, 1=closed — inversion handled by RAW_RANGE being negative
    frac[8]  = 0.0
    frac[9]  = float(np.clip( joint_angle(wrist,             lm(landmarks,13), lm(landmarks,14)) / MP_MAX_MCP, 0, 1))
    frac[10] = float(np.clip( joint_angle(lm(landmarks,13),  lm(landmarks,14), lm(landmarks,15)) / MP_MAX_PIP, 0, 1))
    frac[11] = float(np.clip( joint_angle(lm(landmarks,14),  lm(landmarks,15), lm(landmarks,16)) / MP_MAX_DIP, 0, 1))

    # ── THUMB (12-15) ──────────────────────────────────────────────────────
    frac[12] = 0.0
    frac[13] = float(np.clip( joint_angle(wrist,            lm(landmarks,2), lm(landmarks,3)) / MP_MAX_MCP, 0, 1))
    frac[14] = float(np.clip( joint_angle(lm(landmarks,2),  lm(landmarks,3), lm(landmarks,4)) / MP_MAX_PIP, 0, 1))
    frac[15] = 0.0

    return frac


# =============================================================================
# SMOOTHER
# =============================================================================
class Smoother:
    def __init__(self, alpha=0.4, size=16):
        self.alpha       = alpha
        self.prev        = np.zeros(size)
        self.initialized = False

    def update(self, new_vals):
        if not self.initialized:
            self.prev        = new_vals.copy()
            self.initialized = True
            return self.prev
        self.prev = self.alpha * new_vals + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self):
        self.initialized = False


# =============================================================================
# HUD
# =============================================================================
def draw_hud(frame, fracs, active, hand_detected, fps):
    h, w   = frame.shape[:2]
    color  = (0, 220, 100) if (active and hand_detected) else (60, 60, 200)
    status = "MIRRORING" if (active and hand_detected) else ("PAUSED" if not active else "NO HAND")

    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(frame,
                f"LEAP Teleop  |  {status}  |  {fps:.0f} fps  |  S=pause  Q=quit",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

    finger_names = ['Index', 'Middle', 'Ring', 'Thumb']
    joint_names  = ['side', 'mcp', 'pip', 'dip']
    x0, y0      = w - 185, 50

    cv2.rectangle(frame, (x0-8, y0-20), (w-4, y0+len(fracs)*14+10), (20,20,20), -1)
    cv2.putText(frame, "Curl  0%=open  100%=closed", (x0-20, y0-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1)

    for i, val in enumerate(fracs):
        finger = finger_names[i // 4]
        joint  = joint_names[i % 4]
        pct    = float(abs(val) * 100)
        raw    = fractions_to_raw(fracs)[i]
        label  = f"{finger[0]}.{joint}: {pct:4.0f}%  [{raw:4.0f}]"
        bar_w  = int(pct / 100.0 * 80)
        bar_c  = (100, 200, 100)
        cv2.putText(frame, label, (x0-20, y0+i*14+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200,200,200), 1)
        cv2.rectangle(frame,
                      (x0+90, y0+i*14+3),
                      (x0+90+bar_w, y0+i*14+10),
                      bar_c, -1)

    legend = [
        "Index  -> Index  (0-3)   1800-2800",
        "Middle -> Middle (4-7)   1800-2800",
        "Ring   -> Ring   (8-11)  1800-800 inv",
        "Thumb  -> Thumb  (12-15) update Wizard",
        "Pinky  -> not mapped",
    ]
    lx, ly = 8, h - len(legend)*16 - 8
    cv2.rectangle(frame, (lx-4, ly-14), (lx+260, h-4), (20,20,20), -1)
    for j, line in enumerate(legend):
        cv2.putText(frame, line, (lx, ly+j*16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160,160,160), 1)

    return frame


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Connecting to LEAP Hand...")
    hand = LeapNode()

    hands_detector = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        model_complexity=1
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}")
        return

    cv2.namedWindow("LEAP Hand Teleop", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("LEAP Hand Teleop", cv2.WND_PROP_TOPMOST, 1)

    smoother      = Smoother(alpha=SMOOTHING_ALPHA)
    active        = True
    hand_detected = False
    last_send     = time.time()
    send_interval = 1.0 / SEND_HZ
    fps           = 0.0
    prev_time     = time.time()

    print("\nMirroring is LIVE. Show your hand to the camera.")
    print("S = pause/resume  |  R = reset smoother  |  Q = quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            now       = time.time()
            fps       = 0.9*fps + 0.1*(1.0/max(now-prev_time, 1e-6))
            prev_time = now

            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb)

            hand_detected = results.multi_hand_landmarks is not None

            if hand_detected:
                hand_lm      = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands_mod.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                raw_fracs    = landmarks_to_fractions(hand_lm.landmark)
                smooth_fracs = smoother.update(raw_fracs)

                if active and (now - last_send) >= send_interval:
                    hand.send_fractions(smooth_fracs)
                    last_send = now
            else:
                smooth_fracs = smoother.prev

            frame = draw_hud(frame, smooth_fracs, active, hand_detected, fps)
            cv2.imshow("LEAP Hand Teleop", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                active = not active
                print(f"Mirroring {'RESUMED' if active else 'PAUSED'}")
                if not active:
                    hand.open_hand()
            elif key == ord('r'):
                smoother.reset()
                print("Smoother reset.")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Shutting down — returning to open hand...")
        hand.open_hand()
        time.sleep(0.5)
        cap.release()
        cv2.destroyAllWindows()
        try:    hands_detector.close()
        except: pass
        print("Done.")


if __name__ == "__main__":
    main()