"""
LEAP HAND — VISION TELEOPERATION
==================================
Controls the LEAP Hand by tracking your hand with a webcam + MediaPipe.

CONTROLS:
  S = start / pause mirroring
  Q = quit
  R = reset smoother (if hand drifts)

TUNING:
  SMOOTHING_ALPHA : 0.0 = very smooth but laggy | 1.0 = instant but jittery
  SCALE_BEND      : bigger = more curl on LEAP for same finger movement
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
SCALE_BEND      = 1.4
SCALE_SIDE      = 0.6
CAMERA_INDEX    = 0
SEND_HZ         = 30


# =============================================================================
# LEAP NODE
# =============================================================================
class LeapNode:
    def __init__(self):
        self.kP       = 600
        self.kI       = 0
        self.kD       = 200
        self.curr_lim = 350
        self.motors   = motors = list(range(16))
        self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        for port in ['COM13', 'COM5', 'COM3', '/dev/ttyUSB0', '/dev/ttyUSB1']:
            try:
                print(f"Trying {port}...")
                self.dxl_client = DynamixelClient(motors, port, 4000000)
                self.dxl_client.connect()
                print(f"Connected on {port}!")
                break
            except Exception:
                pass

        # Clear errors and reboot
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
 
        try:
            self.curr_pos = np.array(self.dxl_client.read_pos())
        except:
            self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        print("LEAP Hand ready.")

    def send(self, allegro_pose_16):
        leap_pose = lhu.allegro_to_LEAPhand(np.array(allegro_pose_16), zeros=False)
        self.curr_pos = leap_pose
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def open_hand(self):
        self.send(np.zeros(16))

    def read_pos(self):
        try:    return self.dxl_client.read_pos()
        except: return self.curr_pos


# =============================================================================
# ANGLE MATH
# =============================================================================
def joint_angle(A, B, C):
    v1 = A - B
    v2 = C - B
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.pi - angle)


def lm(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y, p.z])


# =============================================================================
# LANDMARKS -> 16 LEAP JOINTS
# =============================================================================
def landmarks_to_leap(landmarks):
    pose  = np.zeros(16)
    wrist = lm(landmarks, 0)

    # Index (0-3)
    side_vec = lm(landmarks, 5) - lm(landmarks, 9)
    pose[0]  = float(np.clip(side_vec[0] * SCALE_SIDE, -0.35, 0.35))
    pose[1]  = joint_angle(wrist,            lm(landmarks, 5),  lm(landmarks, 6))  * SCALE_BEND
    pose[2]  = joint_angle(lm(landmarks, 5), lm(landmarks, 6),  lm(landmarks, 7))  * SCALE_BEND
    pose[3]  = joint_angle(lm(landmarks, 6), lm(landmarks, 7),  lm(landmarks, 8))  * SCALE_BEND

    # Middle (4-7)
    pose[4]  = 0.0
    pose[5]  = joint_angle(wrist,             lm(landmarks, 9),  lm(landmarks, 10)) * SCALE_BEND
    pose[6]  = joint_angle(lm(landmarks, 9),  lm(landmarks, 10), lm(landmarks, 11)) * SCALE_BEND
    pose[7]  = joint_angle(lm(landmarks, 10), lm(landmarks, 11), lm(landmarks, 12)) * SCALE_BEND

    # Ring (8-11)
    pose[8]  = 0.0
    pose[9]  = joint_angle(wrist,             lm(landmarks, 13), lm(landmarks, 14)) * SCALE_BEND
    pose[10] = joint_angle(lm(landmarks, 13), lm(landmarks, 14), lm(landmarks, 15)) * SCALE_BEND
    pose[11] = joint_angle(lm(landmarks, 14), lm(landmarks, 15), lm(landmarks, 16)) * SCALE_BEND

    # Thumb (12-15)
    pose[12] = 0.0
    pose[13] = joint_angle(wrist,            lm(landmarks, 2), lm(landmarks, 3)) * SCALE_BEND
    pose[14] = joint_angle(lm(landmarks, 2), lm(landmarks, 3), lm(landmarks, 4)) * SCALE_BEND
    pose[15] = 0.0

    # Clamp to safe ranges
    for i in [0, 4, 8, 12]:
        pose[i] = np.clip(pose[i], -0.35, 0.35)
    for i in [1, 2, 5, 6, 9, 10, 13, 14]:
        pose[i] = np.clip(pose[i], 0.0, 1.57)
    for i in [3, 7, 11, 15]:
        pose[i] = np.clip(pose[i], 0.0, 1.0)

    return pose


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
def draw_hud(frame, pose, active, hand_detected, fps):
    h, w   = frame.shape[:2]
    color  = (0, 220, 100) if (active and hand_detected) else (60, 60, 200)
    status = "MIRRORING" if (active and hand_detected) else ("PAUSED" if not active else "NO HAND")

    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(frame,
                f"LEAP Teleop  |  {status}  |  {fps:.0f} fps  |  S=start/stop  Q=quit",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    finger_names = ['Index', 'Middle', 'Ring', 'Thumb']
    joint_names  = ['side', 'mcp', 'pip', 'dip']
    x0, y0      = w - 180, 50

    cv2.rectangle(frame, (x0-8, y0-20), (w-4, y0+len(pose)*14+10), (20,20,20), -1)
    cv2.putText(frame, "Joint angles (deg)", (x0, y0-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

    for i, val in enumerate(pose):
        finger = finger_names[i // 4]
        joint  = joint_names[i % 4]
        deg    = float(np.degrees(val))
        label  = f"{finger[0]}.{joint}: {deg:+5.1f}"
        bar_w  = int(abs(deg) / 90.0 * 80)
        bar_c  = (100, 200, 100) if deg >= 0 else (100, 100, 220)
        cv2.putText(frame, label, (x0, y0+i*14+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
        cv2.rectangle(frame,
                      (x0+95, y0+i*14+3),
                      (x0+95+bar_w, y0+i*14+10),
                      bar_c, -1)
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
    cv2.namedWindow("LEAP Hand Teleop", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("LEAP Hand Teleop", cv2.WND_PROP_TOPMOST, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}")
        return

    smoother      = Smoother(alpha=SMOOTHING_ALPHA)
    active        = True
    hand_detected = False
    last_send     = time.time()
    send_interval = 1.0 / SEND_HZ
    fps           = 0.0
    prev_time     = time.time()

    print("\nCamera open. Mirroring is LIVE. Q to quit. S to pause.\n")

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
                hand_lm = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands_mod.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                raw_pose    = landmarks_to_leap(hand_lm.landmark)
                smooth_pose = smoother.update(raw_pose)

                if active and (now - last_send) >= send_interval:
                    hand.send(smooth_pose)
                    last_send = now
            else:
                smooth_pose = smoother.prev

            frame = draw_hud(frame, smooth_pose, active, hand_detected, fps)
            cv2.imshow("LEAP Hand Teleop", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                active = not active
                print(f"Mirroring {'STARTED' if active else 'PAUSED'}")
                if not active:
                    hand.open_hand()
            elif key == ord('r'):
                smoother.reset()
                print("Smoother reset.")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Shutting down — opening hand...")
        hand.open_hand()
        time.sleep(0.5)
        cap.release()
        cv2.destroyAllWindows()
        try:    hands_detector.close()
        except: pass
        print("Done.")


if __name__ == "__main__":
    main()