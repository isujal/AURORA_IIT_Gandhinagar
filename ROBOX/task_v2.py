import numpy as np
import time
import math
from robot_core.robot_model import Robot
from esp_bridge import ESPActuatorController as ActuatorController
from esp_bridge import OperatingMode

stop_requested = False

def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")


def run_challenge(params):
    global stop_requested
    stop_requested = False

    # ── Oscillation parameters ────────────────────────────────────────────────
    omega       = 1.0    # rad/s — oscillation frequency. Lower = slower/smoother
                         # 0.3 → full cycle every ~21s
                         # 0.5 → full cycle every ~12.5s
                         # 1.0 → full cycle every ~6.3s

    duration    = 60.0   # total run time in seconds
    update_rate = 0.02   # 50 Hz control loop

    # ── Poses (updated) ───────────────────────────────────────────────────────
    pose1 = (math.radians(60),   math.radians(70),  math.radians(-40))
    pose2 = (math.radians(120),  math.radians(-70), math.radians(40))

    # ── Derived sinusoidal parameters per joint ───────────────────────────────
    # center_i    = (pose1_i + pose2_i) / 2
    # amplitude_i = (pose2_i - pose1_i) / 2
    # position(t) = center_i + amplitude_i * sin(omega * t)
    # velocity(t) = amplitude_i * omega * cos(omega * t)   ← what we send

    centers    = {}
    amplitudes = {}
    joint_ids  = [1, 2, 3]

    for i, jid in enumerate(joint_ids):
        centers[jid]    = (pose1[i] + pose2[i]) / 2.0
        amplitudes[jid] = (pose2[i] - pose1[i]) / 2.0

    print("=== Sinusoidal Oscillation Mode ===")
    print(f"omega = {omega} rad/s  |  period = {2*math.pi/omega:.1f}s  |  duration = {duration}s")
    for jid in joint_ids:
        print(f"  J{jid}: center={math.degrees(centers[jid]):+.1f}°  "
              f"amplitude={math.degrees(amplitudes[jid]):+.1f}°  "
              f"range=[{math.degrees(centers[jid]-abs(amplitudes[jid])):+.1f}°, "
              f"{math.degrees(centers[jid]+abs(amplitudes[jid])):+.1f}°]")

    # ── Sign correction (J2 physically flipped) ───────────────────────────────
    signs = { 1: 1, 2: -1, 3: 1 }

    # ── Velocity cap per joint (RPM units) ────────────────────────────────────
    # Max theoretical velocity = amplitude * omega (in rad/s)
    # J1: 0.52 * 0.3 = 0.16 rad/s,  J2: 1.22 * 0.3 = 0.37 rad/s
    # These are well within limits — caps are just safety clamps
# ── Outside loop — replace RPM_MAX for max_vels ──────────────────────────
    RPM_MAX = { 1: 20.0, 2: 35.0, 3: 25.0 }   # proven working values

    # ── Inside the while loop — replace the entire for jid block ─────────────
    for jid in joint_ids:
        cos_val = math.cos(omega * elapsed + phase)   # [-1, +1], smooth direction signal
        vel     = cos_val * RPM_MAX[jid] * signs[jid] # scale to RPM, apply sign correction
        actuator_controller.set_velocity(jid, vel)
    try:
        actuator_controller = ActuatorController("actuator_config.json")

        # ── Hard reset into VELOCITY mode ─────────────────────────────────────
        print("\nHard reset: switching to VELOCITY mode...")
        for jid in joint_ids:
            actuator_controller.disable_torque(jid)
        time.sleep(1.0)

        for jid in joint_ids:
            actuator_controller.change_operating_mode(jid, OperatingMode.VELOCITY)
        time.sleep(0.5)

        for jid in joint_ids:
            actuator_controller.set_velocity(jid, 0.0)
            actuator_controller.enable_torque(jid)
        time.sleep(0.5)

        for jid in joint_ids:
            actuator_controller.set_velocity(jid, 0.0)
        time.sleep(0.3)
        print("Hard reset complete.\n")

        # ── Phase offset: start at pose1 ──────────────────────────────────────
        # sin(omega*t + phase) = -1  at t=0  →  phase = -pi/2
        # This means at t=0, position = center - amplitude = pose1
        phase = -math.pi / 2

        print("Starting sinusoidal oscillation...")
        print(f"{'t':>6s} | {'J1 pos':>8s} {'J1 vel':>8s} | "
              f"{'J2 pos':>8s} {'J2 vel':>8s} | "
              f"{'J3 pos':>8s} {'J3 vel':>8s}")
        print("-" * 75)

        start_time = time.time()

        while (time.time() - start_time) < duration:
            if stop_requested:
                print("Stop requested.")
                break

            elapsed = time.time() - start_time

            for jid in joint_ids:
                # Sinusoidal velocity command (derivative of desired position)
                # vel(t) = amplitude * omega * cos(omega * t + phase)
                raw_vel = amplitudes[jid] * omega * math.cos(omega * elapsed + phase)

                # Convert rad/s to RPM-equivalent units and apply sign + clamp
                vel = math.copysign(
                    min(abs(raw_vel * (180.0 / math.pi)), max_vels[jid]),
                    raw_vel
                ) * signs[jid]

                actuator_controller.set_velocity(jid, vel)

            # ── Status print every ~2s ────────────────────────────────────────
            if int(elapsed * 10) % 20 == 0:
                p = [math.degrees(actuator_controller.relative_joint_angle(jid))
                     for jid in joint_ids]
                v = [amplitudes[jid] * omega * math.cos(omega * elapsed + phase)
                     for jid in joint_ids]
                print(f"{elapsed:6.1f}s | "
                      f"{p[0]:+7.1f}° {math.degrees(v[0]):+7.2f}°/s | "
                      f"{p[1]:+7.1f}° {math.degrees(v[1]):+7.2f}°/s | "
                      f"{p[2]:+7.1f}° {math.degrees(v[2]):+7.2f}°/s")

            time.sleep(update_rate)

        return {
            "completed_successfully": not stop_requested,
            "final_j1_deg": math.degrees(actuator_controller.relative_joint_angle(1)),
            "final_j2_deg": math.degrees(actuator_controller.relative_joint_angle(2)),
            "final_j3_deg": math.degrees(actuator_controller.relative_joint_angle(3)),
        }

    except Exception as e:
        print(f"Error: {e}")
        raise

    finally:
        if 'actuator_controller' in locals():
            try:
                for jid in joint_ids:
                    actuator_controller.set_velocity(jid, 0.0)
                time.sleep(0.2)
                for jid in joint_ids:
                    actuator_controller.disable_torque(jid)
                print("Cleanup done — torque disabled.")
            except:
                pass

if __name__ == "__main__":
    print(run_challenge(params={}))