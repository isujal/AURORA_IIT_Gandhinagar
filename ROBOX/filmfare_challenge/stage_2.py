import numpy as np
import math
import time

from esp_bridge import ESPActuatorController as ActuatorController
from robot_core.robot_model import Robot

stop_requested = False


def stop_code():
    global stop_requested
    stop_requested = True
    print("Stop requested.")


def run_challenge(params):

    global stop_requested
    stop_requested = False

    actuator_controller = None

    try:

        robot = Robot.from_config("robot_parameters.json")
        actuator_controller = ActuatorController("actuator_config.json")

        joint1_id = 1
        joint2_id = 2
        joint3_id = 3

        # -------------------------------------------------
        # USER PARAMETERS
        # -------------------------------------------------

        sweep_duration = 25.0

        # 90 deg sweep in 25 s
        joint1_velocity_rad_s = 0.15

        target_x = 0.0
        target_y = 0.25

        update_rate = 0.02
         
        #Init Pose 
        q_init = [
            0.288,
            1.243,
            0.661
        ]

        # -------------------------------------------------
        # J1 VELOCITY MODE
        # -------------------------------------------------

        actuator_controller.disable_torque(joint1_id)
        time.sleep(0.05)

        actuator_controller.change_operating_mode(
            joint1_id,
            1
        )

        time.sleep(0.05)

        actuator_controller.enable_torque(joint1_id)

        # -------------------------------------------------
        # J2 POSITION MODE
        # -------------------------------------------------

        actuator_controller.disable_torque(joint2_id)
        time.sleep(0.05)

        actuator_controller.change_operating_mode(
            joint2_id,
            3
        )

        time.sleep(0.05)

        actuator_controller.disable_torque(joint2_id)

        # -------------------------------------------------
        # J3 POSITION MODE
        # -------------------------------------------------

        actuator_controller.disable_torque(joint3_id)
        time.sleep(0.05)

        actuator_controller.change_operating_mode(
            joint3_id,
            3
        )

        time.sleep(0.05)

        actuator_controller.enable_torque(joint3_id)

        # -------------------------------------------------
        # GO TO INITIAL POSITION FIRST
        # -------------------------------------------------

        raw2 = actuator_controller.relative_joint_angle_to_raw(
            joint2_id,
            q_init[1]
        )

        raw3 = actuator_controller.relative_joint_angle_to_raw(
            joint3_id,
            q_init[2]
        )

        actuator_controller.set_position(joint2_id, raw2)
        actuator_controller.set_position(joint3_id, raw3)

        time.sleep(2.0)

        # move J1 last

        actuator_controller.disable_torque(joint1_id)
        time.sleep(0.05)

        actuator_controller.change_operating_mode(
            joint1_id,
            3
        )

        time.sleep(0.05)

        actuator_controller.enable_torque(joint1_id)

        raw1 = actuator_controller.relative_joint_angle_to_raw(
            joint1_id,
            q_init[0]
        )

        actuator_controller.set_position(
            joint1_id,
            raw1
        )

        time.sleep(2.0)

        print("Initial pose reached.")

        # -------------------------------------------------
        # SWITCH J1 TO VELOCITY MODE
        # -------------------------------------------------

        actuator_controller.disable_torque(joint1_id)
        time.sleep(0.05)

        actuator_controller.change_operating_mode(
            joint1_id,
            1
        )

        time.sleep(0.05)

        actuator_controller.enable_torque(joint1_id)

        velocity_raw = actuator_controller.convert_velocity_to_raw(
            joint1_id,
            joint1_velocity_rad_s
        )

        actuator_controller.set_velocity(
            joint1_id,
            velocity_raw
        )

        # -------------------------------------------------
        # START SWEEP
        # -------------------------------------------------

        start_time = time.time()

        while True:

            if stop_requested:
                break

            elapsed = time.time() - start_time

            if elapsed >= sweep_duration:
                break

            q1 = actuator_controller.relative_joint_angle(
                joint1_id
            )

            q2 = actuator_controller.relative_joint_angle(
                joint2_id
            )

            q3 = actuator_controller.relative_joint_angle(
                joint3_id
            )

            current_q = [q1, q2, q3]

            T = robot.forward_kinematics(current_q)

            ee_x = T[0, 3]
            ee_y = T[1, 3]

            # ---------------------------------------------
            # FIRST TWO QUADRANTS ONLY
            # ---------------------------------------------

            if ee_y < 0:

                actuator_controller.set_velocity(
                    joint1_id,
                    0
                )

                print("Robot left upper half plane.")
                break

            # ---------------------------------------------
            # LINE EQUATION APPROACH
            # ---------------------------------------------

            dx = target_x - ee_x
            dy = target_y - ee_y

            tool_angle = math.atan2(
                dy,
                dx
            )

            q3_target = tool_angle - q1 - q2

            q3_raw = actuator_controller.relative_joint_angle_to_raw(
                joint3_id,
                q3_target
            )

            actuator_controller.set_position(
                joint3_id,
                q3_raw
            )

            print(
                f"x={ee_x:.4f} "
                f"y={ee_y:.4f} "
                f"tool_angle={np.degrees(tool_angle):.2f}"
            )

            time.sleep(update_rate)

        actuator_controller.set_velocity(
            joint1_id,
            0
        )

    except Exception as e:

        print(f"Error occurred: {e}")
        raise

    finally:

        try:

            if actuator_controller is not None:

                try:
                    actuator_controller.set_velocity(
                        joint1_id,
                        0
                    )
                except:
                    pass

                for jid in [1, 2, 3]:

                    try:
                        actuator_controller.disable_torque(
                            jid
                        )
                    except:
                        pass

                actuator_controller.close()

        except Exception as e:

            print(
                f"Cleanup error: {e}"
            )


if __name__ == "__main__":
    run_challenge({})

