from spacemouse_class import SpaceMouse
import time
import numpy as np


#test and find the spacemouse
import pywinusb.hid as hid
import threading
import time

spacemouse_data = [0] * 8
data_received = False

def handle_input_factory():
    def handler(data):
        global spacemouse_data, data_received
        report_id = data[0]
        data_received = True

        if report_id == 1 and len(data) >= 13:
            x = int.from_bytes(data[1:3], byteorder='little', signed=True)
            y = int.from_bytes(data[3:5], byteorder='little', signed=True)
            z = int.from_bytes(data[5:7], byteorder='little', signed=True)
            roll = int.from_bytes(data[7:9], byteorder='little', signed=True)
            pitch = int.from_bytes(data[9:11], byteorder='little', signed=True)
            yaw = int.from_bytes(data[11:13], byteorder='little', signed=True)
            spacemouse_data[0:6] = [x, y, z, roll, pitch, yaw]

        elif report_id == 3 and len(data) >= 2:
            buttons = data[1]
            spacemouse_data[6] = buttons & 1
            spacemouse_data[7] = (buttons >> 1) & 1

        print("SpaceMouse Data:", spacemouse_data)
    return handler

def try_device(device, timeout=3):
    global data_received
    data_received = False
    try:
        device.open()
        device.set_raw_data_handler(handle_input_factory())
        print(f"Testing device: {device.vendor_name} - {device.product_name}")
        time.sleep(timeout)
        return data_received
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        device.close()

def main():
    devices = hid.HidDeviceFilter(vendor_id=0x256F, product_id=0xC652).get_devices()

    if not devices:
        print("No 3Dconnexion devices found.")
        return

    for idx, device in enumerate(devices):
        print(f"Trying device #{idx}...")
        if try_device(device):
            print(f"✅ Found working device at index {idx}")
            # Reopen and run indefinitely
            device.open()
            device.set_raw_data_handler(handle_input_factory())
            try:
                print("Now listening. Move SpaceMouse or press buttons. Ctrl+C to exit.")
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Exiting...")
            finally:
                device.close()
            return

    print("❌ Could not find an active SpaceMouse interface.")

if __name__ == "__main__":
    main()








# ###TEST SPACEMOUSE CLASS AND EVALUATION###
# def update_goals_with_spacemouse(xyz_goal, sm_data, translation_gain, rotation_gain):
#     dx, dy, dz, droll, dpitch, dyaw = sm_data[:6]

#     xyz_goal[0] += dx * translation_gain
#     xyz_goal[1] += dy * translation_gain
#     xyz_goal[2] += dz * translation_gain
#     xyz_goal[3] += dx * translation_gain
#     xyz_goal[4] += dy * translation_gain
#     xyz_goal[5] += dz * translation_gain
#     xyz_goal[3:6] = update_gripper_position(xyz_goal[:3], xyz_goal[3:6], droll * rotation_gain, dpitch * rotation_gain, dyaw * rotation_gain)
    
#     return xyz_goal

# def update_gripper_position(P, Q, delta_theta_x_degrees, delta_theta_y_degrees, delta_theta_z_degrees):
#     Q_x, Q_y, Q_z = Q
#     delta_theta_x = np.deg2rad(delta_theta_x_degrees)/2
#     delta_theta_y = np.deg2rad(delta_theta_y_degrees)/2
#     delta_theta_z = np.deg2rad(delta_theta_z_degrees)/2

#     Rx_increment = np.array([[1, 0, 0],
#                              [0, np.cos(delta_theta_x), -np.sin(delta_theta_x)],
#                              [0, np.sin(delta_theta_x), np.cos(delta_theta_x)]])
    
#     Ry_increment = np.array([[np.cos(delta_theta_y), 0, np.sin(delta_theta_y)],
#                              [0, 1, 0],
#                              [-np.sin(delta_theta_y), 0, np.cos(delta_theta_y)]])

#     Rz_increment = np.array([[np.cos(delta_theta_z), -np.sin(delta_theta_z), 0],
#                              [np.sin(delta_theta_z), np.cos(delta_theta_z), 0],
#                              [0, 0, 1]])

#     R_increment = np.dot(Rz_increment, np.dot(Ry_increment, Rx_increment))
#     direction = np.array([Q_x - P[0], Q_y - P[1], Q_z - P[2]])
#     rotated_direction = np.dot(R_increment, direction)
#     updated_Q = P + rotated_direction
#     return updated_Q
# # ============================
# # 🚀 Main Loop for Live Testing
# # ============================
# if __name__ == "__main__":
#     sm = SpaceMouse()  # Adjust index if needed
#     xyz_goal = [0.0, 0.0, 0.0, 0, 0, 50]  # Initial position and direction
#     translation_gain = 1  # Scales for fine control
#     rotation_gain = 1     # Same here — tweak as needed

#     try:
#         sm.start()
#         print("Streaming xyz_goal... Move the SpaceMouse to update. Press Ctrl+C to stop.")
#         while sm.is_running():
#             sm_data = sm.get_array()  # [x, y, z, roll, pitch, yaw, btn1, btn2]
#             xyz_goal = update_goals_with_spacemouse(xyz_goal, sm_data, translation_gain, rotation_gain)
#             print("xyz_goal:", np.round(xyz_goal, 3))  # Rounded for readability
#             time.sleep(0.05)
#     except KeyboardInterrupt:
#         print("User stopped the stream.")
#     finally:
#         sm.stop()








# #TEST PSCONTROLLER
# import pygame
# import time

# # Config
# DRIFT_THRESHOLD = 0.05  # Treat values within this range as zero

# # Initialize
# pygame.init()
# pygame.joystick.init()

# if pygame.joystick.get_count() == 0:
#     print("No joystick connected!")
#     exit()

# joystick = pygame.joystick.Joystick(0)
# joystick.init()

# print(f"Initialized controller: {joystick.get_name()}")

# # --- Step 1: Calibrate Rest State ---
# print("Calibrating... Do not touch the controller.")
# time.sleep(2)  # Give user time to release the controller

# pygame.event.pump()
# axis_count = joystick.get_numaxes()
# offset_ps = [joystick.get_axis(i) for i in range(axis_count)]

# print(f"offset_ps (calibrated rest position): {offset_ps}")
# print("Starting input loop...\n")

# # --- Step 2: Start Input Loop ---
# try:
#     while True:
#         pygame.event.pump()

#         # Axes: subtract offset and apply dead zone
#         raw_axes = [joystick.get_axis(i) for i in range(axis_count)]
#         axes = []
#         for i in range(axis_count):
#             val = raw_axes[i] - offset_ps[i]
#             if abs(val) < DRIFT_THRESHOLD:
#                 val = 0.0
#             axes.append(round(val, 3))  # Round for cleaner display

#         # Buttons
#         buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

#         # D-pad
#         hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

#         print("Axes:", axes)
#         print("Buttons:", buttons)
#         print("Hats (D-pad):", hats)
#         print("-" * 40)

#         pygame.time.wait(100)

# except KeyboardInterrupt:
#     print("Exiting...")

# finally:
#     pygame.quit()

