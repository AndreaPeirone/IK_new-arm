import time
# import json
from dynamixel_sdk import *  # Dynamixel SDK library
import keyboard
import numpy as np

# Dynamixel settings
DEVICENAME = "COM13"  # Change if needed
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11  # Extended position mode for multi-turn
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
EXTENDED_POSITION_MODE = 4  # Enables multi-turn mode

# Dynamixel IDs
# Order: [gripper (ID 10), section 1: base (ID 11), bend1 (ID 12), bend2 (ID 13),
#         section 2: base (ID 14), bend (ID 15), section 3: base (ID 16), bend (ID 17)]

MOTOR_IDs = np.array([10, 11, 12, 13, 14, 15, 16, 17, 20])
DYNAMIXEL_IDS = MOTOR_IDs.tolist()
DXL_STEP = 50  # Step size for movement

# Initialize port handler & packet handler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort() and portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to open the port and set baudrate!")
else:
    print("Failed to open the port")
    exit()

# Set extended position mode & enable torque for each motor
for dxl_id in DYNAMIXEL_IDS:
    # Disable torque before changing operating mode
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    time.sleep(0.1)  # Slight delay for safety
    # Set extended position mode
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
    time.sleep(0.1)
    # Re-enable torque
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    time.sleep(0.1)

# Get the initial positions for all motors and store in offset array.
offset = []
for dxl_id in DYNAMIXEL_IDS:
    pos, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
    offset.append(pos)
print(f"Initial positions (multi-turn raw values): {offset}")

# Start with a default selected motor; we'll use keys 0-7 to select motors based on the order in DYNAMIXEL_IDS.
selected_motor = DYNAMIXEL_IDS[0]

def move_motor(motor_id, step):
    """Move selected motor based on its current position plus step."""
    global offset
    # Read current position
    pos, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)
    new_position = pos + step
    # Find the index in the offset array corresponding to motor_id
    index = DYNAMIXEL_IDS.index(motor_id)
    offset[index] = new_position  # update offset with new value
    packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_POSITION, new_position)
    print(f"Motor {motor_id} moved to {new_position}")

print("\nControls:")
print("  Number keys (0-7) to select motor in this order:")
print("    0 -> gripper (ID 10)")
print("    1 -> section 1 base (ID 11)")
print("    2 -> section 1 bend1 (ID 12)")
print("    3 -> section 1 bend2 (ID 13)")
print("    4 -> section 2 base (ID 14)")
print("    5 -> section 2 bend (ID 15)")
print("    6 -> section 3 base (ID 16)")
print("    7 -> section 3 bend (ID 17)")
print("    8 -> pitch base joint (ID 20)")
print("  x/c to move the selected motor.")
print("  Press ESC to finish calibration, print results, and generate the configuration JSON.\n")

# Main loop for keyboard control
running = True
while running:
    # Check for motor selection keys 0-7
    for i in range(len(DYNAMIXEL_IDS)):
        if keyboard.is_pressed(str(i)):
            selected_motor = DYNAMIXEL_IDS[i]
            print(f"Selected motor: {selected_motor}")
            time.sleep(0.2)  # Debounce

    # Adjust selected motor position with arrow keys
    if keyboard.is_pressed("x"):
        move_motor(selected_motor, DXL_STEP)
        time.sleep(0.05)
    elif keyboard.is_pressed("c"):
        move_motor(selected_motor, -DXL_STEP)
        time.sleep(0.05)

    # Exit the loop when ESC is pressed
    if keyboard.is_pressed("esc"):
        running = False

print("\nEnding calibration...")
print(f"Final offset values (multi-turn raw): {offset}")