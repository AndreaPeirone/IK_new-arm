import time
import pickle
import keyboard
import numpy as np

# -------------------------------------------------------------------------
# Import your controller classes and exception from dynamixel_controller.py
# (Adjust the module name if yours is different.)
# -------------------------------------------------------------------------
from dynamixel_controller import DynamixelController, BaseModel, PortCommError

# -------------------------------------------------------------------------
# Configuration (same as before)
# -------------------------------------------------------------------------
DEVICENAME = "COM13"         # Change if needed
BAUDRATE    = 57600
PROTOCOL    = 2.0

DXL_STEP    = 50             # how many encoder ticks to move per keypress
MOTOR_IDs   = [10, 11, 12, 13, 14, 15, 16, 17, 20]  # [gripper, base1, bend1, bend2, base2, bend2_2, base3, bend3]

# -------------------------------------------------------------------------
# Build a list of BaseModel instances, one per motor ID.
# -------------------------------------------------------------------------
motor_list = []
for mid in MOTOR_IDs:
    m = BaseModel(motor_id=mid)
    motor_list.append(m)

# -------------------------------------------------------------------------
# Instantiate the controller and activate it.
# -------------------------------------------------------------------------
controller = DynamixelController(
    port_name       = DEVICENAME,
    motor_list      = motor_list,
    protocol        = PROTOCOL,
    baudrate        = BAUDRATE,
    latency_time    = 1,
    reverse_direction=False
)

try:
    controller.activate_controller()
except PortCommError as e:
    print(f"[ERROR] Could not activate controller: {e}")
    exit(1)

# -------------------------------------------------------------------------
# Put all motors into Extended Position Mode (multi-turn), and enable torque.
# -------------------------------------------------------------------------
controller.torque_off()
controller.set_operating_mode_all("extended_position_control")
time.sleep(0.1)
controller.torque_on()
time.sleep(0.1)

# -------------------------------------------------------------------------
# Read initial raw encoder positions (multi-turn values) for every motor
# by calling read_info(fast_read=False) (to use txRxPacket() under the hood).
# -------------------------------------------------------------------------
try:
    # We set fast_read=False so that `read_info` calls txRxPacket() directly.
    pos_list_raw, _, _, _ = controller.read_info(fast_read=False)
except PortCommError as e:
    print(f"[ERROR] Failed to read initial positions: {e}")
    controller.torque_off()
    exit(1)

offset = pos_list_raw.tolist()
print(f"Initial positions (multi-turn raw values): {offset}")

# -------------------------------------------------------------------------
# Keyboard-control loop: select one of the 8 motors (0–7) then move it ±DXL_STEP.
# -------------------------------------------------------------------------
selected_index = 0
selected_motor = MOTOR_IDs[selected_index]

print("\nControls:")
print("  Number keys (0–7) to select motor in this order:")
print("    0 → gripper (ID 10)")
print("    1 → section 1 base (ID 11)")
print("    2 → section 1 bend1 (ID 12)")
print("    3 → section 1 bend2 (ID 13)")
print("    4 → section 2 base (ID 14)")
print("    5 → section 2 bend (ID 15)")
print("    6 → section 3 base (ID 16)")
print("    7 → section 3 bend (ID 17)")
print("    8 → pitch (ID 20)")
print(f"  x/c to move the selected motor by ±{DXL_STEP} ticks.")
print("  ESC → finish calibration, print results, and save offset.pkl\n")

running = True

def move_motor_by_ticks(index, delta_ticks):
    """
    index: 0–7, which motor in MOTOR_IDs list
    delta_ticks: integer, positive or negative
    """
    global offset
    offset[index] += delta_ticks
    # Send a sync-write of all eight goal positions (raw encoders).
    # set_goal_position expects a list of raw‐encoder ints, length == number of motors.
    controller.set_goal_position(offset)
    mid = MOTOR_IDs[index]
    print(f"Motor {mid} moved to raw {offset[index]}")

while running:
    # Check keys 0–7 to reselect the current motor
    for i in range(len(MOTOR_IDs)):
        if keyboard.is_pressed(str(i)):
            selected_index = i
            selected_motor = MOTOR_IDs[selected_index]
            print(f"Selected motor: ID {selected_motor} (index {selected_index})")
            time.sleep(0.2)  # debounce

    # “x” to increment, “c” to decrement
    if keyboard.is_pressed("x"):
        move_motor_by_ticks(selected_index, +DXL_STEP)
        time.sleep(0.05)
    elif keyboard.is_pressed("c"):
        move_motor_by_ticks(selected_index, -DXL_STEP)
        time.sleep(0.05)

    # ESC to exit
    if keyboard.is_pressed("esc"):
        running = False

# -------------------------------------------------------------------------
# After exiting: print and save the offset array
# -------------------------------------------------------------------------
print("\nEnding calibration...")
print("Final offset (raw ticks) per motor:", offset)

with open("offset.pkl", "wb") as f:
    pickle.dump(offset, f)
print("Offset saved to offset.pkl")

# from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
# from dynamixel_controller import BaseModel

# PORT = PortHandler("COM13")
# PACKET = PacketHandler(2.0)

# assert PORT.openPort(), "Cannot open port"
# assert PORT.setBaudRate(57600), "Cannot set baud"

# ID = 20
# GOAL_POS = 0  # +100 ticks from center
# TORQUE_ON = 1

# # enable torque
# err, e = PACKET.write1ByteTxRx(PORT, ID, BaseModel.torque_enable.address, TORQUE_ON)
# if err != COMM_SUCCESS or e != 0:
#     print("Torque-on failed", err, e)

# # set position
# err, e = PACKET.write4ByteTxRx(PORT, ID, BaseModel.goal_position.address, GOAL_POS)
# if err != COMM_SUCCESS or e != 0:
#     print("Pos write failed", err, e)

# # wait & read back
# import time; time.sleep(0.5)
# pos = PACKET.read4ByteTxRx(PORT, ID, BaseModel.present_position.address)
# print("Read back position:", pos)
