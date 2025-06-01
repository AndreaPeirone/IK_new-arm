import numpy as np
from sympy import symbols, cos, sin, Matrix, lambdify, pi
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
import pickle
from motor_mapping_class import MotorMapping
import keyboard
from dynamixel_sdk import *  # Dynamixel SDK library
import time

#------------------------------------------------------------MOTORS SETUP------------------------------------------------------------

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
ADDR_PROFILE_VELOCITY = 112  # X-series models (confirm in your motor's control table)
ADDR_MOVING_SPEED = 32  # Protocol 1.0 motors like AX-12

MOTOR_IDs = np.array([10, 11, 12, 13, 14, 15, 16, 17])
DYNAMIXEL_IDS = MOTOR_IDs.tolist()

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


# Example speed (in ticks per second). Lower = slower. Adjust this as needed.
slow_velocity = 50
for motor_id in DYNAMIXEL_IDS:
    # Write speed to motor (adjust address for your model!)
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
        portHandler, motor_id, ADDR_PROFILE_VELOCITY, slow_velocity
    )

    if dxl_comm_result != COMM_SUCCESS:
        print(f"Failed to set speed for Motor {motor_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Dynamixel error on Motor {motor_id}: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Set slow speed for Motor {motor_id}")

#------------------------------------------------------------STORE CALIBRATION------------------------------------------------------------
# Get the initial positions for all motors and store in offset array.
offset = []
for dxl_id in DYNAMIXEL_IDS:
    pos, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
    offset.append(pos)
print(f"Initial positions (multi-turn raw values): {offset}")

#------------------------------------------------------------LOAD MAPPING------------------------------------------------------------
# Load the combined mapping
with open('combined_motor_mapping.pkl', 'rb') as f:
    motor_mapping = pickle.load(f)

#------------------------------------------------------------FUNCTIONS------------------------------------------------------------
#Geometric parameters
L1 = 305 #height 1st section [mm]
L2 = 225 #height 2nd section [mm]
L3 = 228 #height 3rd section [mm]
d = 120  #Tuned diameter of the base [mm]
offset_gripper = 50 #height of the gripper centre [mm]
rigid_support = 55 #height of the actuation between each section [mm]

# Define symbolic variables
delta1, rot1, delta2, rot2, delta3, rot3 = symbols('delta1 rot1 delta2 rot2 delta3 rot3')

def Jacobian():
    # --- Section 1 ---
    R1 = Matrix([
        [cos(rot1)**2*(cos(delta1)-1)+1, sin(rot1)*cos(rot1)*(cos(delta1)-1), cos(rot1)*sin(delta1)],
        [sin(rot1)*cos(rot1)*(cos(delta1)-1), sin(rot1)**2*(cos(delta1)-1)+1, sin(rot1)*sin(delta1)],
        [-cos(rot1)*sin(delta1), -sin(rot1)*sin(delta1), cos(delta1)]
    ])
    t1 = Matrix([
        L1/delta1 * cos(rot1)*(1-cos(delta1)),
        L1/delta1 * sin(rot1)*(1-cos(delta1)),
        L1/delta1 * sin(delta1)
    ])
    
    offset1 = Matrix([0, 0, rigid_support])
    start_section2 = t1 + R1 * offset1
    
    # --- Section 2 ---
    R12 = Matrix([
        [cos(rot1 + pi + rot2)**2*(cos(delta2)-1)+1, sin(rot1 + pi + rot2)*cos(rot1 + pi + rot2)*(cos(delta2)-1), cos(rot1 + pi + rot2)*sin(delta2)],
        [sin(rot1 + pi + rot2)*cos(rot1 + pi + rot2)*(cos(delta2)-1), sin(rot1 + pi + rot2)**2*(cos(delta2)-1)+1, sin(rot1 + pi + rot2)*sin(delta2)],
        [-cos(rot1 + pi + rot2)*sin(delta2), -sin(rot1 + pi + rot2)*sin(delta2), cos(delta2)]
    ])
    t12 = Matrix([
        L2/delta2 * cos(rot1 + pi + rot2)*(1-cos(delta2)),
        L2/delta2 * sin(rot1 + pi + rot2)*(1-cos(delta2)),
        L2/delta2 * sin(delta2)
    ])
    # --- Correction: use R1 instead of R12 for the transformation ---
    t2 = start_section2 + R1 * t12
    R2 = R1 * R12
    
    offset2 = Matrix([0, 0, rigid_support])
    start_section3 = t2 + R2 * offset2
    
    # --- Section 3 ---
    R23 = Matrix([
        [cos(rot1 + pi + rot2 + pi + rot3)**2*(cos(delta3)-1)+1, sin(rot1 + pi + rot2 + pi + rot3)*cos(rot1 + pi + rot2 + pi + rot3)*(cos(delta3)-1), cos(rot1 + pi + rot2 + pi + rot3)*sin(delta3)],
        [sin(rot1 + pi + rot2 + pi + rot3)*cos(rot1 + pi + rot2 + pi + rot3)*(cos(delta3)-1), sin(rot1 + pi + rot2 + pi + rot3)**2*(cos(delta3)-1)+1, sin(rot1 + pi + rot2 + pi + rot3)*sin(delta3)],
        [-cos(rot1 + pi + rot2 + pi + rot3)*sin(delta3), -sin(rot1 + pi + rot2 + pi + rot3)*sin(delta3), cos(delta3)]
    ])
    t23 = Matrix([
        L3/delta3 * cos(rot1 + pi + rot2 + pi + rot3)*(1-cos(delta3)),
        L3/delta3 * sin(rot1 + pi + rot2 + pi + rot3)*(1-cos(delta3)),
        L3/delta3 * sin(delta3)
    ])
    t3 = start_section3 + R2 * t23
    R3 = R2 * R23

    # --- Gripper Offset ---
    t34 = Matrix([0, 0, offset_gripper])
    t3_gripper = t3 + R3 * t34

    # Stack key points: start_section2, start_section3, t3, and t3_gripper
    t_total = Matrix.vstack(t3, t3_gripper)

    variables_vector = Matrix([delta1, rot1, delta2, rot2, delta3, rot3])
    J = t_total.jacobian(variables_vector)
    return lambdify((delta1, rot1, delta2, rot2, delta3, rot3), J, modules=['numpy'])

def cartesian_configuration(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv):
    e = xyz_goal - xyz_old
    print("error", e)
    J = result_J(delta_rot_goal[0], delta_rot_goal[1], 
                 delta_rot_goal[2], delta_rot_goal[3], 
                 delta_rot_goal[4], delta_rot_goal[5])

    # J_inv = np.linalg.pinv(J, rcond=1e-5)
    J_transposed = J.T
    JK_inv = np.dot(J, K_inv)
    JK_inv_J_transposed = np.dot(JK_inv, J_transposed)
    inv_JK_inv_J_transposed = np.linalg.pinv(JK_inv_J_transposed, rcond=1e-5)
    J_inv = np.dot(K_inv, np.dot(J_transposed, inv_JK_inv_J_transposed))
    delta_rot_goal += np.dot(J_inv, e)
    
    print("delta_rot_current before boundaries", delta_rot_goal)
    delta_rot_goal, K_inv = check_boundaries(delta_rot_goal, K_inv)

    end_effector = result_end_effector(delta_rot_goal[0], delta_rot_goal[1], 
                                       delta_rot_goal[2], delta_rot_goal[3], 
                                       delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
    e = xyz_goal - end_effector
    print("xyz_goal", xyz_goal)
    print("end_effector evaluation", end_effector)
    print("error", e)
    print("norm error", norm(e))
    print("stiffness matrix", np.diag(K_inv))
    count = 0
    while np.linalg.norm(e) > threshold:
        count += 1
        print("\niteration:", count)
        J = result_J(delta_rot_goal[0], delta_rot_goal[1], 
                     delta_rot_goal[2], delta_rot_goal[3], 
                     delta_rot_goal[4], delta_rot_goal[5])
                
        # J_inv = np.linalg.pinv(J, rcond=1e-5)
        J_transposed = J.T
        JK_inv = np.dot(J, K_inv)
        JK_inv_J_transposed = np.dot(JK_inv, J_transposed)
        inv_JK_inv_J_transposed = np.linalg.pinv(JK_inv_J_transposed, rcond=1e-5)
        J_inv = np.dot(K_inv, np.dot(J_transposed, inv_JK_inv_J_transposed))
        delta_rot_goal += np.dot(J_inv, e)

        print("delta_rot_current before boundaries", delta_rot_goal)
        delta_rot_goal, K_inv = check_boundaries(delta_rot_goal, K_inv)

        end_effector = result_end_effector(delta_rot_goal[0], delta_rot_goal[1], 
                                           delta_rot_goal[2], delta_rot_goal[3], 
                                           delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
        print("xyz_goal", xyz_goal)
        print("end_effector evaluation", end_effector)
        e = xyz_goal - end_effector
        print("error", e)
        print("norm error", norm(e))
        print("stiffness matrix", np.diag(K_inv))

        if count == 10:
            print("\nSATURATION OF ITERATIONS REACHED:", count)
            delta_rot_goal = delta_rot_old.copy()
            end_effector = xyz_old.copy()
            break

    return end_effector, delta_rot_goal

def cartesian_configuration_noconv(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv):
    e = xyz_goal - xyz_old
    print("error", e)
    J = result_J(delta_rot_goal[0], delta_rot_goal[1], 
                 delta_rot_goal[2], delta_rot_goal[3], 
                 delta_rot_goal[4], delta_rot_goal[5])

    # J_inv = np.linalg.pinv(J, rcond=1e-5)
    J_transposed = J.T
    JK_inv = np.dot(J, K_inv)
    JK_inv_J_transposed = np.dot(JK_inv, J_transposed)
    inv_JK_inv_J_transposed = np.linalg.pinv(JK_inv_J_transposed, rcond=1e-5)
    J_inv = np.dot(K_inv, np.dot(J_transposed, inv_JK_inv_J_transposed))
    delta_rot_goal += np.dot(J_inv, e)
    
    print("delta_rot_current before boundaries", delta_rot_goal)
    delta_rot_goal, K_inv = check_boundaries(delta_rot_goal, K_inv)

    end_effector = result_end_effector(delta_rot_goal[0], delta_rot_goal[1], 
                                       delta_rot_goal[2], delta_rot_goal[3], 
                                       delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
    e = xyz_goal - end_effector
    print("xyz_goal", xyz_goal)
    print("end_effector evaluation", end_effector)
    print("error", e)
    print("norm error", norm(e))
    print("stiffness matrix", np.diag(K_inv))

    if norm(e) > 10:
        print("\nPOINT NON REACHABLE:")
        delta_rot_goal = delta_rot_old.copy()
        end_effector = xyz_old.copy()

    return end_effector, delta_rot_goal

def check_boundaries(delta_rot_goal, K_inv_original):
    K_inv = K_inv_original.copy()
    eps = 1e-2

    if delta_rot_goal[0] > section_limits[0]:
        print("----------------------------------Max bending on 1st section reached----------------------------------")
        delta_rot_goal[0] = section_limits[0]
        K_inv [0,0] = K_inv[0,0] / 1e6

    if delta_rot_goal[2] > section_limits[1]:
        print("----------------------------------Max bending on 2nd section reached----------------------------------")
        delta_rot_goal[2] = section_limits[1]
        K_inv [2,2] = K_inv[2,2] / 1e6
    
    if delta_rot_goal[4] > section_limits[2]:
        print("----------------------------------Max bending on 3rd section reached----------------------------------")
        delta_rot_goal[4] = section_limits[2]
        K_inv [4,4] = K_inv[4,4] / 1e6 

    if np.isnan(delta_rot_goal[0]):
        delta_rot_goal[0] = eps
    if np.isnan(delta_rot_goal[2]):
        delta_rot_goal[2] = eps
    if np.isnan(delta_rot_goal[4]):
        delta_rot_goal[4] = eps

    if delta_rot_goal[1] < eps:
        # delta_rot_goal[1] = 1e-10
        K_inv [1,1] = K_inv[1,1] / 1e10
    if delta_rot_goal[3] < eps:
        # delta_rot_goal[3] = 1e-10
        K_inv [3,3] = K_inv[3,3] / 1e10
    if delta_rot_goal[5] < eps:
        # delta_rot_goal[5] = 1e-10
        K_inv [5,5] = K_inv[5,5] / 1e10

    print("delta_rot_current after boundaries", delta_rot_goal)
    return delta_rot_goal, K_inv

def end_effector_position():
    # --- Section 1 ---
    R1 = Matrix([
        [cos(rot1)**2*(cos(delta1)-1)+1, sin(rot1)*cos(rot1)*(cos(delta1)-1), cos(rot1)*sin(delta1)],
        [sin(rot1)*cos(rot1)*(cos(delta1)-1), sin(rot1)**2*(cos(delta1)-1)+1, sin(rot1)*sin(delta1)],
        [-cos(rot1)*sin(delta1), -sin(rot1)*sin(delta1), cos(delta1)]
    ])
    t1 = Matrix([
        L1/delta1 * cos(rot1)*(1-cos(delta1)),
        L1/delta1 * sin(rot1)*(1-cos(delta1)),
        L1/delta1 * sin(delta1)
    ])
    
    offset1 = Matrix([0, 0, rigid_support])
    start_section2 = t1 + R1 * offset1
    
    # --- Section 2 ---
    R12 = Matrix([
        [cos(rot1 + pi + rot2)**2*(cos(delta2)-1)+1, sin(rot1 + pi + rot2)*cos(rot1 + pi + rot2)*(cos(delta2)-1), cos(rot1 + pi + rot2)*sin(delta2)],
        [sin(rot1 + pi + rot2)*cos(rot1 + pi + rot2)*(cos(delta2)-1), sin(rot1 + pi + rot2)**2*(cos(delta2)-1)+1, sin(rot1 + pi + rot2)*sin(delta2)],
        [-cos(rot1 + pi + rot2)*sin(delta2), -sin(rot1 + pi + rot2)*sin(delta2), cos(delta2)]
    ])
    t12 = Matrix([
        L2/delta2 * cos(rot1 + pi + rot2)*(1-cos(delta2)),
        L2/delta2 * sin(rot1 + pi + rot2)*(1-cos(delta2)),
        L2/delta2 * sin(delta2)
    ])
    # --- Correction: use R1 instead of R12 for the transformation ---
    t2 = start_section2 + R1 * t12
    R2 = R1 * R12
    
    offset2 = Matrix([0, 0, rigid_support])
    start_section3 = t2 + R2 * offset2
    
    # --- Section 3 ---
    R23 = Matrix([
        [cos(rot1 + pi + rot2 + pi + rot3)**2*(cos(delta3)-1)+1, sin(rot1 + pi + rot2 + pi + rot3)*cos(rot1 + pi + rot2 + pi + rot3)*(cos(delta3)-1), cos(rot1 + pi + rot2 + pi + rot3)*sin(delta3)],
        [sin(rot1 + pi + rot2 + pi + rot3)*cos(rot1 + pi + rot2 + pi + rot3)*(cos(delta3)-1), sin(rot1 + pi + rot2 + pi + rot3)**2*(cos(delta3)-1)+1, sin(rot1 + pi + rot2 + pi + rot3)*sin(delta3)],
        [-cos(rot1 + pi + rot2 + pi + rot3)*sin(delta3), -sin(rot1 + pi + rot2 + pi + rot3)*sin(delta3), cos(delta3)]
    ])
    t23 = Matrix([
        L3/delta3 * cos(rot1 + pi + rot2 + pi + rot3)*(1-cos(delta3)),
        L3/delta3 * sin(rot1 + pi + rot2 + pi + rot3)*(1-cos(delta3)),
        L3/delta3 * sin(delta3)
    ])
    t3 = start_section3 + R2 * t23
    R3 = R2 * R23

    # --- Gripper Offset ---
    t34 = Matrix([0, 0, offset_gripper])
    t3_gripper = t3 + R3 * t34

    # Stack key points: start_section2, start_section3, t3, and t3_gripper
    t_total = Matrix.vstack(t3, t3_gripper)
    
    # Return a callable function
    return lambdify((delta1, rot1, delta2, rot2, delta3, rot3), t_total, modules=['numpy'])

def convert_rad_to_motorposition(rot, gear_ratio):
    return int(rot * (4095 * gear_ratio) / (2 * np.pi))

def handle_key_event(event):
    global x_ee, y_ee, z_ee, x_grip, y_grip, z_grip

    # If event is None, return the current end_effector position
    if event is None:
        return [x_ee, y_ee, z_ee, x_grip, y_grip, z_grip]

    # Check if it's a key press event
    if event.event_type == keyboard.KEY_DOWN:
        if event.name == 'w':
            y_ee += step_size
            y_grip += step_size
        elif event.name == 's':
            y_ee -= step_size
            y_grip -= step_size
        elif event.name == 'q':
            x_ee += step_size
            x_grip += step_size
        elif event.name == 'a':
            x_ee -= step_size
            x_grip -= step_size
        elif event.name == 'e':
            z_ee += step_size
            z_grip += step_size
        elif event.name == 'd':
            z_ee -= step_size
            z_grip -= step_size
        
        elif event.name == 'u':
            delta_point = (x_grip - x_ee, y_grip - y_ee, z_grip - z_ee)
            rotated_x = delta_point[0]
            rotated_y = np.cos(rot_step) * delta_point[1] - np.sin(rot_step) * delta_point[2]
            rotated_z = np.sin(rot_step) * delta_point[1] + np.cos(rot_step) * delta_point[2]
            x_grip = rotated_x + x_ee
            y_grip = rotated_y + y_ee
            z_grip = rotated_z + z_ee
        elif event.name == 'j':
            delta_point = (x_grip - x_ee, y_grip - y_ee, z_grip - z_ee)
            rotated_x = delta_point[0]
            rotated_y = np.cos(-rot_step) * delta_point[1] - np.sin(-rot_step) * delta_point[2]
            rotated_z = np.sin(-rot_step) * delta_point[1] + np.cos(-rot_step) * delta_point[2]
            x_grip = rotated_x + x_ee
            y_grip = rotated_y + y_ee
            z_grip = rotated_z + z_ee
        
        elif event.name == 'i':
            delta_point = (x_grip - x_ee, y_grip - y_ee, z_grip - z_ee)
            rotated_x = np.cos(-rot_step) * delta_point[0] - np.sin(-rot_step) * delta_point[2]
            rotated_y = delta_point[1]
            rotated_z = np.sin(-rot_step) * delta_point[0] + np.cos(-rot_step) * delta_point[2]        
            x_grip = rotated_x + x_ee
            y_grip = rotated_y + y_ee
            z_grip = rotated_z + z_ee
        elif event.name == 'k':
            delta_point = (x_grip - x_ee, y_grip - y_ee, z_grip - z_ee) 
            rotated_x = np.cos(rot_step) * delta_point[0] - np.sin(rot_step) * delta_point[2]
            rotated_y = delta_point[1]
            rotated_z = np.sin(rot_step) * delta_point[0] + np.cos(rot_step) * delta_point[2]       
            x_grip = rotated_x + x_ee
            y_grip = rotated_y + y_ee
            z_grip = rotated_z + z_ee       

        elif event.name == 'o':
            delta_point = (x_grip - x_ee, y_grip - y_ee, z_grip - z_ee)        
            rotated_x = np.cos(rot_step) * delta_point[0] - np.sin(rot_step) * delta_point[1]
            rotated_y = np.sin(rot_step) * delta_point[0] + np.cos(rot_step) * delta_point[1]
            rotated_z = delta_point[2]
            x_grip = rotated_x + x_ee
            y_grip = rotated_y + y_ee
            z_grip = rotated_z + z_ee
        elif event.name == 'l':
            delta_point = (x_grip - x_ee, y_grip - y_ee, z_grip - z_ee)        
            rotated_x = np.cos(-rot_step) * delta_point[0] - np.sin(-rot_step) * delta_point[1]
            rotated_y = np.sin(-rot_step) * delta_point[0] + np.cos(-rot_step) * delta_point[1]
            rotated_z = delta_point[2]
            x_grip = rotated_x + x_ee
            y_grip = rotated_y + y_ee
            z_grip = rotated_z + z_ee
        
    xyz_goal = [x_ee, y_ee, z_ee, x_grip, y_grip, z_grip]
    return xyz_goal

def compute_rpy(xyz_goal):
    # Extract points
    p1 = xyz_goal[0:3]  # End effector
    p2 = xyz_goal[3:6]  # Gripper

    # Direction vector from p1 to p2
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)

    # Define a rotation that aligns the Z-axis to the direction vector
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(z_axis, direction)
    sin_angle = np.linalg.norm(rotation_vector)
    cos_angle = np.dot(z_axis, direction)
    
    if sin_angle < 1e-8:
        # Already aligned (or opposite)
        if cos_angle > 0:
            rotation = R.identity()
        else:
            # 180 degree rotation around any perpendicular axis
            rotation = R.from_rotvec(np.pi * np.array([1, 0, 0]))
    else:
        rotation_vector = rotation_vector / sin_angle
        angle = np.arctan2(sin_angle, cos_angle)
        rotation = R.from_rotvec(rotation_vector * angle)

    # Convert to Euler angles (roll, pitch, yaw)
    rpy = rotation.as_euler('xyz', degrees=False)
    return rpy

def compute_quaternion(xyz_goal):
    # Same logic as above
    p1 = xyz_goal[0:3]
    p2 = xyz_goal[3:6]
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)

    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(z_axis, direction)
    sin_angle = np.linalg.norm(rotation_vector)
    cos_angle = np.dot(z_axis, direction)
    
    if sin_angle < 1e-8:
        if cos_angle > 0:
            rotation = R.identity()
        else:
            rotation = R.from_rotvec(np.pi * np.array([1, 0, 0]))
    else:
        rotation_vector = rotation_vector / sin_angle
        angle = np.arctan2(sin_angle, cos_angle)
        rotation = R.from_rotvec(rotation_vector * angle)

    return rotation.as_quat()  # [x, y, z, w]

result_end_effector = end_effector_position ()
result_J = Jacobian()

#------------------------------------------------------------Initial condition------------------------------------------------------------
x_ee = 0
y_ee = 0
z_ee = L1 + rigid_support + L2 + rigid_support+ L3
x_grip = 0
y_grip = 0
z_grip = z_ee + offset_gripper
xyz_goal = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])

# delta_rot_goal = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
# delta_rot_old = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
delta_rot_goal = np.array([0.07643267, 0.02254561, 0.16332373, 0.00762068, 0.08694207, 0.02946593])
delta_rot_old = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])

section_limits = [1, 0.8, 0.8]
step_size = 1 #mm
rot_step = np.deg2rad(1) #degree
threshold = 1
K = np.diag([1, 1, 1, 1, 1, 1])
K_inv = np.linalg.inv(K)

print("\nNow you can type. For TRANSLATION: q/a for x axis, w/s for y axis, e/d for z-axis")
print("For ROTATION: u/j for x axis (roll), i/k for y axis (pitch), o/l for z-axis (yaw)")
print("type c to RETURN IN CALIBRATED POSITION")
print("and x to END THE LOOP")
# Register the callback for handling keyboard events
keyboard.on_press(handle_key_event)

#------------------------------------------------------------IK LOOP------------------------------------------------------------
exit_flag = False
while not exit_flag :
    # Check if 'e' key is pressed to exit the loop 
    if keyboard.is_pressed('x'):
        exit_flag = True
    
    if keyboard.is_pressed('c'):
        #Initial condition
        x_ee = 0
        y_ee = 0
        z_ee = L1 + rigid_support + L2 + rigid_support+ L3
        x_grip = 0
        y_grip = 0
        z_grip = z_ee + offset_gripper
        xyz_goal = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
        xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])

        delta_rot_goal = np.array([0.07643267, 0.02254561, 0.16332373, 0.00762068, 0.08694207, 0.02946593])
        delta_rot_old = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])
        for i, motor_id in enumerate(DYNAMIXEL_IDS):
            # Calculate the target position with the stored offset
            goal_position = int(offset[i])
            
            # Write the goal position (4 bytes for multi-turn mode)
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                portHandler, motor_id, ADDR_GOAL_POSITION, goal_position
            )
            if dxl_comm_result != COMM_SUCCESS:
                print("Motor ID {} communication error: {}".format(
                    motor_id, packetHandler.getTxRxResult(dxl_comm_result)))
            elif dxl_error != 0:
                print("Motor ID {} error: {}".format(
                    motor_id, packetHandler.getRxPacketError(dxl_error)))
        
        print("'c' pressed. Calibrated position reached")

    xyz_goal = np.array(handle_key_event(None))

    if np.any(xyz_goal != xyz_old):
        print("\n xyz_old:", xyz_old)
        print("xyz_goal:", xyz_goal)
        # print("K_inv", K_inv)

        # xyz_goal, delta_rot_goal = cartesian_configuration(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv)
        xyz_goal, delta_rot_goal = cartesian_configuration_noconv(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv)
        
        if (delta_rot_goal[4] * delta_rot_old[4]) < 0:
            if delta_rot_goal[5] > np.pi:
                delta_rot_goal[5] = delta_rot_goal[5] - np.pi
            else:
                delta_rot_goal[5] = delta_rot_goal[5] + np.pi
            delta_rot_goal[4] = - delta_rot_goal[4]
        if (delta_rot_goal[2] * delta_rot_old[2]) < 0:
            if delta_rot_goal[3] > np.pi:
                delta_rot_goal[3] = delta_rot_goal[3] - np.pi
                delta_rot_goal[5] = delta_rot_goal[5] + np.pi
            else:
                delta_rot_goal[3] = delta_rot_goal[3] + np.pi
                delta_rot_goal[5] = delta_rot_goal[5] - np.pi
            delta_rot_goal[2] = - delta_rot_goal[2]
        if norm(delta_rot_goal[1]) >= 2 * np.pi:
            delta_rot_goal[1] = delta_rot_goal[1] - 2 * np.pi * np.sign(delta_rot_goal[1])
        if norm(delta_rot_goal[3]) >= 2 * np.pi:
            delta_rot_goal[3] = delta_rot_goal[3] - 2 * np.pi * np.sign(delta_rot_goal[3])
        if norm(delta_rot_goal[4]) >= 2 * np.pi:
            delta_rot_goal[4] = delta_rot_goal[4] - 2 * np.pi * np.sign(delta_rot_goal[5])
        
        if np.isclose(delta_rot_goal[1], 2 * np.pi, atol=1):
            delta_rot_goal[1] = 0
        if np.isclose(delta_rot_goal[3], 2 * np.pi, atol=1):
            delta_rot_goal[3] = 0
        if np.isclose(delta_rot_goal[5], 2 * np.pi, atol=1):
            delta_rot_goal[5] = 0

        Rot = np.array([delta_rot_goal[1], delta_rot_goal[3], delta_rot_goal[5]])
        Delta = np.array([delta_rot_goal[0], norm(delta_rot_goal[2]), norm(delta_rot_goal[4])])
        print("xyz_current:", xyz_goal)
        print("roll pitch yaw [deg]", np.rad2deg(compute_rpy(xyz_goal)))
        print("quaternion", compute_quaternion(xyz_goal))

        x_ee, y_ee, z_ee, x_grip, y_grip, z_grip = xyz_goal
        print("delta_rot_current:", delta_rot_goal)

        xyz_old = xyz_goal.copy()
        delta_rot_old = delta_rot_goal.copy()

        print("\nTotal bending for each section [deg]:", np.rad2deg(Delta))
        print("Total rotation for each section [deg]:", np.rad2deg(Rot))    
        if Delta [0] > 0:
            motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(Delta[0], Delta[1], Delta[2])
        else:
            motor2_bend_section1, motor1_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(norm(Delta[0]), Delta[1], Delta[2])
        motor_rot_section1 = convert_rad_to_motorposition(Rot[0], 2)
        motor_rot_section2 = convert_rad_to_motorposition(Rot[1], 3)
        motor_rot_section3 = convert_rad_to_motorposition(Rot[2], 2)
        motor_gripper = 500
        motors_position = [motor_gripper, motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3]
        print("Input for the rotary motors [4095 encoder based]:", motor_rot_section1, motor_rot_section2, motor_rot_section3)
        print("Input for bending motors [4095 encoder based]:", motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3)
        print("Input for all motors in daisy chain [4095 encoder based]:", motors_position)

        for i, motor_id in enumerate(DYNAMIXEL_IDS):
            # Calculate the target position with the stored offset
            goal_position = int(motors_position[i] + offset[i])
            
            # Write the goal position (4 bytes for multi-turn mode)
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                portHandler, motor_id, ADDR_GOAL_POSITION, goal_position
            )
            if dxl_comm_result != COMM_SUCCESS:
                print("Motor ID {} communication error: {}".format(
                    motor_id, packetHandler.getTxRxResult(dxl_comm_result)))
            elif dxl_error != 0:
                print("Motor ID {} error: {}".format(
                    motor_id, packetHandler.getRxPacketError(dxl_error)))
        
        time.sleep(0.1)


# Close port
portHandler.closePort()
