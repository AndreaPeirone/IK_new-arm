import numpy as np
from sympy import symbols, cos, sin, Matrix, lambdify, pi
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
import pickle
from motor_mapping_class import MotorMapping
import keyboard
from dynamixel_sdk import *
import time
import pygame
from spacemouse_class import SpaceMouse
from gamepad_controller import Gamepad
from panther import Panther
from dynamixel_controller import DynamixelController, BaseModel

#------------------------------------------------------------MOTORS SETUP------------------------------------------------------------
MOTOR_IDs  = np.array([10, 11, 12, 13, 14, 15, 16, 17, 20])
motor_list = [ BaseModel(motor_id=int(mid)) for mid in MOTOR_IDs ]
ids_list   = MOTOR_IDs.tolist()

#Create and activate the controller
ctrl = DynamixelController(port_name="COM13", motor_list=motor_list, protocol=2.0, baudrate=57600, latency_time=1, reverse_direction=False)
ctrl.activate_controller()

#SET EXTENDED POSITION
ctrl.torque_off()
ctrl.set_operating_mode_all("current_based_position_control")
ctrl.torque_on()
print("All motors are now in extended‑position mode with torque enabled!")


#SET VELOCITY
set_velocity = 50
vel_list     = [set_velocity] * len(ids_list)
idx11        = ids_list.index(11)
vel_list[idx11] = int(set_velocity * 4/3) # Adjust velocity for motor 11 (base1) to be 4/3 of the set velocity
idx13        = ids_list.index(12)
vel_list[idx13] = int(set_velocity * 2)
idx13        = ids_list.index(12)
vel_list[idx13] = int(set_velocity * 2)

tx_result = ctrl.set_profile_velocity(vel_list)
if tx_result != 0:
    print(f"Error sending profile velocities: {tx_result}")
else:
    print("Profile velocities set on all motors!")

#------------------------------------------------------------STORE CALIBRATION------------------------------------------------------------
# raw_positions, _, _, _ = ctrl.read_info(fast_read=False)
# offset = raw_positions.tolist()
# print(f"Initial positions (multi‑turn raw values): {offset}")
# Load offset from file
with open("offset.pkl", "rb") as f:
    offset = pickle.load(f)
print(f"Initial positions (multi‑turn raw values): {offset}")
time.sleep(1)

#Initial condition
goal_position = (np.array(offset)).astype(int).tolist()
tx_result = ctrl.set_goal_position(goal_position)
if tx_result != COMM_SUCCESS:
    print(f"GroupSyncWrite COMM error: {ctrl._DynamixelController__packet_handler.getTxRxResult(tx_result)}")
else:
    print("All goal positions sent successfully!")


# ------------------------------------------------------------LOAD MAPPING------------------------------------------------------------
# Load the combined mapping
with open('combined_motor_mapping.pkl', 'rb') as f:
    motor_mapping = pickle.load(f)

#------------------------------------------------------------FUNCTIONS------------------------------------------------------------
#Geometric parameters
L1 = 228 #height 1st section [mm]
L2 = 228 #height 2nd section [mm]
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

    variables_vector = Matrix([delta1, rot1, delta2, rot2, delta3, rot3])
    J = t3.jacobian(variables_vector)
    return lambdify((delta1, rot1, delta2, rot2, delta3, rot3), J, modules=['numpy'])

def Jacobian_orientation():
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
    # timer = time.time()
    e = xyz_goal - xyz_old
    # print("error", e)
    if moving_keeping_orientation:
        J = result_J_orientation(delta_rot_goal[0], delta_rot_goal[1], 
                 delta_rot_goal[2], delta_rot_goal[3], 
                 delta_rot_goal[4], delta_rot_goal[5])
    else:
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
    
    # print("delta_rot_current before boundaries", delta_rot_goal)
    delta_rot_goal, K_inv = check_boundaries(delta_rot_goal, K_inv)
    if moving_keeping_orientation:
        end_effector = configuration_to_cartesian_orientation(delta_rot_goal[0], delta_rot_goal[1],
                                                delta_rot_goal[2], delta_rot_goal[3], 
                                                delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
    else:
        end_effector = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1],
                                                delta_rot_goal[2], delta_rot_goal[3], 
                                                delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
    e = xyz_goal - end_effector
    # print("xyz_goal", xyz_goal)
    # print("end_effector evaluation", end_effector)
    # print("error", e)
    # print("norm error", norm(e))
    # print("stiffness matrix", np.diag(K_inv))
    count = 0
    while np.linalg.norm(e) > threshold:
        count += 1
        if moving_keeping_orientation:
            J = result_J_orientation(delta_rot_goal[0], delta_rot_goal[1], 
                    delta_rot_goal[2], delta_rot_goal[3], 
                    delta_rot_goal[4], delta_rot_goal[5])
        else:
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

        # print("delta_rot_current before boundaries", delta_rot_goal)
        delta_rot_goal, K_inv = check_boundaries(delta_rot_goal, K_inv)

        if moving_keeping_orientation:
            end_effector = configuration_to_cartesian_orientation(delta_rot_goal[0], delta_rot_goal[1],
                                                    delta_rot_goal[2], delta_rot_goal[3], 
                                                    delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
        else:
            end_effector = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1],
                                                    delta_rot_goal[2], delta_rot_goal[3], 
                                                    delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
        # print("xyz_goal", xyz_goal)
        # print("end_effector evaluation", end_effector)
        e = xyz_goal - end_effector
        # print("error", e)
        # print("norm error", norm(e))
        # print("stiffness matrix", np.diag(K_inv))

        if count == 10:
            # print("\nSATURATION OF ITERATIONS REACHED:", count)
            delta_rot_goal = delta_rot_old.copy()
            end_effector = xyz_old.copy()
            break
    # print(round(time.time() - timer, 4))
    return end_effector, delta_rot_goal

def check_boundaries(delta_rot_goal, K_inv_original):
    K_inv = K_inv_original.copy()
    eps = 1e-2

    if norm(delta_rot_goal[0]) > section_limits[0]:
        print("----------------------------------Max bending on 1st section reached----------------------------------")
        delta_rot_goal[0] = np.sign(delta_rot_goal[0]) * section_limits[0]
        K_inv [0,0] = K_inv[0,0] / 1e6

    if norm(delta_rot_goal[2]) > section_limits[1]:
        print("----------------------------------Max bending on 2nd section reached----------------------------------")
        delta_rot_goal[2] = np.sign(delta_rot_goal[2]) * section_limits[1]
        K_inv [2,2] = K_inv[2,2] / 1e6
    
    if norm(delta_rot_goal[4]) > section_limits[2]:
        print("----------------------------------Max bending on 3rd section reached----------------------------------")
        delta_rot_goal[4] = np.sign(delta_rot_goal[4]) * section_limits[2]
        K_inv [4,4] = K_inv[4,4] / 1e6 

    if np.isnan(delta_rot_goal[0]):
        delta_rot_goal[0] = eps
    if np.isnan(delta_rot_goal[2]):
        delta_rot_goal[2] = eps
    if np.isnan(delta_rot_goal[4]):
        delta_rot_goal[4] = eps

    if delta_rot_goal[1] < eps:
        K_inv [1,1] = K_inv[1,1] / 1e10
    if delta_rot_goal[3] < eps:
        K_inv [3,3] = K_inv[3,3] / 1e10
    if delta_rot_goal[5] < eps:
        K_inv [5,5] = K_inv[5,5] / 1e10

    # print("delta_rot_current after boundaries", delta_rot_goal)
    return delta_rot_goal, K_inv

def adjust_configuration(delta_rot_goal, delta_rot_old):
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
    # if norm(delta_rot_goal[1]) >= 2 * np.pi:
    #     delta_rot_goal[1] = delta_rot_goal[1] - 2 * np.pi * np.sign(delta_rot_goal[1])
    # if norm(delta_rot_goal[3]) >= 2 * np.pi:
    #     delta_rot_goal[3] = delta_rot_goal[3] - 2 * np.pi * np.sign(delta_rot_goal[3])
    # if norm(delta_rot_goal[4]) >= 2 * np.pi:
    #     delta_rot_goal[4] = delta_rot_goal[4] - 2 * np.pi * np.sign(delta_rot_goal[5])
    
    # if np.isclose(delta_rot_goal[1], 2 * np.pi, atol=1):
    #     delta_rot_goal[1] = 0
    # if np.isclose(delta_rot_goal[3], 2 * np.pi, atol=1):
    #     delta_rot_goal[3] = 0
    # if np.isclose(delta_rot_goal[5], 2 * np.pi, atol=1):
    #     delta_rot_goal[5] = 0

    Rot = np.array([delta_rot_goal[1], delta_rot_goal[3], delta_rot_goal[5]])
    Delta = np.array([delta_rot_goal[0], norm(delta_rot_goal[2]), norm(delta_rot_goal[4])])

    return Delta, Rot

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
    
    # Return a callable function
    return lambdify((delta1, rot1, delta2, rot2, delta3, rot3), t3, modules=['numpy'])

def end_effector_position_orientation():
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

def convert_rad_to_motorposition(encoder, gear_ratio):
    return int(encoder * (4095 * gear_ratio) / (2 * np.pi))

def convert_motorposition_to_rad(encoder, gear_ratio):
    return (2 * np.pi * encoder) / (4095 * gear_ratio)

def update_goals_with_spacemouse_pitch(xyz_goal, pitch_angle, y_hat, gain_translation, gain_pitch, sm_data):
    dx, dy, dz = sm_data[:3]
    dpitch = y_hat
    pitch_angle += dpitch * gain_pitch 
    if abs(pitch_angle) > np.deg2rad(60):
        pitch_angle = np.sign(pitch_angle) * np.deg2rad(60)
    Dx = dx * np.cos(pitch_angle) - dz * np.sin(pitch_angle)
    Dz = dx * np.sin(pitch_angle) + dz * np.cos(pitch_angle) 

    xyz_goal[0] += Dx * gain_translation 
    xyz_goal[1] += dy * gain_translation
    xyz_goal[2] += Dz * gain_translation

    return xyz_goal, pitch_angle

def update_goals_with_spacemouse_pitch_orientation(xyz_goal, pitch_angle, y_hat, gain_translation, gain_rotation, gain_pitch, sm_data):
    dx, dy, dz, droll, dpitch, dyaw = sm_data[:6]
    dpitch_base = y_hat
    pitch_angle += dpitch_base * gain_pitch 
    if abs(pitch_angle) > np.deg2rad(60):
        pitch_angle = np.sign(pitch_angle) * np.deg2rad(60)
    Dx = dx * np.cos(pitch_angle) - dz * np.sin(pitch_angle)
    Dz = dx * np.sin(pitch_angle) + dz * np.cos(pitch_angle) 

    xyz_goal[0] += Dx * gain_translation 
    xyz_goal[1] += dy * gain_translation
    xyz_goal[2] += Dz * gain_translation
    xyz_goal[3] += Dx * gain_translation
    xyz_goal[4] += dy * gain_translation
    xyz_goal[5] += Dz * gain_translation
    xyz_goal[3:6] = update_gripper_position(xyz_goal[:3], xyz_goal[3:6], droll * gain_rotation, dpitch * gain_rotation, dyaw * gain_rotation)

    return xyz_goal, pitch_angle

def translation_keeping_orientation(xyz_goal, pitch_angle, y_hat, gain_translation, gain_rotation, gain_pitch, axes):
    dz = - axes[3]
    dx = - axes[2]
    dy, droll, dpitch, dyaw = [0] * 4
    dpitch_base = y_hat
    pitch_angle += dpitch_base * gain_pitch 
    if abs(pitch_angle) > np.deg2rad(60):
        pitch_angle = np.sign(pitch_angle) * np.deg2rad(60)
    Dx = dx * np.cos(pitch_angle) - dz * np.sin(pitch_angle)
    Dz = dx * np.sin(pitch_angle) + dz * np.cos(pitch_angle) 

    xyz_goal[0] += Dx * gain_translation 
    xyz_goal[1] += dy * gain_translation
    xyz_goal[2] += Dz * gain_translation
    xyz_goal[3] += Dx * gain_translation
    xyz_goal[4] += dy * gain_translation
    xyz_goal[5] += Dz * gain_translation
    # xyz_goal[3:6] = update_gripper_position(xyz_goal[:3], xyz_goal[3:6], droll * gain_rotation, dpitch * gain_rotation, dyaw * gain_rotation)

    return xyz_goal, pitch_angle

def update_gripper_position(P, Q, delta_theta_x_degrees, delta_theta_y_degrees, delta_theta_z_degrees):
    Q_x, Q_y, Q_z = Q
    delta_theta_x = np.deg2rad(delta_theta_x_degrees)/2
    delta_theta_y = np.deg2rad(delta_theta_y_degrees)/2
    delta_theta_z = np.deg2rad(delta_theta_z_degrees)/2

    Rx_increment = np.array([[1, 0, 0],
                             [0, np.cos(delta_theta_x), -np.sin(delta_theta_x)],
                             [0, np.sin(delta_theta_x), np.cos(delta_theta_x)]])
    
    Ry_increment = np.array([[np.cos(delta_theta_y), 0, np.sin(delta_theta_y)],
                             [0, 1, 0],
                             [-np.sin(delta_theta_y), 0, np.cos(delta_theta_y)]])

    Rz_increment = np.array([[np.cos(delta_theta_z), -np.sin(delta_theta_z), 0],
                             [np.sin(delta_theta_z), np.cos(delta_theta_z), 0],
                             [0, 0, 1]])

    R_increment = np.dot(Rz_increment, np.dot(Ry_increment, Rx_increment))
    direction = np.array([Q_x - P[0], Q_y - P[1], Q_z - P[2]])
    rotated_direction = np.dot(R_increment, direction)
    updated_Q = P + rotated_direction
    return updated_Q

def from_pose_get_cartesian(delta_rot_goal, pitch_angle):
    Rot = np.array([delta_rot_goal[1], delta_rot_goal[3], delta_rot_goal[5]])
    Delta = np.array([delta_rot_goal[0], delta_rot_goal[2], delta_rot_goal[4]])
    xyz_goal = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1], 
                            delta_rot_goal[2], delta_rot_goal[3], 
                            delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
    # print("\nTotal bending for each section [deg]:", np.rad2deg(Delta))
    # print("Total rotation for each section [deg]:", np.rad2deg(Rot))
    # print("Pose reached [x_ee, y_ee, z_ee]:", xyz_goal)
    if Delta [0] > 0:
        motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(Delta[0], norm(Delta[1]), norm(Delta[2]))
    else:
        motor2_bend_section1, motor1_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(norm(Delta[0]), norm(Delta[1]), norm(Delta[2]))
    if Delta[2] < 0:
        motor_bend_section3 = - int(motor_bend_section3 * 10 / 9)
    if Delta[1] < 0:
        motor_bend_section2 = - int(motor_bend_section2 * 10 / 9)
    motor_rot_section1 = convert_rad_to_motorposition(Rot[0], 4)
    motor_rot_section2 = convert_rad_to_motorposition(Rot[1], 3)
    motor_rot_section3 = convert_rad_to_motorposition(Rot[2], 3)
    motor_pitch = convert_rad_to_motorposition(pitch_angle, ((200/25) * (22/15))*2 )
    motors_position = [motor_gripper, motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3, motor_pitch]
    # print("Input for the rotary motors [4095 encoder based]:", motor_rot_section1, motor_rot_section2, motor_rot_section3)
    # print("Input for bending motors [4095 encoder based]:", motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3)
    # print("Input for all motors in daisy chain [4095 encoder based]:", motors_position)
    goal_position = (np.array(motors_position) + np.array(offset)).astype(int).tolist()
    tx_result = ctrl.set_goal_position(goal_position)
    # if tx_result != COMM_SUCCESS:
    #     print(f"GroupSyncWrite COMM error: {ctrl._DynamixelController__packet_handler.getTxRxResult(tx_result)}")
    # else:
    #     print("All goal positions sent successfully!")
    return xyz_goal, motors_position

#------------------------------------------------------------SET ORIENTATION FLAG------------------------------------------------------------
orientation_flag = False

#------------------------------------------------------------Initial condition------------------------------------------------------------
x_ee = 0
y_ee = 0
z_ee = L1 + rigid_support + L2 + rigid_support+ L3
x_grip = 0
y_grip = 0
z_grip = z_ee + offset_gripper
gain_translation = 5
gain_rotation = 5
gain_gripper = 10
gain_pitch = np.deg2rad(.15)
pitch_angle = 0
pitch_angle_old = 0
if orientation_flag:
    configuration_to_cartesian = end_effector_position_orientation()
    result_J = Jacobian_orientation()
    xyz_goal = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
    xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
    threshold = 1
else:
    configuration_to_cartesian = end_effector_position()
    result_J = Jacobian()
    configuration_to_cartesian_orientation = end_effector_position_orientation()
    result_J_orientation = Jacobian_orientation()
    xyz_goal = np.array([x_ee, y_ee, z_ee])
    xyz_old = np.array([x_ee, y_ee, z_ee])
    threshold = 1

delta_rot_goal = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
delta_rot_old = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])

section_limits = [1.4, 1.2, 1.2]
K = np.diag([1, 1, 1, 1, 1, 1])
K_inv = np.linalg.inv(K)

motor_gripper, motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3, motor_pitch = [0] * 9
motors_position = [motor_gripper, motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3, motor_pitch]
_, _, _, max_bend_section3 = motor_mapping(0, 0, norm(section_limits[2]))
min_val = offset[7] - abs(max_bend_section3)
max_val = offset[7] + abs(max_bend_section3)
print("\nNow you can use the ps controller")
print("type c to RETURN IN CALIBRATED POSITION")
print("type v to MOVE TO DESIRED POSE")
print("and x to END THE LOOP")

# Register the callback for handling spacemouse events
sm = SpaceMouse()
sm.start()

# Initialize pygame and the joystick module
DRIFT_THRESHOLD = 0.05  # Treat values within this range as zero
# Initialize
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Initialized controller: {joystick.get_name()}")

print("Calibrating... Do not touch the controller.")
time.sleep(1)  # Give user time to release the controller

pygame.event.pump()
axis_count = joystick.get_numaxes()
offset_ps = [joystick.get_axis(i) for i in range(axis_count)]

print(f"offset_ps (calibrated rest position): {offset_ps}")
print("Starting input loop...\n")
#------------------------------------------------------------PANTHER------------------------------------------------------------
# MAX_LINEAR_VEL = .18 #18
# MAX_ANGULAR_VEL = 0.23 #0.23
# MAX_LINEAR_ACC = 0.01
# MAX_ANGULAR_ACC = 0.02

MAX_LINEAR_VEL = 0.74
MAX_ANGULAR_VEL = 0.63
MAX_LINEAR_ACC = 0.015
MAX_ANGULAR_ACC = 0.02
JOY_CUTOFF = 0.2

myPanther = Panther(max_linear_vel=MAX_LINEAR_VEL,
                    max_angular_vel=MAX_ANGULAR_VEL,
                    max_linear_accel=MAX_LINEAR_ACC,
                    max_angular_accel=MAX_ANGULAR_ACC,
                    ip="10.15.20.2", port = 9090)

goal_current = [1193] * 9
goal_current[0] = 100
goal_current[4] = 900

ctrl.set_goal_current(goal_current)

#------------------------------------------------------------IK LOOP------------------------------------------------------------
exit_flag = False
no_input = True
command_point = True
first_stop = True
moving_keeping_orientation = False
last_sec_bend = 0
gripper_position = 0
gripper_min = 0
gripper_max = 2200

try:
    while not exit_flag :
        # Check if 'e' key is pressed to exit the loop c
        if keyboard.is_pressed('x'):
            exit_flag = True

        #############################################
        # USER INPUTS          
        #############################################
        pygame.event.pump()
        raw_axes = [joystick.get_axis(i) for i in range(axis_count)]
        axes = []
        for i in range(axis_count):
            val = raw_axes[i] - offset_ps[i]
            if abs(val) < DRIFT_THRESHOLD:
                val = 0.0
            axes.append(round(val, 3))  # Round for cleaner display
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        x_hat, y_hat = joystick.get_hat(0)

        sm_data = sm.get_array() #get input from the spacemouse to move the arm
        
        joystick_zero = np.allclose(sm_data[:6], 0.0, atol=1e-4)
        axes_zero = np.allclose(axes[2:4], 0.0, atol=1e-2)
        # gripper_zero     = (sm_data[6] == 0 and sm_data[7] == 0)
        pitch_zero = (y_hat == 0)

        #Bend manually the last section if "LB" or "RB" on the controller are pressed
        if sm_data[6] == 1:
            last_sec_bend = 1
            command_point = False
        elif sm_data[7] == 1:
            last_sec_bend = -1
            command_point = False
        else:
            last_sec_bend = 0
        

        # PANTHER
        fwd = axes[1] * -1
        trn = axes[0] * -1
        if abs(fwd) < JOY_CUTOFF:
            fwd = 0
        if abs(trn) < JOY_CUTOFF:
            trn = 0
        fwd_vel = MAX_LINEAR_VEL * fwd
        trn_vel = MAX_ANGULAR_VEL * trn

        myPanther.command_robot(fwd_vel, trn_vel) #get input from the controller to move the panther

        #############################################
        # GRIPPER
        #############################################
        gripper_flag = False
        if buttons[5] != 0:
            motor_gripper += gain_gripper
        if buttons[4] != 0:
            motor_gripper -= gain_gripper

        motor_gripper = max(min(gripper_max, motor_gripper), gripper_min)
        
        #############################################
        # HANDLING NO INPUT
        #############################################
        if joystick_zero and pitch_zero and axes_zero:
            no_input = True
        else:
            no_input = False
            command_point = False


        #############################################
        # COMMAND POINTS
        #############################################
        is_command_point_pressed = False

        if buttons[0] == 1: #Go to desierd pose if "A" on the controller is pressed, CALIBRATION
            delta_rot_goal = np.array([np.deg2rad(0.001), np.deg2rad(0.001), np.deg2rad(0.001), np.deg2rad(0.001), np.deg2rad(0.001), np.deg2rad(0.001)])
            pitch_angle = np.deg2rad(0)
            is_command_point_pressed = True

        if buttons[1] == 1: #Go to desierd pose if "B" on the controller is pressed, RIGHT
            delta_rot_goal = np.array([np.deg2rad(30), np.deg2rad(-90), np.deg2rad(45), np.deg2rad(180), np.deg2rad(45), np.deg2rad(-180)])
            pitch_angle = np.deg2rad(0)
            is_command_point_pressed = True
        
        if buttons[2] == 1: #Go to desierd pose if "X" on the controller is pressed, LEFT
            delta_rot_goal = np.array([np.deg2rad(30), np.deg2rad(90), np.deg2rad(45), np.deg2rad(180), np.deg2rad(45), np.deg2rad(-180)])
            pitch_angle = np.deg2rad(0)
            is_command_point_pressed = True

        if buttons[3] == 1: #Go to desierd pose if "Y" on the controller is pressed, FRONT
            delta_rot_goal = np.array([np.deg2rad(30), np.deg2rad(0.01), np.deg2rad(30), np.deg2rad(180), np.deg2rad(30), np.deg2rad(-180)])
            pitch_angle = np.deg2rad(0)
            is_command_point_pressed = True

        if buttons[7] == 1: #Go to desierd pose if "menu" on the controller is pressed, back to the box
            delta_rot_goal = np.array([-section_limits[0], np.deg2rad(0), section_limits[1], np.deg2rad(10), section_limits[2], np.deg2rad(-180)])
            pitch_angle = np.deg2rad(50)
            is_command_point_pressed = True

        if abs(axes[3]) < 0.07:
            axes[3] = 0
        if abs(axes[2]) < 0.07:
            axes[2] = 0

        print(axes[3], axes[2])
        if axes[3] != 0 or axes[2] != 0: #Move in the plane for raspberry
            moving_keeping_orientation = True
            xyz_orientation = configuration_to_cartesian_orientation(delta_rot_goal[0], delta_rot_goal[1], 
                    delta_rot_goal[2], delta_rot_goal[3], 
                    delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
            xyz_orientation_old = xyz_orientation.copy()
            xyz_orientation, pitch_angle = translation_keeping_orientation(xyz_orientation, pitch_angle, y_hat, gain_translation, gain_rotation, gain_pitch, axes)
            xyz_orientation, delta_rot_goal = cartesian_configuration(xyz_orientation, xyz_orientation_old, delta_rot_goal, delta_rot_old, K_inv)
            # is_command_point_pressed = True
            command_point = True

        if is_command_point_pressed:
            xyz_goal, motors_position = from_pose_get_cartesian(delta_rot_goal, pitch_angle)
            # print(motors_position)
            xyz_old = xyz_goal.copy()
            delta_rot_old = delta_rot_goal.copy()
            command_point = True

        #############################################
        # INVERSE KINEMATICS
        #############################################
        if moving_keeping_orientation:
            moving_keeping_orientation = False
            if not orientation_flag:
                xyz_goal = xyz_orientation[:3]
        else:
            if orientation_flag:
                xyz_goal, pitch_angle = update_goals_with_spacemouse_pitch_orientation(xyz_goal, pitch_angle, y_hat, gain_translation, gain_rotation, gain_pitch, sm_data)
            else:
                xyz_goal, pitch_angle = update_goals_with_spacemouse_pitch(xyz_goal, pitch_angle, y_hat, gain_translation, gain_pitch, sm_data)

            xyz_goal, delta_rot_goal = cartesian_configuration(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv)
            # Delta, Rot = adjust_configuration(delta_rot_goal, delta_rot_old)
        
        Rot = np.array([delta_rot_goal[1], delta_rot_goal[3], delta_rot_goal[5]])
        Delta = np.array([delta_rot_goal[0], delta_rot_goal[2], delta_rot_goal[4]])

        # print("xyz_current:", xyz_goal)
        if orientation_flag:
            x_ee, y_ee, z_ee, x_grip, y_grip, z_grip = xyz_goal
        else:
            x_ee, y_ee, z_ee = xyz_goal
        # print("delta_rot_current:", delta_rot_goal)

        xyz_old = xyz_goal.copy()
        delta_rot_old = delta_rot_goal.copy()
        pitch_angle_old = pitch_angle.copy()

        # print("\nTotal bending for each section [deg]:", np.rad2deg(Delta))
        # print("Total rotation for each section [deg]:", np.rad2deg(Rot))
        # print("Total pitch for the base [deg]:", np.rad2deg(pitch_angle)) 
        
        if Delta [0] > 0:
            motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(Delta[0], norm(Delta[1]), norm(Delta[2]))
        else:
            motor2_bend_section1, motor1_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(norm(Delta[0]), norm(Delta[1]), norm(Delta[2]))
        if Delta[2] < 0:
            motor_bend_section3 = - int(motor_bend_section3 * 10 / 9)
        if Delta[1] < 0:
            motor_bend_section2 = - int(motor_bend_section2 * 10 / 9)
        motor_rot_section1 = convert_rad_to_motorposition(Rot[0], 4)
        motor_rot_section2 = convert_rad_to_motorposition(Rot[1], 3)
        motor_rot_section3 = convert_rad_to_motorposition(Rot[2], 3)
        motor_pitch = convert_rad_to_motorposition(pitch_angle, ((200/25) * (22/15))*2 )
        motors_position = [motor_gripper, motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3, motor_pitch]
        # print("Input for the rotary motors [4095 encoder based]:", motor_rot_section1, motor_rot_section2, motor_rot_section3)
        # print("Input for bending motors [4095 encoder based]:", motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3)
        # print("Input for pitch motors [4095 encoder based]:", motor_pitch)
        # print("Input for all motors in daisy chain [4095 encoder based]:", motors_position)

        goal_position = (np.array(motors_position) + np.array(offset)).astype(int).tolist()

        # print("No input", no_input, " cmd point", command_point)
        if (no_input == True) and (command_point == False):
            # print("No input")
            if first_stop:
                raw_positions , _, _, _ = ctrl.read_info(fast_read=False)

            first_stop = False

            goal_position = raw_positions.copy()
            goal_position[7] += (last_sec_bend * 20)
            if goal_position[7] < min_val or goal_position[7] > max_val:
                goal_position[7] = max(min(goal_position[7], max_val), min_val)

            raw_positions = goal_position.copy()
            motors_position = goal_position - offset

            # print(motors_position)
            delta_rot_goal[1] = convert_motorposition_to_rad(motors_position[1], 4)
            delta_rot_goal[3] = convert_motorposition_to_rad(motors_position[4], 3)
            delta_rot_goal[5] = convert_motorposition_to_rad(motors_position[6], 3)
            if motors_position[5] > 0:
                motors_position[5] *= (9 / 10)
            if motors_position[7] > 0:
                motors_position[7] *= (9 / 10) 
            if motors_position[2] < motors_position[3]:
                delta_1, delta_1_ext, delta_2, delta_3 = motor_mapping.inverse(motors_position[2], motors_position[3], -norm(motors_position[5]), -norm(motors_position[7]))
            else:
                delta_1, delta_1_ext, delta_2, delta_3 = motor_mapping.inverse(motors_position[3], motors_position[2], -norm(motors_position[5]), -norm(motors_position[7]))
                delta_1 = - delta_1
                delta_1_ext = - delta_1_ext
            if motors_position[5] > 0:
                delta_2 = - delta_2
            if motors_position[7] > 0:
                delta_3 = - delta_3
            delta_rot_goal[0] = delta_1
            delta_rot_goal[2] = delta_2
            delta_rot_goal[4] = delta_3
            # print("CONFIGURATION EVALUATED", delta_rot_goal)
            delta_rot_old = delta_rot_goal.copy()
            xyz_goal = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1], 
                    delta_rot_goal[2], delta_rot_goal[3], 
                    delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
            # print("CARTESIAN COORDINATES", xyz_goal)
            xyz_old = xyz_goal.copy()

            pitch_angle = convert_motorposition_to_rad(motors_position[8], ((200/25) * (22/15)*2))
            pitch_angle_old = pitch_angle.copy()

        else:
            first_stop = True

        print(motor_gripper)
        goal_position[0] = motor_gripper + offset[0]
        tx_result = ctrl.set_goal_position(goal_position)

finally:
    pygame.quit()
    sm.stop()

# Close port
ctrl._port_handler.closePort()
print("Port closed.")