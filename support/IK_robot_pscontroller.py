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
from dynamixel_controller import DynamixelController, BaseModel

#------------------------------------------------------------MOTORS SETUP------------------------------------------------------------
MOTOR_IDs  = np.array([10, 11, 12, 13, 14, 15, 16, 17, 20])
motor_list = [ BaseModel(motor_id=int(mid)) for mid in MOTOR_IDs ]
ids_list   = MOTOR_IDs.tolist()

#Create and activate the controller
ctrl = DynamixelController(port_name="COM13", motor_list=motor_list, protocol=2.0, baudrate=57600, latency_time=1, reverse_direction=False)
ctrl.activate_controller()

#SET EXTENDED POSITION
ctrl.set_operating_mode_all("extended_position_control")
time.sleep(0.1)
ctrl.torque_on()
print("All motors are now in extended‑position mode with torque enabled!")

#SET VELOCITY
set_velocity = 70
vel_list     = [set_velocity] * len(ids_list)
idx14        = ids_list.index(14)
vel_list[idx14] = int(set_velocity * 1)

tx_result = ctrl.set_profile_velocity(vel_list)
if tx_result != 0:
    print(f"Error sending profile velocities: {tx_result}")
else:
    print("Profile velocities set on all motors!")

#------------------------------------------------------------STORE CALIBRATION------------------------------------------------------------
raw_positions, _, _, _ = ctrl.read_info(fast_read=False)
offset = raw_positions.tolist()
print(f"Initial positions (multi‑turn raw values): {offset}")


# ------------------------------------------------------------LOAD MAPPING------------------------------------------------------------
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
    # timer = time.time()
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

    end_effector = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1], 
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

        end_effector = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1], 
                                           delta_rot_goal[2], delta_rot_goal[3], 
                                           delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
        print("xyz_goal", xyz_goal)
        print("end_effector evaluation", end_effector)
        e = xyz_goal - end_effector
        print("error", e)
        print("norm error", norm(e))
        print("stiffness matrix", np.diag(K_inv))

        # if count == 5:
        #     end_effector = np.array([0, 0, L1 + rigid_support + L2 + rigid_support + L3, 0, 0, L1 + rigid_support + L2 + rigid_support + L3 + offset_gripper])
        #     delta_rot_goal = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])
        #     K_inv = np.linalg.inv(np.diag([1, 1, 1, 1, 1, 1]))
        #     e = xyz_goal - end_effector
        #     print("\nIK FROM ZERO POSITION")
        #     print("error", e)
        #     print("norm error", norm(e))
        #     print("stiffness matrix", np.diag(K_inv))

        if count == 10:
            print("\nSATURATION OF ITERATIONS REACHED:", count)
            delta_rot_goal = delta_rot_old.copy()
            end_effector = xyz_old.copy()
            break
    # print(round(time.time() - timer, 4))
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

    end_effector = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1], 
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

def update_goals_with_pscontroller(xyz_goal, axes, buttons, gain_translation, gain_rotation):
    dy, dx, droll, dpitch, dz1, dz2 = axes
    dyaw1, dyaw2 = buttons[9:11]

    xyz_goal[0] += - dx * gain_translation
    xyz_goal[1] += - dy * gain_translation
    xyz_goal[2] += (dz1 - dz2) / 2 * gain_translation
    xyz_goal[3] += - dx * gain_translation
    xyz_goal[4] += - dy * gain_translation
    xyz_goal[5] += (dz1 - dz2) / 2 * gain_translation
    xyz_goal[3:6] = update_gripper_position(xyz_goal[:3], xyz_goal[3:6], droll * gain_rotation, - dpitch * gain_rotation, (dyaw1 - dyaw2) * gain_rotation)
    
    return xyz_goal

def update_goals_with_pscontroller_pitch(xyz_goal, pitch_angle, axes, buttons, gain_translation, gain_rotation, gain_pitch):
    dy, dx, droll, dpitch, dz1, dz2 = axes
    dyaw1, dyaw2 = buttons[9:11]
    dpitch1, dpitch2 = buttons[11:13]
    pitch_angle += (dpitch1 - dpitch2) * gain_pitch
    Dx = dx * np.cos(pitch_angle) + (dz1 - dz2)/2 * np.sin(pitch_angle)
    Dz = - dx * np.sin(pitch_angle) + (dz1 - dz2) / 2 * np.cos(pitch_angle)       

    xyz_goal[0] += - Dx * gain_translation 
    xyz_goal[1] += - dy * gain_translation
    xyz_goal[2] +=  Dz * gain_translation
    xyz_goal[3] += - Dx * gain_translation
    xyz_goal[4] += - dy * gain_translation
    xyz_goal[5] += Dz * gain_translation
    xyz_goal[3:6] = update_gripper_position(xyz_goal[:3], xyz_goal[3:6], droll * gain_rotation, - dpitch * gain_rotation, (dyaw1 - dyaw2) * gain_rotation)

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

configuration_to_cartesian = end_effector_position ()
result_J = Jacobian()

#------------------------------------------------------------Initial condition------------------------------------------------------------
x_ee = 0
y_ee = 0
z_ee = L1 + rigid_support + L2 + rigid_support+ L3
x_grip = 0
y_grip = 0
z_grip = z_ee + offset_gripper
gain_translation = .5
gain_rotation = .5
gain_gripper = 50
motor_gripper = 0
gain_pitch = np.deg2rad(.075)
pitch_angle = 0
pitch_angle_old = 0
xyz_goal = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])

delta_rot_goal = np.array([0.07643267, 0.02254561, 0.16332373, 0.00762068, 0.08694207, 0.02946593])
delta_rot_old = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])

section_limits = [1.15, 0.95, 0.95]
step_size = 1 #mm
rot_step = np.deg2rad(1) #degree
threshold = 10
K = np.diag([1, 1, 1, 1, 1, 1])
K_inv = np.linalg.inv(K)

print("\nNow you can use the ps controller")
print("type c to RETURN IN CALIBRATED POSITION")
print("type v to MOVE TO DESIRED POSE")
print("and x to END THE LOOP")

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
time.sleep(2)  # Give user time to release the controller

pygame.event.pump()
axis_count = joystick.get_numaxes()
offset_ps = [joystick.get_axis(i) for i in range(axis_count)]

print(f"offset_ps (calibrated rest position): {offset_ps}")
print("Starting input loop...\n")

#------------------------------------------------------------IK LOOP------------------------------------------------------------
exit_flag = False
last_ps_nonzero = False

try:
    while not exit_flag :
        # Check if 'e' key is pressed to exit the loop c
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
            motor_gripper = 0
            pitch_angle = 0
            pitch_angle_old = 0
            xyz_goal = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
            xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])

            delta_rot_goal = np.array([0.07643267, 0.02254561, 0.16332373, 0.00762068, 0.08694207, 0.02946593])
            delta_rot_old = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])
            goal_position = (np.array(offset)).astype(int).tolist()
            tx_result = ctrl.set_goal_position(goal_position)
            if tx_result != COMM_SUCCESS:
                print(f"GroupSyncWrite COMM error: {ctrl._DynamixelController__packet_handler.getTxRxResult(tx_result)}")
            else:
                print("All goal positions sent successfully!")
            
            print("'c' pressed. Calibrated position reached")

        if keyboard.is_pressed('v'):
            delta_rot_goal = np.array([np.deg2rad(-45), np.deg2rad(00), np.deg2rad(25), np.deg2rad(0), np.deg2rad(-25), np.deg2rad(0)])
            delta_rot_old = np.array([0.07643267, 0.02254561, 0.16332373, 0.00762068, 0.08694207, 0.02946593])
            Delta, Rot = adjust_configuration(delta_rot_goal, delta_rot_old)
            delta_rot_old = delta_rot_goal.copy()
            xyz_goal = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1], 
                                    delta_rot_goal[2], delta_rot_goal[3], 
                                    delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
            xyz_old = xyz_goal.copy()
            print("\nTotal bending for each section [deg]:", np.rad2deg(Delta))
            print("Total rotation for each section [deg]:", np.rad2deg(Rot))
            print("Pose reached [x_ee, y_ee, z_ee, x_grip, y_grip, z_grip]:", xyz_goal)
            if Delta [0] > 0:
                motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(Delta[0], Delta[1], Delta[2])
            else:
                motor2_bend_section1, motor1_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(norm(Delta[0]), Delta[1], Delta[2])
            motor_rot_section1 = convert_rad_to_motorposition(Rot[0], 2)
            motor_rot_section2 = convert_rad_to_motorposition(Rot[1], 2)
            motor_rot_section3 = convert_rad_to_motorposition(Rot[2], 2)
            motors_position = [motor_gripper, motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3, motor_pitch]
            print("Input for the rotary motors [4095 encoder based]:", motor_rot_section1, motor_rot_section2, motor_rot_section3)
            print("Input for bending motors [4095 encoder based]:", motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3)
            print("Input for all motors in daisy chain [4095 encoder based]:", motors_position)
            goal_position = (np.array(motors_position) + np.array(offset)).astype(int).tolist()
            tx_result = ctrl.set_goal_position(goal_position)
            if tx_result != COMM_SUCCESS:
                print(f"GroupSyncWrite COMM error: {ctrl._DynamixelController__packet_handler.getTxRxResult(tx_result)}")
            else:
                print("All goal positions sent successfully!")


        pygame.event.pump()
        raw_axes = [joystick.get_axis(i) for i in range(axis_count)]
        axes = []
        for i in range(axis_count):
            val = raw_axes[i] - offset_ps[i]
            if abs(val) < DRIFT_THRESHOLD:
                val = 0.0
            axes.append(round(val, 3))  # Round for cleaner display
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

        translation_zero = np.allclose(axes[:3], 0.0, atol=1e-4)
        rotation_zero    = np.allclose(axes[3:6], 0.0, atol=1e-4)
        gripper_zero     = (buttons[13] == 0 and buttons[14] == 0)
        pitch_zero = (buttons[11] == 0 and buttons[12] == 0)

        if translation_zero and rotation_zero and gripper_zero and pitch_zero:
            # JOYSTICK RELEASED ➔ STOP
            # Only send a stop command once per release
            if last_ps_nonzero:
                # read the actual current motor positions
                raw_positions, _, _, _ = ctrl.read_info(fast_read=False)
                # immediately re‑command those as the goal
                raw_positions[4] = goal_position[4]
                raw_positions[6] = goal_position[6]
                raw_positions[0] = goal_position[0]
                tx = ctrl.set_goal_position(raw_positions.tolist())
                if tx != COMM_SUCCESS:
                    print("STOP: COMM error:", ctrl._packet_handler.getTxRxResult(tx))
                else:
                    print("STOP: re-sent current position, motors braking.")
                motors_position = raw_positions - offset
                print("\n MOTOR ENCODER", motors_position)
                print("\n raw", raw_positions)
                print("\n offset", offset)
                delta_rot_goal[1] = convert_motorposition_to_rad(motors_position[1], 2)
                delta_rot_goal[3] = convert_motorposition_to_rad(motors_position[4], 2)
                delta_rot_goal[5] = convert_motorposition_to_rad(motors_position[6], 2)
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
                print("CONFIGURATION EVALUATED", delta_rot_goal)
                delta_rot_old = delta_rot_goal.copy()
                xyz_goal = configuration_to_cartesian(delta_rot_goal[0], delta_rot_goal[1], 
                        delta_rot_goal[2], delta_rot_goal[3], 
                        delta_rot_goal[4], delta_rot_goal[5]).reshape(-1)
                print("CARTESIAN COORDINATES", xyz_goal)
                xyz_old = xyz_goal.copy()
                pitch_angle = convert_motorposition_to_rad(motors_position[8], ((200/25) * (22/15)*2))
                pitch_angle_old = pitch_angle.copy()
            last_ps_nonzero = False

        else:
            # JOYSTICK ACTIVE ➔ compute new xyz_goal, IK, and send:
            last_ps_nonzero = True

            # xyz_goal = update_goals_with_pscontroller(xyz_goal, axes, buttons, gain_translation, gain_rotation)
            xyz_goal, pitch_angle = update_goals_with_pscontroller_pitch(xyz_goal, pitch_angle, axes, buttons, gain_translation, gain_rotation, gain_pitch)

            gripper_flag = False
            if buttons[13] != 0:
                motor_gripper += gain_gripper
                gripper_flag = True
            if buttons[14] != 0:
                motor_gripper -= gain_gripper
                gripper_flag = True
            
            if gripper_flag:
                gripper_flag = False
                goal_gripper = int(motor_gripper + offset[0])
                raw_positions, _, _, _ = ctrl.read_info(fast_read=False)
                goal_position = raw_positions.tolist()
                goal_position[0] = goal_gripper

                tx_result = ctrl.set_goal_position(goal_position)

                if tx_result != COMM_SUCCESS:
                    err_str = ctrl._packet_handler.getTxRxResult(tx_result)
                    print(f"Motor ID 10 communication error: {err_str}")
                else:
                    print(f"Motor ID 10 moving to raw position {goal_gripper}")
            
            if np.any(xyz_goal != xyz_old) or np.any(pitch_angle != pitch_angle_old):
                print("\n xyz_old:", xyz_old)
                print("xyz_goal:", xyz_goal)
                # print("K_inv", K_inv)
 
                xyz_goal, delta_rot_goal = cartesian_configuration(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv)
                # xyz_goal, delta_rot_goal = cartesian_configuration_noconv(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv)

                Delta, Rot = adjust_configuration(delta_rot_goal, delta_rot_old)

                print("xyz_current:", xyz_goal)
                print("roll pitch yaw [deg]", np.rad2deg(compute_rpy(xyz_goal)))
                print("quaternion", compute_quaternion(xyz_goal))

                x_ee, y_ee, z_ee, x_grip, y_grip, z_grip = xyz_goal
                print("delta_rot_current:", delta_rot_goal)

                xyz_old = xyz_goal.copy()
                delta_rot_old = delta_rot_goal.copy()

                print("\nTotal bending for each section [deg]:", np.rad2deg(Delta))
                print("Total rotation for each section [deg]:", np.rad2deg(Rot))
                print("Total pitch for the base [deg]:", np.rad2deg(pitch_angle)) 
                if Delta [0] > 0:
                    motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(Delta[0], Delta[1], Delta[2])
                else:
                    motor2_bend_section1, motor1_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(norm(Delta[0]), Delta[1], Delta[2])
                motor_rot_section1 = convert_rad_to_motorposition(Rot[0], 2)
                motor_rot_section2 = convert_rad_to_motorposition(Rot[1], 2)
                motor_rot_section3 = convert_rad_to_motorposition(Rot[2], 2)
                motor_pitch = convert_rad_to_motorposition(pitch_angle, ((200/25) * (22/15))*2 )
                motors_position = [motor_gripper, motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3, motor_pitch]
                print("Input for the rotary motors [4095 encoder based]:", motor_rot_section1, motor_rot_section2, motor_rot_section3)
                print("Input for bending motors [4095 encoder based]:", motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3)
                print("Input for pitch motors [4095 encoder based]:", motor_pitch)
                print("Input for all motors in daisy chain [4095 encoder based]:", motors_position)

                # timer = time.time()
                goal_position = (np.array(motors_position) + np.array(offset)).astype(int).tolist()
                tx_result = ctrl.set_goal_position(goal_position)
                if tx_result == COMM_SUCCESS:
                    # immediately start waiting for this goal to arrive
                    # waiting_for_reach = True
                    # last_goal = goal_position.copy()
                    print("Motors moved")
                else:
                    print("COMM error…")
                # print(round(time.time() - timer, 4))
finally:
    pygame.quit()

# Close port
ctrl._port_handler.closePort()
print("Port closed.")