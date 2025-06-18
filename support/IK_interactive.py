#IK FROM CARTESIAN COORDINATES OF END EFFECTOR AND GRIPPER TO TENDON LENGTHS controlling also the orientation 
#Input: cartesian coordinates of the end effector and the gripper/goal to reach using the keyboard [x_ee, y_ee, z_ee, x_grip, y_grip, z_grip]
#Output: differential tendond lengths which is the input for the motors L = [l1, l2, l3, l4, l5, l6, l7, l8, l9]

import numpy as np
from sympy import symbols, sqrt, cos, sin, Matrix, lambdify, pi
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
import pickle
from motor_mapping_class import MotorMapping
import keyboard
import pygame
import time

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
    e = xyz_goal - end_effector
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

        # if count == 5:
        #     if norm(xyz_goal[0]) > norm(xyz_goal[1]):
        #         delta_rot_goal = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])
        #     else:
        #         delta_rot_goal = np.array([1e-1, np.pi/2, 1e-1, 1e-2, 1e-1, 1e-2])
        #     end_effector = np.array([0, 0, L1 + rigid_support + L2 + rigid_support + L3, 0, 0, L1 + rigid_support + L2 + rigid_support + L3 + offset_gripper])
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

def convert_rad_to_motorposition(rot):
    gear_ratio = 2
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

def update_goals_with_pscontroller(xyz_goal, axes, buttons, gain_translation, gain_rotation):
    dy, dx, droll, dpitch, dz1, dz2 = axes
    dyaw1, dyaw2 = buttons[9:11]
    dpitch1, dpitch2 = buttons[11:13]
    pitch_angle += (dpitch1 - dpitch2) * gain_pitch 

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

def update_goals_with_pscontroller_try(
    xyz_goal,         # length-6 array: two 3D points in the CURRENT base frame
    pitch_angle,      # total base pitch (radians) before this update
    base_offset,      # z-offset of the pitch pivot
    axes,             # (dy, dx, droll, dpitch, dz1, dz2)
    buttons,          # buttons[9:11]=dyaw1,dyaw2; [11:13]=dpitch1,dpitch2
    gain_translation,
    gain_rotation,
    gain_pitch):
    # --- 1) ensure float ---
    xyz_goal = xyz_goal.astype(float)

    # --- 2) unpack & compute small pitch change ---
    dy, dx, droll, dpitch, dz1, dz2 = axes
    dyaw1, dyaw2      = buttons[9:11]
    dpitch1, dpitch2  = buttons[11:13]

    delta_pitch = (dpitch1 - dpitch2) * gain_pitch
    pitch_angle += delta_pitch

    # --- 3) rotate BOTH goal‐points around the pivot P by -delta_pitch ---
    c0 = np.cos(delta_pitch)
    s0 = np.sin(delta_pitch)
    R_delta = np.array([[ c0, 0,  s0],
                        [  0, 1,   0],
                        [-s0, 0,  c0]])
    P = np.array([0., 0., base_offset])

    for i in (0, 3):
        p_local = xyz_goal[i:i+3]
        # undo the tiny pitch around P
        xyz_goal[i:i+3] = R_delta.T.dot(p_local - P) + P

    # --- 4) build world-frame translation delta_world ---
    delta_world = np.array([-dx, -dy, (dz1 - dz2)/2]) * gain_translation

    # --- 5) apply that delta in the CURRENT base frame ---
    c2 = np.cos(pitch_angle)
    s2 = np.sin(pitch_angle)
    R_full = np.array([[ c2, 0,  s2],
                       [  0, 1,   0],
                       [-s2, 0,  c2]])
    delta_local = R_full.T.dot(delta_world)

    xyz_goal[0:3] += delta_local
    xyz_goal[3:6] += delta_local

    # --- 6) wrist/gripper rotation as before ---
    xyz_goal[3:6] = update_gripper_position(
        xyz_goal[:3],
        xyz_goal[3:6],
        droll * gain_rotation,
        -dpitch * gain_rotation,
        (dyaw1 - dyaw2) * gain_rotation
    )

    # --- 7) now map both points back *into* the ORIGINAL base frame (== world) ---
    world_goal = np.empty_like(xyz_goal)
    for i in (0, 3):
        p_local = xyz_goal[i:i+3]
        world_goal[i:i+3] = R_full.dot(p_local - P) + P

    return world_goal, pitch_angle

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


result_end_effector = end_effector_position ()
result_J = Jacobian()

#------------------------------------------------------------Initial condition------------------------------------------------------------
x_ee = 0
y_ee = 0
z_ee = L1 + rigid_support + L2 + rigid_support+ L3
x_grip = 0
y_grip = 0
z_grip = z_ee + offset_gripper
gain_translation = .1
gain_rotation = .1
gain_gripper = 50
gain_pitch = np.deg2rad(1)
pitch_angle = 0
pitch_angle_old = 0
base_offset = -120 #mm
xyz_goal = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])

delta_rot_goal = np.array([0.07643267, 0.02254561, 0.16332373, 0.00762068, 0.08694207, 0.02946593])
delta_rot_old = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])

section_limits = [1, 0.8, 0.8]
step_size = 1 #mm
rot_step = np.deg2rad(1) #rad
threshold = 1
K = np.diag([1, 1, 1, 1, 1, 1])
K_inv = np.linalg.inv(K)

print("\nNow you can type. For TRANSLATION: q/a for x axis, w/s for y axis, e/d for z-axis")
print("For ROTATION: u/j for x axis (roll), i/k for y axis (pitch), o/l for z-axis (yaw)")
print("type c to RETURN IN CALIBRATED POSITION")
print("and x to END THE LOOP")
# Register the callback for handling keyboard events
keyboard.on_press(handle_key_event)

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
while not exit_flag :
    # Check if 'e' key is pressed to exit the loop 
    if keyboard.is_pressed('x'):
        exit_flag = True
    
    pygame.event.pump()
    raw_axes = [joystick.get_axis(i) for i in range(axis_count)]
    axes = []
    for i in range(axis_count):
        val = raw_axes[i] - offset_ps[i]
        if abs(val) < DRIFT_THRESHOLD:
            val = 0.0
        axes.append(round(val, 3))  # Round for cleaner display
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    
    # xyz_goal = np.array(handle_key_event(None))
    # xyz_goal = update_goals_with_pscontroller(xyz_goal, axes, buttons, gain_translation, gain_rotation)
    # xyz_goal, pitch_angle = update_goals_with_pscontroller_try(xyz_goal, pitch_angle, base_offset, axes, buttons, gain_translation, gain_rotation, gain_pitch)
    xyz_goal, pitch_angle = update_goals_with_pscontroller_pitch(xyz_goal, pitch_angle, axes, buttons, gain_translation, gain_rotation, gain_pitch)

    if keyboard.is_pressed('c'):
        #Initial condition
        x_ee = 0
        y_ee = 0
        z_ee = L1 + rigid_support + L2 + rigid_support+ L3
        x_grip = 0
        y_grip = 0
        z_grip = z_ee + offset_gripper
        pitch_angle = 0
        pitch_angle_old = 0
        xyz_goal = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])
        xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])

        delta_rot_goal = np.array([0.07643267, 0.02254561, 0.16332373, 0.00762068, 0.08694207, 0.02946593])
        delta_rot_old = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])
        
        print("'c' pressed. Calibrated position reached")

    if np.any(xyz_goal != xyz_old) or np.any(pitch_angle != pitch_angle_old):
        print("\n xyz_old:", xyz_old)
        print("xyz_goal:", xyz_goal)

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
        Delta = np.array([delta_rot_goal[0], delta_rot_goal[2], delta_rot_goal[4]])
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
        motor_rot_section1 = convert_rad_to_motorposition(Rot[0])
        motor_rot_section2 = convert_rad_to_motorposition(Rot[1])
        motor_rot_section3 = convert_rad_to_motorposition(Rot[2])
        motor_pitch = convert_rad_to_motorposition(pitch_angle * (200/25) * (22/15))
        pitch_angle_old = pitch_angle.copy()
        motors_position = [motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3, motor_pitch]
        print("Input for the rotary motors [4095 encoder based]:", motor_rot_section1, motor_rot_section2, motor_rot_section3)
        print("Input for bending motors [4095 encoder based]:", motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3)
        print("Input for pitch motors [4095 encoder based]:", motor_pitch)
        print("Input for all motors in daisy chain [4095 encoder based]:", motors_position)
                
    # Sleep for a short time to avoid high CPU usage