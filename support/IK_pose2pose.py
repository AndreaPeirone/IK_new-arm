#IK FROM CARTESIAN COORDINATES OF END EFFECTOR AND GRIPPER TO TENDON LENGTHS controlling also the orientation 
#Input: cartesian coordinates of the end effector and the gripper/goal to reach using the keyboard [x_ee, y_ee, z_ee, x_grip, y_grip, z_grip]
#Output: differential tendond lengths which is the input for the motors L = [l1, l2, l3, l4, l5, l6, l7, l8, l9]

import numpy as np
from sympy import symbols, cos, sin, Matrix, lambdify, pi
from numpy.linalg import norm
import pickle
from motor_mapping_class import MotorMapping
import math 

# Load the combined mapping
with open('combined_motor_mapping.pkl', 'rb') as f:
    motor_mapping = pickle.load(f)

#Input parameters
L1 = 305 #height 1st section [mm]
L2 = 225 #height 2nd section [mm]
L3 = 228 #height 3rd section [mm]
d = 120  #Tuned diameter of the base [mm]
r = d / 2
offset_gripper = 50 #height of the gripper centre [mm]
rigid_support = 55 #height of the actuation between each section [mm]
threshold = 1

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

        # if count == 5:
        #     delta_rot_goal[3] = delta_rot_goal[3] + np.pi

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
            xyz_goal = xyz_old.copy()
            break

    return xyz_goal, delta_rot_goal

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
        K_inv [1,1] = K_inv[1,1] / 1e2
    if delta_rot_goal[3] < eps:
        # delta_rot_goal[3] = 1e-10
        K_inv [3,3] = K_inv[3,3] / 1e2
    if delta_rot_goal[5] < eps:
        # delta_rot_goal[5] = 1e-10
        K_inv [5,5] = K_inv[5,5] / 1e2

    print("delta_rot_current after boundaries", delta_rot_goal)
    return delta_rot_goal, K_inv

def configuration_tendon(D):
    D1 = D[0]
    D2 = D[1]
    D3 = D[2]
    
    l1 = (L1 / D1 - r) * D1
    l2 = (L1 / D1 + r) * D1
    l3 = (L2 / D2 - r) * D2
    l4 = (L2 / D2 + r) * D2
    l5 = (L3 / D3 - r) * D3
    l6 = (L3 / D3 + r) * D3

    L=np.array([l1, l2, l3, l4, l5, l6])
    return L

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

result_end_effector = end_effector_position ()
result_J = Jacobian()

#Initial condition
x_ee = 0
y_ee = 0
z_ee = L1 + rigid_support + L2 + rigid_support+ L3
x_grip = 0
y_grip = 0
z_grip = z_ee + offset_gripper
xyz_goal = np.array([0, 0, 860, 0, 0, 860 + offset_gripper])
xyz_old = np.array([x_ee, y_ee, z_ee, x_grip, y_grip, z_grip])

delta_rot_goal = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])
delta_rot_old = np.array([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2])

section_limits = [1, 0.8, 0.8]
step_size = 1 #mm
rot_step = np.deg2rad(1) #degree
K_elements = [1, 1, 1, 1, 1, 1] 
K = np.diag(K_elements)
K_inv = np.linalg.inv(K)

xyz_goal, delta_rot_goal = cartesian_configuration(xyz_goal, xyz_old, delta_rot_goal, delta_rot_old, K_inv)
# L=configuration_tendon(Delta)

if (delta_rot_goal[4] * delta_rot_old[4]) < 0:
    delta_rot_goal[5] = delta_rot_goal[5] + np.pi
    delta_rot_goal[4] = - delta_rot_goal[4]
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

print("\nTotal bending for each section [deg]:")
print(np.rad2deg(Delta))
print("\nTotal rotation for each section [deg]:")
print(np.rad2deg(Rot))

motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3 = motor_mapping(Delta[0], Delta[1], Delta[2])
motor_rot_section1 = convert_rad_to_motorposition(Rot[0])
motor_rot_section2 = convert_rad_to_motorposition(Rot[1])
motor_rot_section3 = convert_rad_to_motorposition(Rot[2])
motors_position = [motor_rot_section1, motor1_bend_section1, motor2_bend_section1, motor_rot_section2, motor_bend_section2, motor_rot_section3, motor_bend_section3]
print("\nInput for the rotary motors [4095 encoder based]:")
print(motor_rot_section1, motor_rot_section2, motor_rot_section3)
print("\nInput for bending motors [4095 encoder based]:")
print(motor1_bend_section1, motor2_bend_section1, motor_bend_section2, motor_bend_section3)
print("\nInput for all motors in daisy chain [4095 encoder based]:")
print(motors_position)