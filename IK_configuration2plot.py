import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from sympy import Matrix, sqrt, cos, sin, symbols, lambdify

# ------------------------
# Helper rotation matrices
# ------------------------
def rotation_matrix_y(delta):
    """Rotation about y-axis by delta."""
    return np.array([[cos(delta), 0, sin(delta)],
                     [0, 1, 0],
                     [-sin(delta), 0, cos(delta)]])

def rotation_matrix_z(rot):
    """Rotation about z-axis by rot."""
    return np.array([[cos(rot), -sin(rot), 0],
                     [sin(rot),  cos(rot), 0],
                     [0, 0, 1]])

def seg_rotation(delta, rot):
    """
    Returns the rotation matrix associated with a bending angle delta and
    an in-plane rotation rot. In our case the local bending is assumed to
    occur in the x–z plane and then the bending plane is rotated by rot about z.
    """
    return rotation_matrix_z(rot) @ rotation_matrix_y(delta)

# ------------------------
# Constant curvature arc generator
# ------------------------
def arc_segment(L, delta, rot, n_points=100):
    """
    Compute a constant curvature arc in a local frame.
    
    The arc starts at the origin and is generated as:
      - If |delta| is nearly 0, the arc is a straight line along z.
      - Otherwise, let R = L/delta, and for a normalized parameter s in [0,1]:
            x(s) = R*(1 - cos(delta * s))
            z(s) = R*sin(delta * s)
        (y is 0 in the local frame.)
    
    Finally, the local arc is rotated about z by rot.
    """
    s = np.linspace(0, 1, n_points)
    pts_local = np.zeros((n_points, 3))
    if abs(delta) < 1e-6:
        # Straight line (vertical) along z
        pts_local[:, 2] = L * s
    else:
        R_val = L / delta
        pts_local[:, 0] = R_val * (1 - np.cos(delta * s))
        pts_local[:, 2] = R_val * np.sin(delta * s)
    # Rotate local arc about z by rot to orient the bending plane
    Rz = rotation_matrix_z(rot)
    pts_rotated = pts_local @ Rz.T
    return pts_rotated

# ------------------------
# Forward kinematics with arcs and rigid offsets
# ------------------------
def compute_fk_arcs(delta1, rot1, delta2, rot2, delta3, rot3):
    """
    Computes the forward kinematics based on three sequential arcs with
    rigid (non-deformable) offsets between them.
    
    Parameters: (all angles in radians)
      - Section 1: arc length L1 (305 mm), bending (delta1, rot1)
      - A rigid offset of rigid_support (55 mm) applied in the direction of R1.
      - Section 2: arc length L2 (225 mm), bending (delta2, rot2)
      - A second rigid offset of rigid_support (55 mm) along R2.
      - Section 3: arc length L3 (228 mm), bending (delta3, rot3)
      - A final offset (offset_gripper = 50 mm) along the final frame orientation.
    
    Returns a dictionary with:
      - 'arc1', 'arc2', 'arc3': arrays of points along each arc (global coordinates)
      - 'key_points': key points including:
            - End of arc1 (t1)
            - Start of section 2 (after first rigid offset)
            - End of arc2 (t2)
            - Start of section 3 (after second rigid offset)
            - End of arc3 (t3)
            - Gripper position (t_gripper)
    """
    # --- Define dimensional constants [mm] ---
    L1 = 305         # length of arc/section 1
    L2 = 225         # length of arc/section 2
    L3 = 228         # length of arc/section 3
    rigid_support = 55   # rigid offset between sections
    offset_gripper = 50  # final gripper offset
    
    # -------------------
    # Section 1
    # -------------------
    # Base frame: start at origin.
    P0 = np.array([0, 0, 0])
    # Compute arc 1 in its local frame (bending in x–z) and then rotated by rot1.
    arc1_local = arc_segment(L1, delta1, rot1, n_points=100)
    # For section 1, the global arc is the same as the local one because the base frame is identity.
    arc1_global = P0 + arc1_local
    # At the end of arc1, the orientation is given by
    R1 = seg_rotation(delta1, rot1)
    P1 = arc1_global[-1]  # end point of arc 1

    # Rigid offset after section 1: move along the z-axis of frame R1
    offset_vec = np.array([0, 0, rigid_support])
    start_section2 = P1 + R1 @ offset_vec

    # -------------------
    # Section 2
    # -------------------
    # Compute arc 2 in its own local frame. This arc (length L2) is with parameters (delta2, rot2).
    arc2_local = arc_segment(L2, delta2, rot2, n_points=100)
    # The global arc for section 2 is expressed in the frame of section 1 (R1) and translated by start_section2.
    arc2_global = start_section2 + (R1 @ arc2_local.T).T
    # Orientation update: the rotation at the end of section 2 is
    R2 = R1 @ seg_rotation(delta2, rot2)
    P2 = arc2_global[-1]  # end point of arc 2

    # Add rigid offset after section 2 along the new frame R2
    start_section3 = P2 + R2 @ offset_vec

    # -------------------
    # Section 3
    # -------------------
    # Compute arc 3 in its local frame using (delta3, rot3) and arc length L3.
    arc3_local = arc_segment(L3, delta3, rot3, n_points=100)
    arc3_global = start_section3 + (R2 @ arc3_local.T).T
    # Final orientation update after section 3:
    R3 = R2 @ seg_rotation(delta3, rot3)
    P3 = arc3_global[-1]  # end of arc 3

    # Final gripper offset along the new frame’s z-axis (R3)
    P_gripper = P3 + R3 @ np.array([0, 0, offset_gripper])
    
    # -------------------
    # Assemble key points
    # -------------------
    # Key points: end of arc1 (t1), start_section2, end of arc2 (t2),
    # start_section3, end of arc3 (t3), gripper position.
    key_points = np.array([P1, start_section2, P2, start_section3, P3, P_gripper])
    
    return {
        'arc1': arc1_global,
        'arc2': arc2_global,
        'arc3': arc3_global,
        'key_points': key_points
    }

def end_effector_position():
    # Define symbolic variables
    delta1, rot1, delta2, rot2, delta3, rot3 = symbols('delta1 rot1 delta2 rot2 delta3 rot3')
    
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
        [cos(rot1 + rot2)**2*(cos(delta2)-1)+1, sin(rot1 + rot2)*cos(rot1 + rot2)*(cos(delta2)-1), cos(rot1 + rot2)*sin(delta2)],
        [sin(rot1 + rot2)*cos(rot1 + rot2)*(cos(delta2)-1), sin(rot1 + rot2)**2*(cos(delta2)-1)+1, sin(rot1 + rot2)*sin(delta2)],
        [-cos(rot1 + rot2)*sin(delta2), -sin(rot1 + rot2)*sin(delta2), cos(delta2)]
    ])
    t12 = Matrix([
        L2/delta2 * cos(rot1 + rot2)*(1-cos(delta2)),
        L2/delta2 * sin(rot1 + rot2)*(1-cos(delta2)),
        L2/delta2 * sin(delta2)
    ])
    # --- Correction: use R1 instead of R12 for the transformation ---
    t2 = start_section2 + R1 * t12
    R2 = R1 * R12
    
    offset2 = Matrix([0, 0, rigid_support])
    start_section3 = t2 + R2 * offset2
    
    # --- Section 3 ---
    R23 = Matrix([
        [cos(rot1 + rot2 + rot3)**2*(cos(delta3)-1)+1, sin(rot1 + rot2 + rot3)*cos(rot1 + rot2 + rot3)*(cos(delta3)-1), cos(rot1 + rot2 + rot3)*sin(delta3)],
        [sin(rot1 + rot2 + rot3)*cos(rot1 + rot2 + rot3)*(cos(delta3)-1), sin(rot1 + rot2 + rot3)**2*(cos(delta3)-1)+1, sin(rot1 + rot2 + rot3)*sin(delta3)],
        [-cos(rot1 + rot2 + rot3)*sin(delta3), -sin(rot1 + rot2 + rot3)*sin(delta3), cos(delta3)]
    ])
    t23 = Matrix([
        L3/delta3 * cos(rot1 + rot2 + rot3)*(1-cos(delta3)),
        L3/delta3 * sin(rot1 + rot2 + rot3)*(1-cos(delta3)),
        L3/delta3 * sin(delta3)
    ])
    t3 = start_section3 + R2 * t23
    R3 = R2 * R23

    # --- Gripper Offset ---
    t34 = Matrix([0, 0, offset_gripper])
    t3_gripper = t3 + R3 * t34

    # Stack key points: start_section2, start_section3, t3, and t3_gripper
    t_total = Matrix.vstack(start_section2, start_section3, t3, t3_gripper)
    
    # Return a callable function
    return lambdify((delta1, rot1, delta2, rot2, delta3, rot3), t_total, modules=['numpy'])



# ------------------------
# Plotting function
# ------------------------
def plot_robot_fk(delta1, rot1, delta2, rot2, delta3, rot3):
    """
    Plot the arc-based FK model and compare with symbolic FK evaluation.
    """
    fk = compute_fk_arcs(delta1, rot1, delta2, rot2, delta3, rot3)
    arc1 = fk['arc1']
    arc2 = fk['arc2']
    arc3 = fk['arc3']
    pts = fk['key_points']
    
    # Unpack key points
    labels = ['End Arc1 (P1)', 'Start Sec2', 'End Arc2 (P2)', 'Start Sec3', 'End Arc3 (P3)', 'Gripper']
    
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the arc-based robot
    ax.plot(arc1[:,0], arc1[:,1], arc1[:,2], 'r-', linewidth=2, label='Arc 1')
    ax.plot([pts[0,0], pts[1,0]], [pts[0,1], pts[1,1]], [pts[0,2], pts[1,2]], 'k--')
    ax.plot(arc2[:,0], arc2[:,1], arc2[:,2], 'g-', linewidth=2, label='Arc 2')
    ax.plot([pts[2,0], pts[3,0]], [pts[2,1], pts[3,1]], [pts[2,2], pts[3,2]], 'k--')
    ax.plot(arc3[:,0], arc3[:,1], arc3[:,2], 'b-', linewidth=2, label='Arc 3')
    ax.plot([pts[4,0], pts[5,0]], [pts[4,1], pts[5,1]], [pts[4,2], pts[5,2]], 'k--')

    # Mark arc-based key points
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k', s=50)
    for i, pt in enumerate(pts):
        ax.text(pt[0], pt[1], pt[2], f' {labels[i]}', fontsize=9, color='k')

    # -------------------------------
    # Compute and overlay symbolic FK
    # -------------------------------
    end_effector_fn = end_effector_position()
    sym_output = end_effector_fn(delta1, rot1, delta2, rot2, delta3, rot3)
    print("xyz_goal",sym_output[6:])
    # sym_output is a flat vector of shape (12,), reshape it to 4x3
    sym_pts = np.array(sym_output).reshape((4, 3))
    sym_labels = ['start_sec2 (sym)', 'start_sec3 (sym)', 'end_sec3 (sym)', 'gripper (sym)']
    
    # Plot symbolic FK points
    ax.scatter(sym_pts[:,0], sym_pts[:,1], sym_pts[:,2], c='m', marker='x', s=70, label='Symbolic FK')
    for i, pt in enumerate(sym_pts):
        ax.text(pt[0], pt[1], pt[2], f' {sym_labels[i]}', fontsize=9, color='magenta')

    # Final plot settings
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title('Comparison: Arc-based FK vs Symbolic FK')
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.show()

# ------------------------
# Example usage:
# Set the joint angles in radians.
# (rot is the rotation about z and delta is the bending angle.)
# Feel free to adjust these values to test different configurations.
delta1 = np.deg2rad(-64.47526623       ) # bending angle for first arc
rot1   = np.deg2rad( 62.24835531   ) # orientation of first arc (about z)
delta2 = np.deg2rad( 45.79463704) # bending angle for second arc
rot2   = np.pi + np.deg2rad(  268.82666792 ) # orientation of second arc (about z)
delta3 = np.deg2rad( 45.83662361 ) # bending angle for third arc
rot3   = np.pi + np.deg2rad(-354.224499295) # orientation of third arc (about z)
# Robot parameters (example values)
L1 = 305 #height 1st section [mm]
L2 = 225 #height 2nd section [mm]
L3 = 228 #height 3rd section [mm]
offset_gripper = 50 #height of the gripper centre [mm]
rigid_support = 55 #height of the actuation between each section [mm]

plot_robot_fk(delta1, rot1, delta2, rot2, delta3, rot3)
