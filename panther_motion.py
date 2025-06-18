from panther import Panther
from gamepad_controller import Gamepad
import time

gp = Gamepad()

MAX_LINEAR_VEL = 1.24
MAX_ANGULAR_VEL = 0.63
MAX_LINEAR_ACC = 0.015
MAX_ANGULAR_ACC = 0.02

JOY_CUTOFF = 0.2

myPanther = Panther(max_linear_vel=MAX_LINEAR_VEL,
                    max_angular_vel=MAX_ANGULAR_VEL,
                    max_linear_accel=MAX_LINEAR_ACC,
                    max_angular_accel=MAX_ANGULAR_ACC,
                    ip="10.15.20.2", port = 9090)


while 1:
    print(gp.axis_data)
    time.sleep(0.01)
    fwd = gp.axis_data['left']['y'] * -1
    trn = gp.axis_data['left']['x'] * -1
    
    if abs(fwd) < JOY_CUTOFF:
        fwd = 0
    
    if abs(trn) < JOY_CUTOFF:
        trn = 0

    fwd_vel = MAX_LINEAR_VEL * fwd
    trn_vel = MAX_ANGULAR_VEL * trn

    myPanther.command_robot(fwd_vel, trn_vel)
    