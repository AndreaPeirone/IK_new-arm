import time
import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
joystick.get_power_level()

print(f"Initialized controller: {joystick.get_name()}")

while 1:
    time.sleep(0.1)
    pygame.event.pump()
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    print(buttons)
    x_value, y_value = joystick.get_hat(0)
    print(x_value)
    print(y_value)
    axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    print(axes)
