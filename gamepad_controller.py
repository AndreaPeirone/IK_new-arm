import pygame
from pygame.locals import *
from threading import Thread
import time
import numpy as np


class Gamepad:
    def __init__(self) -> None:

        self.button_labels = [("cross", 0), ("circle", 1), ("triangle", 2), ("square", 3), 
                            ("up", None), ("down", None), ("left", None), ("right", None), 
                            ("R1", 5), ("R2", 7), ("L1", 4), ("L2", 6), 
                            ("share", 8), ("options", 9), ("start", 10), ("joy right", 12), ("joy left", 11)]

        self.axis_data = {"left":{"x":0, "y":0, "trigger":0}, "right":{"x":0, "y":0, "trigger":0}}

        self.button_data, self.button_correspondance = self.__populate_button_dictionary()

        self.__thread = Thread(target=self.__get_joystick_data)
        self.__thread.daemon = True
        self.__thread.start()
        
    def __populate_button_dictionary(self):
        dictionary = {}
        button_correspondance = {}
        for label in self.button_labels:
            dictionary[label[0]] = 0
            if label[1] is not None:
                button_correspondance[label[1]] = label[0]

        return dictionary, button_correspondance

    def __get_joystick_data(self):
        pygame.init()
        pygame.joystick.init()
        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except pygame.error:
            print("Cannot connect to gamepad")

        while 1:
            time.sleep(0.00001)
            for e in pygame.event.get():
                if e.type == pygame.locals.JOYHATMOTION:
                    if e.value[1] == 1:
                        self.button_data["up"] = 1
                    else:
                        self.button_data["up"] = 0

                    if e.value[1] == -1:
                        self.button_data["down"] = 1
                    else:
                        self.button_data["down"] = 0

                    if e.value[0] == 1:
                        self.button_data["right"] = 1
                    else:
                        self.button_data["right"] = 0

                    if e.value[0] == -1:
                        self.button_data["left"] = 1
                    else:
                        self.button_data["left"] = 0
                        
                elif e.type == pygame.locals.JOYAXISMOTION:
                    val = round(e.value, 4)
                    if e.axis == 0:
                        self.axis_data["left"]["x"] = val
                    elif e.axis == 1:
                        self.axis_data["left"]["y"] = val
                    elif e.axis == 2:
                        self.axis_data["left"]["trigger"] = val
                    elif e.axis == 3:
                        self.axis_data["right"]["x"] = val
                    elif e.axis == 4:
                        self.axis_data["right"]["y"] = val
                    elif e.axis == 5:
                        self.axis_data["right"]["trigger"] = val

                elif e.type == pygame.locals.JOYBUTTONDOWN:
                    entry_name = self.button_correspondance[e.button]
                    self.button_data[entry_name] = 1
                elif e.type == pygame.locals.JOYBUTTONUP:
                    entry_name = self.button_correspondance[e.button]
                    self.button_data[entry_name] = 0


class Tendon_gamepad_controller(Gamepad):
    def __init__(self, reactiveness) -> None:
        super().__init__()

        self.reactiveness = reactiveness

    def tendon_position(self, prev_dt):
        new_dt = np.copy(prev_dt)

        dt1_raw = self.axis_data['right']['x']
        dt2_raw = -self.axis_data['right']['y']

        for i, raw in enumerate([dt1_raw, dt2_raw]):
            if abs(raw) < 0.05:
                raw = 0

            vel = raw * self.reactiveness
            
            new_dt[i] += vel

        return np.round(new_dt, decimals=6)

    
    def reset_dt(self):
        return self.button_data['square']
    
    def quit_program(self):
        return self.button_data['cross']
