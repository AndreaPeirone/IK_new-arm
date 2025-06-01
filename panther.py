import roslibpy
import time
import numpy as np
import sys

class Panther:
    def __init__(self, max_linear_vel=0.1, max_angular_vel=0.1, 
                 max_linear_accel=0.01, max_angular_accel=0.01, 
                ip="10.15.20.2", port = 9090) -> None:
        # Initialize the robot with acceleration limits for linear and angular velocities.
        self.max_linear_accel = max_linear_accel
        self.max_angular_accel = max_angular_accel
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0

        self.ip = ip
        self.port = port

        self.client = roslibpy.Ros(host=self.ip, port=self.port)
        self.client.run()

        print("\n\n-----------------------------------")
        if self.client.is_connected:
            print("Connected to panther")

        else:
            print("Cannot connect")
            sys.exit()

        # Publishers
        self.cmd_vel_publisher = roslibpy.Topic(self.client, '/cmd_vel', 'geometry_msgs/Twist')

        # Subscribers
        battery_listener = roslibpy.Topic(self.client, '/panther/battery', 'sensor_msgs/BatteryState')
        battery_listener.subscribe(self.battery_callback)

        while hasattr(self, "bat") == False:
                pass

        percentage = round(self.bat["percentage"],3)

        print("\nCurrent battery charge:")
        print(percentage * 100, "%")

        if percentage < 0.5:
            print("Battery less than 50%, consider charging!!!")

        elif percentage < 0.3:
            print("CHARGE BATTERY!")
            sys.exit()

    def command_robot(self, target_linear, target_angular):
        delta_linear = target_linear - self.current_linear_velocity
        delta_angular = target_angular - self.current_angular_velocity

        if abs(delta_linear) > self.max_linear_accel:
            delta_linear = self.max_linear_accel * np.sign(delta_linear)

        if abs(delta_angular) > self.max_angular_accel:
            delta_angular = self.max_angular_accel * np.sign(delta_angular)

        self.current_linear_velocity += delta_linear
        self.current_angular_velocity += delta_angular

        self.current_linear_velocity = min(self.max_linear_vel, max(-self.max_linear_vel, self.current_linear_velocity))
        self.current_angular_velocity = min(self.max_angular_vel, max(-self.max_angular_vel, self.current_angular_velocity))
        
        msg = roslibpy.Message({
            'linear': {'x': self.current_linear_velocity, 'y': 0.0, 'z': 0.0},
            'angular': {'x': 0.0, 'y': 0.0, 'z': self.current_angular_velocity}
        })

        # Publish the message to the ROS topic.
        self.cmd_vel_publisher.publish(msg)
        time.sleep(0.01)

    def change_max_vel(self, max_linear = None, max_angular = None):
        if max_linear is not None:
            self.max_linear_vel = max_linear
        
        if max_angular is not None:
            self.max_angular_vel = max_angular
        
    def battery_callback(self, message):
        self.bat = message