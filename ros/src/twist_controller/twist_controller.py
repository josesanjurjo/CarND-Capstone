import time
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

MIN_SPEED = 0 # No reverse driving
MAX_SPEED = 100 # Sane speed limit

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, 
                        wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle,
                        velocity_kp, velocity_ki, velocity_kd,
                        steering_kp, steering_ki, steering_kd):
        
        # Store parameters
        self.total_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.brake_deadband = brake_deadband
        self.wheel_radius = wheel_radius

        # Init controllers
        self.velocity_pid = PID(velocity_kp, velocity_ki, velocity_kd, decel_limit, accel_limit)
        self.yaw_control = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        self.steering_pid = PID(steering_kp, steering_ki, steering_kd, -max_steer_angle, max_steer_angle)

    def control(self,   cmd_linear_velocity,
                        cmd_angular_velocity,
                        current_linear_velocity,
                        dbw_enabled,
                        dt):

        cmd_throttle = 0.
        cmd_brake = 0.
        cmd_steering = 0.
        
        if not dbw_enabled:
            self.velocity_pid.reset()
            self.steering_pid.reset()
        else:
            # Step speed controller
            speed_error = cmd_linear_velocity.x - current_linear_velocity.x
            cmd_acceleration = self.velocity_pid.step(speed_error, dt)

            if (cmd_acceleration > 0):
                # Set throttle
                cmd_throttle = cmd_acceleration
            else:
                # Step brake
                tmp_force = -cmd_acceleration * self.total_mass
                tmp_brake = tmp_force * self.wheel_radius
                
                if (abs(tmp_brake) > self.brake_deadband):
                    cmd_brake = tmp_brake

            # Step steering controller
            tmp_steering = self.yaw_control.get_steering(cmd_linear_velocity.x, cmd_angular_velocity.z, current_linear_velocity.x)
            cmd_steering = self.steering_pid.step(tmp_steering, dt)

        # Return throttle, brake, steer
        return cmd_throttle, cmd_brake, cmd_steering
