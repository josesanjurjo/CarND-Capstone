import time
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

MIN_SPEED = 0 # No reverse driving
MAX_SPEED = 100 # Sane speed limit

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, 
                        wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle,
                        throttle_kp, throttle_ki, throttle_kd,
                        brake_kp, brake_ki, brake_kd,
                        steering_tau, rate):
        
        # Store parameters
        self.total_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.brake_deadband = brake_deadband
        self.wheel_radius = wheel_radius
        self.rate = rate

        # Init controllers
        self.throttle_pid = PID(throttle_kp, throttle_ki, throttle_kd, 0., accel_limit)
        self.brake_pid = PID(brake_kp, brake_ki, brake_kd, decel_limit, 0.)
        self.yaw_control = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        self.steering_filter = LowPassFilter(steering_tau, 1./rate)

        self.last_time = None

    def control(self,   cmd_linear_velocity,
                        cmd_angular_velocity,
                        current_linear_velocity,
                        dbw_enabled):

        cmd_throttle = 0.
        cmd_brake = 0.
        cmd_steering = 0.
        
        if self.last_time is None or not dbw_enabled:
            self.throttle_pid.reset()
            self.brake_pid.reset()
            self.steering_filter.reset()
            self.last_time = time.time()
        else:
            dt = time.time() - self.last_time
            
            # Step speed controller
            speed_error = cmd_linear_velocity.x - current_linear_velocity.x

            if (speed_error > 0):
                # Step throttle controller
                cmd_throttle = self.throttle_pid.step(speed_error, dt)
                
                # Reset brake filter
                self.brake_pid.reset()
            else:
                # Step brake controller
                tmp_decel = self.brake_pid.step(speed_error, dt)
                tmp_force = tmp_decel * self.total_mass
                tmp_brake = tmp_force * self.wheel_radius
                
                if (abs(tmp_brake) > self.brake_deadband):
                    cmd_brake = tmp_brake
                
                # Reset throttle filter
                self.throttle_pid.reset()

            # Step steering controller
            tmp_steering = self.yaw_control.get_steering(cmd_linear_velocity.x, cmd_angular_velocity.z, current_linear_velocity.x)
            cmd_steering = self.steering_filter.filt(tmp_steering)

        # Return throttle, brake, steer
        return cmd_throttle, cmd_brake, cmd_steering
