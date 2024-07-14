import math
from rsoccer_gym.Simulators.rsim import RSimVSS

import numpy as np
from rsoccer_gym.Entities import Robot

WHEEL_RADIUS = 0.024

class RSimVSSPID(RSimVSS):

    def __init__(
            self, 
            field_type: int, 
            n_robots_blue: int, 
            n_robots_yellow: int, 
            time_step_ms: int, 
            target: Robot,
            sample_time: int = 200, 
            max_v: float = 50.0, 
            min_v: float = 30.0, 
            max_w: float = 1.5,
            min_w: float = 0.3, 
            robot_half_axis: float = 37.5,
            integral: list = [],
            accumulated_error: float = 0.0,
            last_error: float = 0.0
        ):
        super().__init__(field_type, n_robots_blue, n_robots_yellow, time_step_ms)
        self.target = target
        self.sample_time = sample_time
        self.max_v = max_v
        self.min_v = min_v
        self.max_w = max_w
        self.min_w = min_w
        self.robot_half_axis = robot_half_axis
        self.integral = integral
        self.accumulated_error = accumulated_error
        self.last_error = last_error
        self.linear_velocity = 0
        self.angular_velocity = 0
        self.last_frame = None

    def send_commands(self, commands):
        sim_commands = np.zeros(
            (self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id

            nav = self.navigation(cmd.kP, cmd.kI, cmd.kD)
            sim_commands[rbt_id][0] = nav['v_wheel0']
            sim_commands[rbt_id][1] = nav['v_wheel1']
        self.simulator.step(sim_commands)

    def navigation(self, kP: float, kI: float, kD: float):

        frame = self.get_frame()
        robot = list(frame.robots_blue.values())[0]

        vx = (robot.x - self.last_frame.robots_blue[0].x) / 0.025 if self.last_frame else 0
        vy = (robot.y - self.last_frame.robots_blue[0].y) / 0.025 if self.last_frame else 0

        delta_x = self.target.x - robot.x
        delta_y = self.target.y - robot.y
        dist_to_target = math.sqrt(delta_x ** 2 + delta_y ** 2) * 100

        desired_angle = math.atan2(delta_y, delta_x)
        angle_error = self.smallest_angle_diff(robot.theta, desired_angle)
        rear_angle_error = self.smallest_angle_diff(robot.theta + math.pi, desired_angle)
        rear = False
        if abs(rear_angle_error) < abs(angle_error):
            angle_error = rear_angle_error
            rear = True

        pid_output = self.pid(angle_error, kP, kI, kD)

        angular_velocity = np.clip(pid_output, -self.max_w, self.max_w)

        linear_velocity = self.calculate_linear_v(vx, vy) if dist_to_target > self.robot_half_axis else 0

        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

        left_motor_speed, right_motor_speed = 0, 0

        left_motor_speed = linear_velocity - (angular_velocity * WHEEL_RADIUS)
        right_motor_speed = linear_velocity + (angular_velocity * WHEEL_RADIUS)

        self.last_frame = frame

        if rear:
            return {
                'v_wheel0': -right_motor_speed, 
                'v_wheel1': -left_motor_speed
            }

        return {
            'v_wheel0': right_motor_speed, 
            'v_wheel1': left_motor_speed
        }
    
    def calculate_linear_v(self, vx, vy):
        robot_v_length = math.sqrt(vx ** 2 + vy ** 2) * 100
        acc = 30
        linear_velocity = robot_v_length + acc * 0.025
        return np.clip(linear_velocity, self.min_v, self.max_v)

    def pid(self, angle_error, kP, kI, kD):
        if len(self.integral) == self.sample_time:
            self.accumulated_error -= self.integral[0]
            self.integral.pop(0)

        self.integral.append(angle_error)

        self.accumulated_error += angle_error

        pid_output = kP * angle_error + kI * self.accumulated_error + kD * (angle_error - self.last_error)

        self.last_error = angle_error

        return pid_output

    @staticmethod
    def smallest_angle_diff(a, b):
        diff = (b - a + math.pi) % (2 * math.pi) - math.pi
        return abs(diff)

    def set_target(self, target: Robot):
        self.target = target

    @staticmethod
    def get_velocities(sim, kP: float, kI: float, kD: float):
        sim.navigation(kP, kI, kD)
        return sim.linear_velocity, sim.angular_velocity