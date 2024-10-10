import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium.utils.colorize import colorize

import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree
from rsoccer_gym.Simulators.rsim import RSimVSS

from envs.extra.RobotWithPID import RobotWithPID
from collections import namedtuple
from rsoccer_gym.Render import COLORS
from envs.navigation import dist_to, abs_smallest_angle_diff, smallest_angle_diff

import pygame


N_ROBOTS_BLUE = 1
N_ROBOTS_YELLOW = 0
TIMEOUT = 3.5
MAX_DIST_TO_TARGET = 1
MAX_DIST_ANGLE_REWARD = 8
FIELD_WIDTH = 1.3
FIELD_LENGTH = 1.5
ROBOT_WIDTH = 0.008
REWARD_NORM_BOUND = 5.0
TO_CENTIMETERS = 100

MAX_VELOCITY = 1.2

TARGET           = (255 /255, 50  /255, 50  /255)
TARGET_TAG_BLUE  = (138 /255, 164 /255, 255 /255)
TARGET_TAG_GREEN = (204 /255, 220 /255, 153 /255)


ANGLE_TOLERANCE: float = np.deg2rad(5)  # 5 degrees
SPEED_MIN_TOLERANCE: float = 0.05  # m/s == 5 cm/s
SPEED_MAX_TOLERANCE: float = 0.3  # m/s == 30 cm/s
DIST_TOLERANCE: float = 0.05  # m == 5 cm

Point2D = namedtuple("Point2D", ["x", "y"])

TIME_STEP_DIFF = 0.16 / 0.025

class VSSPIDTuningEnv(VSSBaseEnv):
    """This environment controls a single robot to go to a target position using PID control. 

        Description:
        Observation:
            Type: Box(11)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               Target X
            1               Target Y
            2               Target sin(theta)
            3               Target cos(theta)
            4               Robot X
            5               Robot Y
            6               Robot sin(theta)
            7               Robot cos(theta)
            8               Robot Vx
            9               Robot Vy
            10              Robot v_theta
        Actions:
            Type: Box(3, )
            Num             Action
            0 kP            Robot kP
            1 kI            Robot kI
            2 kD            Robot kD
        Reward:
            Sum of Rewards:
                Goal
                Move to Target
                Energy Penalty
                Angle Difference between Robot and Target
        Starting State:
            Randomized Robots and Target initial Position
        Episode Termination:
            Target is reached or Episode length is greater than 3.5 seconds
    """

    def __init__(self, render_mode=None, repeat_action=int(TIME_STEP_DIFF), max_steps=150):
        super().__init__(field_type=0, n_robots_blue=N_ROBOTS_BLUE, n_robots_yellow=N_ROBOTS_YELLOW,
                         time_step=0.025, render_mode=render_mode)
        
        self.rsim = RSimVSS(
            field_type=0,
            n_robots_blue=N_ROBOTS_BLUE,
            n_robots_yellow=N_ROBOTS_YELLOW,
            time_step_ms=int(self.time_step * 1000)
        )

        self.action_space = gym.spaces.Box(low=0, high=1,
                                           shape=(1, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(11, ), dtype=np.float32)

        # Initialize Class Atributes
        self.previous_target_potential = None
        self.actions: Dict = None
        self.v_wheel_deadzone = 0.05

        self.max_kP = 6
        self.max_kD = 20
        self.max_kI = 0.3

        self.last_speed_reward = 0

        self.target_point: Point2D = Point2D(0, 0)
        self.target_angle: float = 0.0
        self.target_velocity: Point2D = Point2D(0, 0)

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        self.cur_action = []
        self.all_actions = []
        self.actual_action = None
        self.last_action = None
        self.last_dist_reward = 0
        self.last_angle_reward = 0
        self.last_speed_reward = 0
        self.action_color = COLORS["PINK"]

        self.robot_path = []
        self.repeat_action = repeat_action
        FPS = 120
        if self.repeat_action > 1:
            FPS = np.ceil((1200 // self.repeat_action) / 10)

        self.metadata["render_fps"] = FPS

        self.reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_action_var": 0,
            "reward_objective": 0,
            "reward_total": 0,
            "reward_steps": 0,
            "reward_path_consistency": 0,
        }

        self.max_steps = max_steps

        # Render
        self.window_surface = None
        self.window_size = self.field_renderer.window_size
        self.clock = None

        # PID
        self.integral = []
        self.accumulated_error = 0
        self.last_error = 0
        self.sample_time = 200

        # Navigation
        MAX_WHEEL_SPEED = 50 # 50 rad/s
        self.max_v = MAX_WHEEL_SPEED * self.field.rbt_wheel_radius
        self.max_w = np.rad2deg(MAX_WHEEL_SPEED)


        print('Environment initialized')

    def reset(self, *, seed=None, options=None):
        self.actions = None
        self.reward_info = None
        self.previous_target_potential = None
        self.steps = 0

        for ou in self.ou_actions:
            ou.reset()

        return super().reset(seed=seed, options=options)


    def step(self, action):
        self.cur_action = action
        for _ in range(self.repeat_action):
            self.steps += 1
            # Join agent action with environment actions
            commands: List[RobotWithPID] = self._get_commands(action)
            
            # Send command to simulator
            self.rsim.send_commands(commands)
            self.sent_commands = commands

            # Get Frame from simulator
            self.last_frame = self.frame
            self.frame = self.rsim.get_frame()

            self.actual_action = [
                self.frame.robots_blue[0].x,
                self.frame.robots_blue[0].y,
                self.frame.robots_blue[0].theta
            ]

            # Calculate environment observation, reward and done condition
            observation = self._frame_to_observations()
            reward, done = self._calculate_reward_and_done()

            if done:
                break   
                
        self.robot_path.append(
            (self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, self.reward_info

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.target_point.x))
        observation.append(self.norm_pos(self.target_point.y))
        observation.append(np.sin(self.target_angle))
        observation.append(np.cos(self.target_angle))
        # observation.append(self.norm_v(self.target_velocity.x))
        # observation.append(self.norm_v(self.target_velocity.y))
    
        observation.append(self.norm_pos(self.frame.robots_blue[0].x))
        observation.append(self.norm_pos(self.frame.robots_blue[0].y))
        observation.append(np.sin(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(np.cos(np.deg2rad(self.frame.robots_blue[0].theta)))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_x))
        observation.append(self.norm_v(self.frame.robots_blue[0].v_y))
        observation.append(self.norm_w(self.frame.robots_blue[0].v_theta))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions
        # kP, kI, kD = self._actions_to_PID(actions)
        kP = actions[0] * self.max_kP

        v_wheel0, v_wheel1 = self.navigation(kP, 0, 0)
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        self.all_actions.append(kP)

        return commands

    def _calculate_reward_and_done(self):
        done = False
        reward = 0
        dist_reward, distance = self._dist_reward()
        angle_reward, angle_diff = self._angle_reward()
        path_reward = self._path_consistency_reward()

        robot_dist = np.linalg.norm(
            np.array(
                [
                    self.frame.robots_blue[0].x - self.target_point.x,
                    self.frame.robots_blue[0].y - self.target_point.y,
                ]
            )
        )
        # robot_vel_error = np.linalg.norm(
        #     np.array(
        #         [
        #             self.frame.robots_blue[0].v_x - self.target_velocity.x,
        #             self.frame.robots_blue[0].v_y - self.target_velocity.y,
        #         ]
        #     )
        # )

        if (
            distance < DIST_TOLERANCE
            and angle_diff < ANGLE_TOLERANCE
        ):
            done = True
            reward = 10
            self.reward_info["reward_objective"] += reward
            print(colorize("TARGET REACHED!", "green", bold=True, highlight=True))
        else:
            reward = dist_reward + angle_reward

        if done or self.steps >= self.max_steps:
            # pairwise distance between all actions
            action_var = np.linalg.norm(
                np.array(self.all_actions[1:]) - np.array(self.all_actions[:-1])
            )
            self.reward_info["reward_action_var"] = action_var
        
        reward += path_reward - (self.steps - self.max_steps/2) * 2


        self.reward_info["reward_dist"] += dist_reward
        self.reward_info["reward_angle"] += angle_reward
        self.reward_info["reward_total"] += reward
        self.reward_info["reward_path_consistency"] += path_reward
        self.reward_info["reward_steps"] += self.steps

        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2


        def get_random_x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def get_random_y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def get_random_theta():
            return random.uniform(0, 360)

        def get_random_speed():
            return random.uniform(0, self.max_v)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=-4, y=-2)

        self.target_point = Point2D(x=get_random_x(), y=get_random_y())
        self.target_angle = np.deg2rad(get_random_theta())
        random_speed: float = 0
        random_velocity_direction: float = np.deg2rad(get_random_theta())

        self.target_velocity = Point2D(
            x=random_speed * np.cos(random_velocity_direction),
            y=random_speed * np.sin(random_velocity_direction),
        )

        # Adjust speed tolerance according to target velocity
        target_speed_norm = np.sqrt(
            self.target_velocity.x**2 + self.target_velocity.y**2
        )
        self.SPEED_TOLERANCE = (
            SPEED_MIN_TOLERANCE
            + (SPEED_MAX_TOLERANCE - SPEED_MIN_TOLERANCE)
            * target_speed_norm
            / self.max_v
        )

        min_gen_dist = 0.2

        places = KDTree()
        places.insert((self.target_point.x, self.target_point.y))
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (get_random_x(), get_random_y())

            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(
                id=i, yellow=False, x=pos[0], y=pos[1], theta=get_random_theta()
            )

        self.start_position = Point2D(pos_frame.robots_blue[0].x, pos_frame.robots_blue[0].y)

        for i in range(self.n_robots_yellow):
            pos = (get_random_x(), get_random_y())
            while places.get_nearest(pos)[1] < min_gen_dist:
                pos = (get_random_x(), get_random_y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(
                id=i, yellow=True, x=pos[0], y=pos[1], theta=get_random_theta()
            )
        self.last_action = None
        self.last_dist_reward = 0
        self.last_angle_reward = 0
        self.last_speed_reward = 0
        self.all_actions = []
        self.reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_action_var": 0,
            "reward_objective": 0,
            "reward_total": 0,
            "reward_steps": 0,
            "reward_path_consistency": 0,
        }
        self.robot_path = [(pos_frame.robots_blue[0].x, pos_frame.robots_blue[0].y)] * 2

        return pos_frame

    def _actions_to_PID(self, actions):
        kP = np.clip(abs(actions[0]), 0, self.max_kP)
        kI = np.clip(abs(actions[1]), 0, self.max_kI)
        kD = np.clip(abs(actions[2]), 0, self.max_kD)

        return kP , kI , kD
    
    def _dist_reward(self):
        robot_pos = Point2D(self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
        target = self.target_point
        actual_dist = dist_to(robot_pos, target)
        reward = -actual_dist if actual_dist > DIST_TOLERANCE else 10
        return reward, actual_dist

    def _angle_reward(self):
        robot_angle = self.frame.robots_blue[0].theta
        target = self.target_angle
        angle_diff = abs_smallest_angle_diff(robot_angle, target)
        angle_reward = -angle_diff / np.pi if angle_diff > ANGLE_TOLERANCE else 1
        return angle_reward, angle_diff

    def _speed_reward(self):
        action_speed_x = self.actual_action[3] * MAX_VELOCITY
        action_speed_y = self.actual_action[4] * MAX_VELOCITY
        action_speed = Point2D(x=action_speed_x, y=action_speed_y)
        target = self.target_velocity
        vel_error = dist_to(action_speed, target)
        reward = -vel_error if vel_error > self.SPEED_TOLERANCE else 0.1
        speed_reward = reward - self.last_speed_reward
        self.last_speed_reward = reward
        return speed_reward

    def _path_consistency_reward(self):
        actual_path = np.array(self.robot_path) 
        target_position = np.array(self.target_point) 
        start_position = np.array(self.start_position)

        num_points = len(actual_path)
        ideal_path = np.linspace(start_position, target_position, num_points)

        error = actual_path - ideal_path
        squared_error = np.square(error)
        mean_squared_error = np.mean(squared_error)
        rmse = np.sqrt(mean_squared_error)

        reward = -rmse * 1000
    
        return reward

    def draw_target(self, screen, transformer, point, angle, color):
        x, y = transformer(point.x, point.y)
        size = 0.04 * self.field_renderer.scale
        pygame.draw.circle(screen, color, (x, y), size, 2)
        pygame.draw.line(
            screen,
            COLORS["BLACK"],
            (x, y),
            (
                x + size * np.cos(angle),
                y + size * np.sin(angle),
            ),
            2,
        )

    def _render(self):
        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        super()._render()

        # kP, kI, kD = self._actions_to_PID(self.cur_action)
        # v, w = RSimVSSPID.get_velocities(self.rsim, kP, kI, kD)

        # Draw Target
        self.draw_target(self.window_surface, pos_transform, self.target_point, self.target_angle, COLORS["PINK"])

        target_position = np.array(self.target_point) 
        start_position = np.array(self.start_position)

        num_points = 20
        ideal_path = np.linspace(start_position, target_position, num_points)

        # Draw ideal path
        if len(ideal_path) > 1:

            my_path = [pos_transform(*p) for p in ideal_path[:-1]]
            for point in my_path:
                pygame.draw.circle(self.window_surface, COLORS["BLUE"], point, 2)
        my_path = [pos_transform(*p) for p in ideal_path]
        pygame.draw.lines(self.window_surface, COLORS["BLUE"], False, my_path, 1)

        # Draw Path
        if len(self.robot_path) > 1:

            my_path = [pos_transform(*p) for p in self.robot_path[:-1]]
            for point in my_path:
                pygame.draw.circle(self.window_surface, COLORS["RED"], point, 2)
        my_path = [pos_transform(*p) for p in self.robot_path]
        pygame.draw.lines(self.window_surface, COLORS["RED"], False, my_path, 1)
  
    def init_reward_shaping_dict(self):
        return {
            'goal_score': 0, 
            'move': 0, 
            'energy': 0, 
            'has_reached_target': 0, 
            'timeout': 0, 
            'angle_diff': 0
        }
    
    def navigation(self, kP: float, kI: float, kD: float):
        target = self.target_point
        robot = list(self.frame.robots_blue.values())[0]
        theta = np.deg2rad(robot.theta)
        robot_half_axis = self.field.rbt_radius

        robot_rear_angle = theta + math.pi
        rear = False

        delta_x = target.x - robot.x
        delta_y = target.y - robot.y

        desired_angle = math.atan2(delta_y, delta_x) if dist_to(robot, target) > robot_half_axis else self.target_angle
        angle_error = smallest_angle_diff(theta, desired_angle)
        rear_angle_error = smallest_angle_diff(robot_rear_angle, desired_angle)

        if abs(rear_angle_error) < abs(angle_error):
            angle_error = rear_angle_error
            rear = True

        MILIMETER_OFFSET = 1000
        pid_output = self.pid(angle_error, kP * MILIMETER_OFFSET, 0, 0)

        angular_velocity = np.clip(pid_output, -self.max_w, self.max_w)

        linear_velocity = self.calculate_linear_v() / robot_half_axis

        left_motor_speed, right_motor_speed = 0, 0

        if dist_to(robot, target) < DIST_TOLERANCE:
            linear_velocity = 0

        left_motor_speed = linear_velocity - (angular_velocity * robot_half_axis / 2)
        right_motor_speed = linear_velocity + (angular_velocity * robot_half_axis / 2)
        
        if rear:
            return -right_motor_speed, -left_motor_speed

        return left_motor_speed, right_motor_speed
    
    def calculate_linear_v(self):
        robot = list(self.frame.robots_blue.values())[0]
        linear_velocity = 0
        vx = (robot.x - self.last_frame.robots_blue[0].x) / self.time_step if self.last_frame else 0
        vy = (robot.y - self.last_frame.robots_blue[0].y) / self.time_step if self.last_frame else 0

        robot_v_length = math.sqrt(vx ** 2 + vy ** 2)

        acc = 1000
        linear_velocity = robot_v_length + acc * self.time_step

        return np.clip(linear_velocity, 0, self.max_v)

    def pid(self, angle_error, kP, kI, kD):
        if len(self.integral) == self.sample_time:
            self.accumulated_error -= self.integral[0]
            self.integral.pop(0)

        self.integral.append(angle_error)

        self.accumulated_error += angle_error

        pid_output = kP * angle_error + kI * self.accumulated_error + kD * (angle_error - self.last_error)

        self.last_error = angle_error

        return pid_output