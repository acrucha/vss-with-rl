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

from envs.extra.RobotWithPID import RobotWithPID
from envs.extra.RSimVSSPID import RSimVSSPID
from collections import namedtuple
from rsoccer_gym.Render import COLORS
from envs.extra.Target import Target

import pygame

from torchrl.data import UnboundedContinuousTensorSpec

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

    def __init__(self, render_mode=None, repeat_action=1, max_steps=1200):
        super().__init__(field_type=0, n_robots_blue=N_ROBOTS_BLUE, n_robots_yellow=N_ROBOTS_YELLOW,
                         time_step=0.025, render_mode=render_mode)
        
        self.rsim = RSimVSSPID(
            field_type=0,
            n_robots_blue=N_ROBOTS_BLUE,
            n_robots_yellow=N_ROBOTS_YELLOW,
            time_step_ms=int(self.time_step * 1000),
            target=Robot(x=0, y=0, theta=0)
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

        self.max_kP = 5
        self.max_kD = 20
        self.max_kI = 0.3

        self.last_speed_reward = 0

        self.target = Robot(x=0, y=0, theta=0)

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        self.action_spec = UnboundedContinuousTensorSpec(1)
        self.cur_action = []
        self.all_actions = []

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
        }

        self.max_steps = max_steps

        print('Environment initialized')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.actions = None
        self.reward_info = None
        self.previous_target_potential = None
        self.steps = 0
        self.target = Robot(x=random.uniform(-FIELD_LENGTH/2 + ROBOT_WIDTH, FIELD_LENGTH/2 - ROBOT_WIDTH),
                            y=random.uniform(-FIELD_WIDTH/2 + ROBOT_WIDTH, FIELD_WIDTH/2 - ROBOT_WIDTH),
                            theta=random.uniform(0, 2*np.pi))
        for ou in self.ou_actions:
            ou.reset()

        return super().reset()

    def step(self, action):
        self.cur_action = action
        for _ in range(self.repeat_action):
            self.steps += 1
            # Join agent action with environment actions
            commands: List[RobotWithPID] = self._get_commands(action)
            
            # Send command to simulator
            self.rsim.set_target(self.target)
            self.rsim.send_commands(commands)
            self.sent_commands = commands

            # Get Frame from simulator
            self.last_frame = self.frame
            self.frame = self.rsim.get_frame()

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

        observation.append(self.norm_pos(self.target.x))
        observation.append(self.norm_pos(self.target.y))
        observation.append(
            np.sin(np.deg2rad(self.target.theta))
        )
        observation.append(
            np.cos(np.deg2rad(self.target.theta))
        )
            

        observation.append(self.norm_pos(self.frame.robots_blue[0].x))
        observation.append(self.norm_pos(self.frame.robots_blue[0].y))
        observation.append(
            np.sin(np.deg2rad(self.frame.robots_blue[0].theta))
        )
        observation.append(
            np.cos(np.deg2rad(self.frame.robots_blue[0].theta))
        )
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
        # print(f"kP={kP:.2f}, kI={kI:.2f}, kD={kD:.2f}")
        commands.append(RobotWithPID(yellow=False, id=0, kP=kP, kI=0, kD=0))

        self.all_actions.append(kP)

        return commands

    # def _calculate_reward_and_done(self):
    #     reward = 0
    #     has_reached_target = False
    #     w_move = 0.8
    #     w_angle_diff = 0.1
    #     w_energy = 0.1
    #     robot = self.frame.robots_blue[0]
    #     if self.reward_info is None:
    #         self.reward_info = self.init_reward_shaping_dict()

    #     dist_to_target = math.dist([self.target.x, self.target.y], [robot.x, robot.y])
    #     dist_to_target *= 100
                
    #     # print(f"Distance to target: {dist_to_target:.2f}cm")

    #     angle_diff = self.__angle_diff()
    #     move_reward = self.__move_reward()
    #     energy_penalty = self.__energy_penalty()

    #     if dist_to_target < MAX_DIST_TO_TARGET and abs(angle_diff) < 5:
            
    #         print(f"Reached target in {self.steps} steps")
    #         reward = np.clip(1/self.steps, -5, 5) + 1000

    #         self.reward_info['has_reached_target'] += reward
            
    #         has_reached_target = True
    #     else:
    #         if self.last_frame is not None:                
    #             if dist_to_target > MAX_DIST_ANGLE_REWARD:
    #                 w_move = 0.9
    #                 w_energy = 0.1
    #                 w_angle_diff = 0

    #             reward = w_move * move_reward + \
    #                 w_energy * energy_penalty + \
    #                 w_angle_diff * angle_diff

    #             self.reward_info['move'] += w_move * move_reward
    #             self.reward_info['energy'] += w_energy \
    #                 * energy_penalty
    #             self.reward_info['angle_diff'] += w_angle_diff \
    #                 * angle_diff

    #     return reward, has_reached_target

    def _calculate_reward_and_done(self):
        done = False
        reward = 0
        dist_reward, distance = self._dist_reward()
        angle_reward, angle_diff = self._angle_reward()
        # robot_vel_error = np.linalg.norm(
        #     np.array(
        #         [
        #             self.frame.robots_blue[0].v_x - self.target_velocity.x,
        #             self.frame.robots_blue[0].v_y - self.target_velocity.y,
        #         ]
        #     )
        # )
        # print(f"dist_reward: {distance < DIST_TOLERANCE} | robot_dist: {robot_dist < DIST_TOLERANCE} | angle: {angle_reward < ANGLE_TOLERANCE} | vel: {robot_vel_error < self.SPEED_TOLERANCE}")
        if (
            distance < DIST_TOLERANCE
            and angle_diff < ANGLE_TOLERANCE
            # and robot_vel_error < self.SPEED_TOLERANCE
        ):
            done = True
            reward = 1000
            self.reward_info["reward_objective"] += reward
            print(colorize("GOAL!", "green", bold=True, highlight=True))
        else:
            reward = dist_reward + angle_reward

        if done or self.steps >= self.max_steps:
            # pairwise distance between all actions
            action_var = np.linalg.norm(
                np.array(self.all_actions[1:]) - np.array(self.all_actions[:-1])
            )
            self.reward_info["reward_action_var"] = action_var

        self.reward_info["reward_dist"] += dist_reward
        self.reward_info["reward_angle"] += angle_reward
        self.reward_info["reward_total"] += reward
        self.reward_info["reward_steps"] = self.steps

        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        self.reward_info = {
            "reward_dist": 0,
            "reward_angle": 0,
            "reward_action_var": 0,
            "reward_objective": 0,
            "reward_total": 0,
            "reward_steps": 0,
        }
        self.robot_path = [(pos_frame.robots_blue[0].x, pos_frame.robots_blue[0].y)] * 2

        return pos_frame

    def _actions_to_PID(self, actions):
        kP = np.clip(abs(actions[0]), 0, self.max_kP)
        kI = np.clip(abs(actions[1]), 0, self.max_kI)
        kD = np.clip(abs(actions[2]), 0, self.max_kD)

        return kP , kI , kD

    def __move_reward(self):
        '''Calculate Move to target reward

        Cosine between the robot vel vector and the vector robot -> target.
        This indicates rather the robot is moving towards the target or not.
        '''

        target = np.array([self.target.x, self.target.y])
        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_x,
                              self.frame.robots_blue[0].v_y])
        robot_target = target - robot
        robot_target = robot_target/np.linalg.norm(robot_target)

        move_reward = np.dot(robot_target, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -REWARD_NORM_BOUND, REWARD_NORM_BOUND)
        return move_reward

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty
    
    def __angle_diff(self):
        '''Calculates the angle difference between the robot and the target'''
        angle_diff = RSimVSSPID.smallest_angle_diff(self.frame.robots_blue[0].theta, self.target.theta) 
        return np.clip(angle_diff, -REWARD_NORM_BOUND, REWARD_NORM_BOUND)
    
    def _dist_reward(self):
        robot = self.frame.robots_blue[0]
        robot = Point2D(x=robot.x, y=robot.y)
        actual_dist = np.linalg.norm([robot.x - self.target.x, robot.y - self.target.y]) * TO_CENTIMETERS
        reward = -actual_dist if actual_dist > DIST_TOLERANCE else 10
        return reward, actual_dist

    def _angle_reward(self):
        robot = self.frame.robots_blue[0]
        robot_angle = np.deg2rad(robot.theta)
        target =  np.deg2rad(self.target.theta)
        angle_diff = RSimVSSPID.smallest_angle_diff(robot_angle, target)
        angle_reward = -angle_diff / np.pi if angle_diff > ANGLE_TOLERANCE else 1
        return angle_reward, angle_diff

    # def _speed_reward(self):
    #     action_speed_x = self.cur_action[0] * MAX_VELOCITY
    #     action_speed_y = self.cur_action[1] * MAX_VELOCITY
    #     action_speed = Point2D(x=action_speed_x, y=action_speed_y)
    #     target = self.target_velocity
    #     vel_error = RSimVSSPID.smallest_angle_diff(action_speed, target)
    #     reward = -vel_error if vel_error > self.SPEED_TOLERANCE else 0.1
    #     speed_reward = reward - self.last_speed_reward
    #     self.last_speed_reward = reward
    #     return speed_reward

    def draw_target(self, transformer, point, angle):
        x, y = transformer(point.x, point.y)
        rbt = Target(
            x,
            y,
            angle,
            self.field_renderer.scale
        )
        rbt.draw(self.window_surface)

    def _render(self):
        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        super()._render()
        self.draw_target(
            pos_transform,
            self.target,
            self.target.theta
        )

        # kP, kI, kD = self._actions_to_PID(self.cur_action)
        # v, w = RSimVSSPID.get_velocities(self.rsim, kP, kI, kD)

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