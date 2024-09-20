import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict, List

import gymnasium as gym
import numpy as np
from rsoccer_gym.Entities import Robot, Frame, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree

N_ROBOTS_BLUE = 3
N_ROBOTS_YELLOW = 3
MAX_DIST_TO_TARGET = 0.5
FIELD_WIDTH = 1.3
FIELD_LENGTH = 1.5
ROBOT_WIDTH = 0.008
REWARD_NORM_BOUND = 5.0
TO_CENTIMETERS = 100
OBSERVATIONS_SIZE = 46
OUTPUT_SIZE = 2
ROBOT_HALF_AXIS = 37.5
WHEEL_RADIUS = 24.0
MAX_V_WHEEL = 85
MAX_V_LINEAR = 1.5
MAX_W_ANGULAR = MAX_V_LINEAR

class VSSAttackerEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match


    Description:
    Observation:
        Type: Box(46)
        Normalized Bounds to [-1.5, 1.5]
        Num             Observation normalized
        0               Ball X
        1               Ball Y
        2               Ball Vx
        3               Ball Vy
        4 + (7 * i)     id i Blue Robot X
        5 + (7 * i)     id i Blue Robot Y
        6 + (7 * i)     id i Blue Robot sin(theta)
        7 + (7 * i)     id i Blue Robot cos(theta)
        8 + (7 * i)     id i Blue Robot Vx
        9  + (7 * i)    id i Blue Robot Vy
        10 + (7 * i)    id i Blue Robot v_theta
        25 + (7 * i)    id i Yellow Robot X
        26 + (7 * i)    id i Yellow Robot Y
        27 + (7 * i)    id i Yellow Robot Vx
        28 + (7 * i)    id i Yellow Robot Vy
        29 + (7 * i)    id i Yellow Robot v_theta
        30 + (7 * i)    id i Yellow Robot sin(theta)
        31 + (7 * i)    id i Yellow Robot cos(theta)
    Actions:
        Type: Box(2, )
        Num     Action
        0       id 0 Blue Linear Velocity
        1       id 0 Blue Angular Velocity
    Reward:
        Sum of Rewards:
            Goal
            Ball Potential Gradient
            Move to Ball
            Energy Penalty
    Starting State:
        Randomized Robots and Ball initial Position
    Episode Termination:
        5 minutes match time
    """

    def __init__(self, render_mode=None, repeat_action=1, max_steps=500):
        super().__init__(field_type=0, n_robots_blue=N_ROBOTS_BLUE, n_robots_yellow=N_ROBOTS_YELLOW,
                         time_step=0.025, render_mode=render_mode)
        
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        
        self.NORM_BOUNDS = MAX_V_LINEAR

        self.observation_space = gym.spaces.Box(
            low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS, shape=(OBSERVATIONS_SIZE,), dtype=np.float32
        )

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_info = None
        self.v_wheel_deadzone = 0.05
        self.robot_half_axis = ROBOT_HALF_AXIS

        self.max_steps = max_steps
        self.repeat_action = repeat_action

        FPS = 120
        if self.repeat_action > 1:
            FPS = np.ceil((1200 // self.repeat_action) / 10)

        self.metadata["render_fps"] = FPS

        print('Environment initialized')

    def reset(self, *, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)
        self.actions = None
        self.reward_shaping_total = None
        self.previous_ball_potential = None

        for ou in self.ou_actions:
            ou.reset()
            
        return obs, {}
    
    def step(self, action):
        self.cur_action = action
        for _ in range(self.repeat_action):
            self.steps += 1
            # Join agent action with environment actions
            commands: List[Robot] = self._get_commands(action)
            
            # Send command to simulator
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

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, self.reward_info

    def _actions_to_v_wheels(self, left_motor_speed, right_motor_speed):

        # left_motor_speed, right_motor_speed = 0, 0

        # linear_velocity *= MAX_V_LINEAR
        # angular_velocity *= MAX_W_ANGULAR

        # linear_velocity = np.clip(linear_velocity, -MAX_V_LINEAR, MAX_V_LINEAR)

        # angular_velocity = np.clip(angular_velocity, -MAX_W_ANGULAR, MAX_W_ANGULAR)

        # print("Linear Velocity: ", linear_velocity)
        # print("Angular Velocity: ", angular_velocity)
        
        # left_motor_speed = linear_velocity - (angular_velocity * WHEEL_RADIUS)
        # right_motor_speed = linear_velocity + (angular_velocity * WHEEL_RADIUS)

        left_motor_speed *= MAX_V_WHEEL
        right_motor_speed *= MAX_V_WHEEL

        left_motor_speed, right_motor_speed = np.clip(
            (left_motor_speed, right_motor_speed), -MAX_V_WHEEL, MAX_V_WHEEL
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_motor_speed < self.v_wheel_deadzone:
            left_motor_speed = 0

        if -self.v_wheel_deadzone < right_motor_speed < self.v_wheel_deadzone:
            right_motor_speed = 0

        left_motor_speed /= WHEEL_RADIUS
        right_motor_speed /= WHEEL_RADIUS

        return left_motor_speed , right_motor_speed

    def _frame_to_observations(self):
        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(np.sin(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(np.cos(np.deg2rad(self.frame.robots_blue[i].theta)))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(np.sin(np.deg2rad(self.frame.robots_yellow[i].theta)))
            observation.append(np.cos(np.deg2rad(self.frame.robots_yellow[i].theta)))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0], actions[1])
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        # Send random commands to the other robots
        for i in range(1, self.n_robots_blue):
            actions = self.ou_actions[i].sample()
            self.actions[i] = actions
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0], actions[1])
            commands.append(
                Robot(yellow=False, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )
        for i in range(self.n_robots_yellow):
            actions = self.ou_actions[self.n_robots_blue + i].sample()
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0], actions[1])
            commands.append(
                Robot(yellow=True, id=i, v_wheel0=v_wheel0, v_wheel1=v_wheel1)
            )

        return commands

    def _get_initial_positions_frame(self):
        """Returns the position of each robot and ball for the initial frame"""
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x():
            return random.uniform(-field_half_length + 0.1, field_half_length - 0.1)

        def y():
            return random.uniform(-field_half_width + 0.1, field_half_width - 0.1)

        def theta():
            return random.uniform(0, 360)

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

        return pos_frame

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = 0.2
        w_ball_grad = 0.6
        w_energy = 2e-4
        if self.reward_info is None:
            self.reward_info = {
                'goal_score': 0, 
                'move': 0,
                'ball_grad': 0, 
                'energy': 0,
                'reward_total': 0
            }

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_info['goal_score'] += 1
            reward = 100
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total["goal_score"] -= 1
            reward = -100
            goal = True
        elif self.steps == self.max_steps - 1:
            reward = -10
            goal = False
        else:
            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()
                # Calculate Move ball   
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()

                reward = w_move * move_reward + \
                    w_ball_grad * grad_ball_potential + \
                    w_energy * energy_penalty

                self.reward_info['move'] += w_move * move_reward
                self.reward_info['ball_grad'] += w_ball_grad \
                    * grad_ball_potential
                self.reward_info['energy'] += w_energy \
                    * energy_penalty

        self.reward_info['reward_total'] += reward

        return reward, goal

    def __move_reward(self):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_x,
                              self.frame.robots_blue[0].v_y])
        robot_ball = ball - robot
        robot_ball = robot_ball/np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty
    
    def __ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0)\
            + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential