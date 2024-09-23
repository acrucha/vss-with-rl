import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict, List

import gymnasium as gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import RSimVSS
from rsoccer_gym.vss.env_vss import VSSEnv
from rsoccer_gym.Utils import KDTree

N_ROBOTS_BLUE = 1
N_ROBOTS_YELLOW = 1
MAX_DIST_TO_TARGET = 0.5
FIELD_WIDTH = 1.3
FIELD_LENGTH = 1.5
ROBOT_WIDTH = 0.008
REWARD_NORM_BOUND = 5.0
TO_CENTIMETERS = 100
OBSERVATIONS_SIZE = 18
OUTPUT_SIZE = 2
ROBOT_HALF_AXIS = 37.5 / 1000
WHEEL_RADIUS = 24.0
MAX_V_WHEEL = 50.0
MAX_V_LINEAR = 1.2
MAX_W_ANGULAR = 1.5

class VSSPenaltyEnv(VSSEnv):
    """This environment controls a single robot to go to a target position using PID control. 

        Description:
        Observation:
            Type: Box(17)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               Ball X
            1               Ball Y
            2               Ball Vx
            3               Ball Vy
            4               Shooter X
            5               Shooter Y
            6               Shooter sin(theta)
            7               Shooter cos(theta)
            8               Shooter Vx
            9               Shooter Vy
            10              Shooter v_theta
            11              Gk X
            12              Gk Y
            13              Gk sin(theta)
            14              Gk cos(theta)
            15              Gk Vx
            16              Gk Vy
            17              Gk v_theta
        Actions:
            Type: Box(2, )
            Num             Action
            0               Shooter v (linear velocity) 
            1               Shooter w (angular velocity)
        Reward:
            Sum of Rewards:
                Goal
                Move to Target
                Energy Penalty
                Angle Difference between Robot and Target
        Starting State:
            Shooter on randomized penalty position, randomized Gk in the goal area and ball on penalty mark
        Episode Termination:
            Goal or Episode length is greater than 3.5 seconds
    """

    def __init__(self, render_mode=None, repeat_action=1, max_steps=500):
        super().__init__(render_mode=render_mode)

        self.n_robots_yellow = N_ROBOTS_YELLOW
        self.n_robots_blue = N_ROBOTS_BLUE

        self.rsim = RSimVSS(
            field_type=0,
            n_robots_blue=self.n_robots_blue,
            n_robots_yellow=self.n_robots_yellow,
            time_step_ms=int(self.time_step * 1000),
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(OBSERVATIONS_SIZE, ), dtype=np.float32)
        

        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_info = None
        self.v_wheel_deadzone = 0.05
        self.robot_half_axis = ROBOT_HALF_AXIS

        self.ou_actions = []
        for i in range(self.n_robots_blue + self.n_robots_yellow):
            self.ou_actions.append(
                OrnsteinUhlenbeckAction(self.action_space, dt=self.time_step)
            )
        
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
        self.reward_info = None
        self.previous_ball_potential = None
        self.steps = 0
        
        for ou in self.ou_actions:
            ou.reset()

        return obs, {}
    
    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        goal = self.field.goal_width / 2
        
        print(f"field_half_length = {field_half_length}; field_half_width = {field_half_width}")

        def gk_x(): return random.uniform(field_half_length - self.robot_half_axis, field_half_length)

        def gk_y(): return random.uniform(-goal, goal)

        def x(): return random.uniform(field_half_length/4, field_half_length/3)

        def y(): return random.uniform(-field_half_width/3, field_half_width/3)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=field_half_length/2, y=0)

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        # pos = (field_half_width - ROBOT_HALF_AXIS - 100, 0)
        pos = (gk_x(), gk_y())
        places.insert(pos)
        pos_frame.robots_yellow[0] = Robot(x=pos[0], y=pos[1], theta=theta())
        
        pos = (x(), y())
        places.insert(pos)
        pos_frame.robots_blue[0] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame
    
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

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

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

        observation.append(self.norm_pos(self.frame.robots_yellow[0].x))
        observation.append(self.norm_pos(self.frame.robots_yellow[0].y))
        observation.append(
            np.sin(np.deg2rad(self.frame.robots_yellow[0].theta))
        )
        observation.append(
            np.cos(np.deg2rad(self.frame.robots_yellow[0].theta))
        )
        observation.append(self.norm_v(self.frame.robots_yellow[0].v_x))
        observation.append(self.norm_v(self.frame.robots_yellow[0].v_y))
        observation.append(self.norm_w( self.frame.robots_yellow[0].v_theta))

        return np.array(observation, dtype=np.float32)


    def _actions_to_v_wheels(self, left_wheel_speed, right_wheel_speed):
        left_wheel_speed *= self.max_v
        right_wheel_speed *= self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed, right_wheel_speed


    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0], actions[1])
        commands.append(Robot(yellow=False, id=0, v_wheel0=v_wheel0, v_wheel1=v_wheel1))

        # print(f"action rad/s: v_wheel0 = {v_wheel0}; v_wheel1 = {v_wheel1}")

        # Send random commands to the GK
        actions = self.ou_actions[0].sample()
        v_wheel0, v_wheel1 = self._actions_to_v_wheels(actions[0], actions[1])
        commands.append(Robot(yellow=True, id=0, v_wheel0=v_wheel0,
                                v_wheel1=v_wheel1))

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = 0.6
        w_ball_grad = 3
        w_energy = 0.0184
        w_goal = 10
        w_gk = 0.1
        if self.reward_info is None:
            self.reward_info = {
                'goal_score': 0, 
                'move': 0,
                'ball_grad': 0, 
                'energy': 0,
                'gk': 0,
                'reward_total': 0
            }

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_info['goal_score'] += 1
            reward = 1 * w_goal
            goal = True
        elif self.steps == self.max_steps - 1:
            reward = -1 * w_goal
            goal = False
        else:
            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()
                # Calculate Move ball   
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()
                # Calculate Goalkeeper reaches ball
                gk_reaches_ball = self.__gk_reaches_ball()

                reward = w_move * move_reward + \
                    w_ball_grad * grad_ball_potential + \
                    w_energy * energy_penalty + \
                    w_gk * gk_reaches_ball

                self.reward_info['move'] += w_move * move_reward
                self.reward_info['ball_grad'] += w_ball_grad * grad_ball_potential
                self.reward_info['energy'] += w_energy * energy_penalty
                self.reward_info['gk'] += w_gk * gk_reaches_ball

        self.reward_info['reward_total'] += reward

        return reward, goal

    def __move_reward(self):
        """Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        """

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x, self.frame.robots_blue[0].y])
        robot_vel = np.array(
            [self.frame.robots_blue[0].v_x, self.frame.robots_blue[0].v_y]
        )
        robot_ball = ball - robot
        robot_ball = robot_ball / np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        return move_reward / 1.2

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty / 92
    
    def __ball_grad(self):
        """Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        """
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -np.sqrt(dx_a**2 + 2 * dy**2)
        dist_2 = np.sqrt(dx_d**2 + 2 * dy**2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            grad_ball_potential = (
                ball_potential - self.previous_ball_potential
            ) / self.time_step

        self.previous_ball_potential = ball_potential

        return grad_ball_potential / 0.8
        
    def __gk_reaches_ball(self):
        '''Calculates the reward when the gk reaches the ball'''
        gk = self.frame.robots_yellow[0]
        ball = self.frame.ball
        dist = math.sqrt((gk.x - ball.x)**2 + (gk.y - ball.y)**2) * TO_CENTIMETERS
        if dist <= (ROBOT_HALF_AXIS*2):
            norm = REWARD_NORM_BOUND/2 # max value of the gk reward is REWARD_NORM_BOUND/2
            dist = dist*norm/(ROBOT_HALF_AXIS*2)
            return -dist
        else:
            return 0
    