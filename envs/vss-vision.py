from typing import Dict
import gym
import numpy as np
import pygame
import torch
import vss_vision
import cv2

from typing import Dict

import gymnasium as gym

import numpy as np

from torchrl.data import UnboundedContinuousTensorSpec

import torchvision.transforms as transforms

from utils.utils import filter_all_colors


ACTION_SIZE = 8
IMG_SIZE = (640, 480)
MAB_IMGS= 'images/mab/'
COLORS_ORDER = np.array(range(1, ACTION_SIZE+1))
CHANNELS = 3 # RGB

class VSSVisionEnv(gym.Env):

    """This environment auto-calibrate the Vss-Vision params.

        Description:
        Observation:
            Type: Box(9)
            Num             Observation normalized  
            0               Orange
            1               Blue
            2               Yellow
            3               Red
            4               Green
            5               Pink
            6               Cyan
            7               Threshold
            8               Segmentation Output (Image)            
        Actions:
            Type: Box(8, )
            Num             Action
            0               Orange
            1               Blue
            2               Yellow
            3               Red
            4               Green
            5               Pink
            6               Cyan
            7               Threshold
        Reward:
            Sum of Rewards:
                MSE
                Segmentation Precision
        Starting State:
            Randomized Params
        Episode Termination:
            Reach 99.5% of the answer image
    """

    def __init__(self, w=640, h=480, repeat_action=1, max_steps=1200, render_mode="human"):
        super().__init__()
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "render_modes": ["human", "rgb_array"],
            "render_fps": 60,
            "render.fps": 60,
        }   
        
        self.action_space = gym.spaces.Box(low=0, high=255, shape=(ACTION_SIZE, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, h, w, 3), dtype=np.float32)
        self.actions: Dict = None

        self.action_spec = UnboundedContinuousTensorSpec(1)
        self.cur_action = []
        self.all_actions = []

        self.robot_path = []
        self.repeat_action = repeat_action
        FPS = 120
        if self.repeat_action > 1:
            FPS = np.ceil((1200 // self.repeat_action) / 10)

        self.metadata["render_fps"] = FPS

        self.image_w = w
        self.image_h = h

        self.reward_info = {
            "reward_mse": 0,
            "reward_sp": 0,
            "reward_action_var": 0,
            "reward_total": 0,
        }

        self.max_steps = max_steps

        self.image = None
        self.answer = None

        # Render
        self.render_mode = render_mode
        self.window_surface = None
        self.window_size = (self.image_w * 2, self.image_h)
        self.clock = None

        print('Environment initialized')

    def step(self, action):
        self.steps += 1
        self.action = action

        output = self.get_vss_vision_output(self.answer, self.image)

        # Calculate environment observation, reward and done condition
        observation = self._frame_to_observations()
        reward, done = self._calculate_reward_and_done()

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, self.reward_info

    def get_vss_vision_output(self, answer, image):
        self.answer = answer
        self.answer = cv2.resize(self.answer, IMG_SIZE)
        
        self.src_image = image
        self.src_image = cv2.resize(self.src_image, IMG_SIZE)

        output = vss_vision.run_seg(self.src_image, np.array(self.action), COLORS_ORDER)
        output = cv2.resize(output, (self.image_w, self.image_h))

        self.output = output

        return output

    def _frame_to_observations(self):

        ## Output
        image = cv2.cvtColor(self.output, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        img_tensor = transform(image)

        ## Answer
        ans = cv2.cvtColor(self.answer, cv2.COLOR_BGR2RGB)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        ans_tensor = transform(ans)

        obs_tensors = torch.cat([ans_tensor, img_tensor], dim=1)

        obs_tensors = torch.clamp(obs_tensors, 0, 1)

        return torch.asarray(obs_tensors)

    def _get_segmentation_output(self, action):
        commands = []
        self.actions = {}

        self.actions[0] = action

        self.all_actions.append(action)

        return commands
    
    def _calculate_reward_and_done(self):
        done = False

        mse_reward, sp_reward = self.calculate_rewards()
        reward = (mse_reward + sp_reward) * 1e5

        if done or self.steps >= self.max_steps:
            # pairwise distance between all actions
            action_var = np.linalg.norm(
                np.array(self.all_actions[1:]) - np.array(self.all_actions[:-1])
            )
            self.reward_info["reward_action_var"] = action_var

        self.reward_info["reward_mse"] += mse_reward
        self.reward_info["reward_sp"] += sp_reward
        self.reward_info["reward_total"] += reward

        print(f"Reward: {reward}")

        return reward, done

    def rgb_to_binary_img(self, output):
        filtered_output = output
       
        image_gray = cv2.cvtColor(self.answer, cv2.COLOR_BGR2GRAY)
        _, ans = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)
        image_gray = cv2.cvtColor(filtered_output, cv2.COLOR_BGR2GRAY)
        _, out = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)
        return ans, out

    def mse_reward(self, ans, out):

        MAX_ERROR = 244800.0 * 2  # difference between a white image and a black image

        ans_white = ans != 0
        ans_black = ans == 0
        out_white = out != 0
        out_black = out == 0

        mse = np.linalg.norm(out_white ^ ans_white) + np.linalg.norm(out_black ^ ans_black)

        return 1 - (mse / MAX_ERROR)

    def segmentation_precision(self, ans, out):
        should_be_black = ans == 0
        should_be_white = ans != 0

        # number of pixels that are white in the segmented image, but should be black
        painted_but_shouldnt     = out[should_be_black].sum()/float(255)
        # number of pixels that are black in the segmented image, but should be white
        not_painted_but_should   = (out[should_be_white] == 0).sum()

        # percentage of pixels that could be correctly segmented as black
        fpr = 1 - float(painted_but_shouldnt)/(should_be_black.sum()+should_be_white.sum())
        
        # percentage of pixels that could be correctly segmented as white
        fnr = 1 - float(not_painted_but_should)/(should_be_white.sum()+should_be_black.sum())

        reward = (fnr + fpr)/2
        
        return reward
    
    def calculate_rewards(self):
        filtered_colors_ans = filter_all_colors(self.answer)
        filtered_colors_out = filter_all_colors(self.output)

        mse_reward = 0
        sp_reward = 0

        for i in range(0, len(filtered_colors_ans)):
            if i == 0:
                # Threshold
                ans, out = self.rgb_to_binary_img(self.output)
            else:
                ans, out = self.rgb_to_binary_img(filtered_colors_out[i])

            mse_reward += self.mse_reward(ans, out)
            sp_reward += self.segmentation_precision(ans, out)

        mse_reward /= len(filtered_colors_ans)
        sp_reward /= len(filtered_colors_ans)

        return mse_reward, sp_reward
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.steps = 0  # Reset step counter
        self.all_actions = []  # Reset actions history

        self.action = self.action_space.sample()

        self.output = self.get_vss_vision_output(self.answer, self.image)
        
        obs = self._frame_to_observations()
        
        return obs, {}

    def set_image_and_answer(self, image, answer):
        self.image = image
        self.answer = answer

    def render(self) -> None:
        """
        Renders the answer image and the segmentation output.
        """

        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("VSS-Vision Environment")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        font = pygame.font.Font(None, 36)

        ans_img = cv2.cvtColor(self.answer, cv2.COLOR_BGR2RGB).transpose(1, 0, 2)
        answer_surface = pygame.surfarray.make_surface(ans_img)
        self.window_surface.blit(answer_surface, (0, 0))

        answer_text = font.render("ANSWER", True, (255, 255, 255)) 
        self.window_surface.blit(answer_text, (10, 10))

        output_img = cv2.cvtColor(self.output, cv2.COLOR_BGR2RGB).transpose(1, 0, 2)
        output_surface = pygame.surfarray.make_surface(output_img)
        self.window_surface.blit(output_surface, (self.image_w, 0))

        output_text = font.render("OUTPUT", True, (255, 255, 255)) 
        self.window_surface.blit(output_text, (self.image_w + 10, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )