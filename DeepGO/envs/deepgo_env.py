import gym
from agent import Agent
import cv2
import numpy as np
from DeepGO.envs.frame_convert import frame


class DeepGOEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DeepGOEnv, self).__init__() 
        self.agent = Agent()
        self.agent.start_connection()

        self.input_width = 512
        self.input_height = 512

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.input_height, self.input_width, 3), dtype=np.uint8)
        
        self.frame = frame(self.agent.get_state().get_frame())

    def step(self, action):
        #screen = self.get_frame()
        screen = self._get_obs()
        reward = 0
        done = False
        #print(screen, reward, done)
        return screen, reward, done, {}

    def reset(self):
        #return self.get_frame()
        return self._get_obs()
    
    def render(self, mode='human'):
        cv2.imshow('DeepGO', self._get_obs())
        cv2.waitKey(1)
        
    def _get_obs(self):
        return frame(self.agent.get_state().get_frame())