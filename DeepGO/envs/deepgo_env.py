import gym
from agent import Agent
import cv2
import numpy as np
from DeepGO.envs.frame_convert import frame
from DeepGO.envs.reward_calculator import RewardCalculator


class DeepGOEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DeepGOEnv, self).__init__() 
        self.agent = Agent()
        self.agent.start_connection()

        self.input_width = 256
        self.input_height = 144

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.input_width, self.input_height, 3), dtype=np.uint8)
        
        self.frame = self._get_obs()
        self.reward_calc = RewardCalculator()

        self.agent_state = self.agent.get_state()
        self.previous_agent_state = self.agent_state

    def step(self, action):
        #screen = self.get_frame()
        self.agent_state = self.agent.get_state()
        screen = self._get_obs()
        
        reward = self.reward_calc.calculate_reward(self.agent_state, self.previous_agent_state)

        agent_action = self._convert_action(action)
        #self.agent.send_input(agent_action)

        done = False
        info = {}

        self.previous_agent_state = self.agent_state

        return screen, reward, done, info

    def reset(self):
        #return self.get_frame()
        self.agent_state = self.agent.get_state()
        return np.zeros((256, 144, 3), dtype=np.uint8) #self._get_obs()
    
    def render(self, mode='human'):
        cv2.imshow('DeepGO', self._get_obs())
        cv2.waitKey(1)
        
    def _get_obs(self):
        return frame(self.agent.get_state().get_frame())

    def _convert_action(self, action):
        converted = []
        for a in range(5):
            if action[a] > 0:
                press = 1
            else:
                press = 0
            converted.append(press)
            
        converted.append(action[5])
        converted.append(action[6])

        return converted