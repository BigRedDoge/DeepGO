import gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
       
        self.device = select_device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weights = '/Users/sean/Documents/DeepGO/yolov5/weights/326_head_body.pt'
        
        self.yolo = DetectMultiBackend(weights=self.weights, device=self.device)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.yolo.warmup(imgsz=1).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: (torch.Tensor)
        :return: (torch.Tensor)
        """
        x = self.yolo(observations)
        return self.linear(x)