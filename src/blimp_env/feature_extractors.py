from typing import Dict

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DinoV2Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.spcace.Box | gym.spaces.Dict) The observation space of the environment;
                               if the observation space is a Dict, it **must** contain an 'img' key)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
            self,
            observation_space: spaces.Box | spaces.Dict,
            features_dim: int = 384,
            dino_model: str = 'dinov2_vits14',
            frozen: bool = True
    ):
        super().__init__(observation_space, features_dim)

        # Load and freeze the DinoV2 model
        self.dino_model = torch.hub.load('facebookresearch/dinov2', dino_model)
        if frozen:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            self.dino_model.eval()

        observation_space = self._handle_obs_space(observation_space)

        assert observation_space.shape[0] % 3 == 0, "Observation space should be a stack of 3 channels"
        self.num_stacks = observation_space.shape[0] // 3
        self.norm_dino_layer = torch.nn.LayerNorm(self.dino_model.num_features)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.dino_model.num_features * self.num_stacks, features_dim),
            torch.nn.ReLU()
        )

    def _handle_obs_space(self, observation_space: spaces.Box | spaces.Dict) -> spaces.Box:
        if isinstance(observation_space, spaces.Dict):
            assert 'img' in observation_space.spaces, "If using a Dict observation space, it must contain an 'img' key"
            observation_space = observation_space['img']
        return observation_space

    def _handle_obs(self, observations: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(observations, dict):
            assert 'img' in observations, "If using a Dict observation space, it must contain an 'img' key"
            observations = observations['img']
        return observations

    def forward(self, observations: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        observations = self._handle_obs(observations)
        batch_d = observations.shape[0]
        destack_obs = observations.reshape((batch_d * self.num_stacks, 3, 224, 224))
        with torch.no_grad():
            features = self.dino_model.forward(destack_obs)
        norm_features = self.norm_dino_layer(features)
        norm_features = norm_features.reshape((batch_d, self.dino_model.num_features * self.num_stacks))
        norm_features = self.linear(norm_features)
        return norm_features
