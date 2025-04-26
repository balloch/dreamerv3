from typing import Dict, Any

import math
import torch
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import ResizeObservation
from stable_baselines3.common.logger import Video
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(torch.ByteTensor([screens]), fps=30),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class ResizeDictWithImageObservation(ResizeObservation):

    def __init__(self, env: gym.Env, shape: tuple[int, int] | int, keys: str | list[str] = 'img'):
        # CHORE: shouldn't use try/except for control flow
        try:
            super(ResizeDictWithImageObservation, self).__init__(env=env, shape=shape)
        except AssertionError:
            # File "blimp_env/wrappers/observation.py", line 10, in __init__
            #     super(ResizeDictObservation, self).__init__(env=env, shape=shape)
            #   File ".../resize_observation.py", line 50, in __init__
            #     assert isinstance(
            # AssertionError: Expected the observation space to be Box, actual type: <class 'gymnasium.spaces.dict.Dict'>
            pass
        self.keys = [keys] if isinstance(keys, str) else keys

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """

        if not isinstance(observation, dict):
            raise ValueError(f'ResizeDictObservation requires observation to be a dict. Got: {type(observation)}')

        for k, v in observation.items():
            if k in self.keys:
                observation[k] = super().observation(v)

        return observation


def normalize_angle(theta: float) -> float:
    """ Normalize an angle theta (in radians) to [-pi, pi].

    Args:
        angle: The angle to normalize

    Returns:
        The normalized angle
    """
    return math.atan2(math.sin(theta), math.cos(theta))


class NormalizePoseWrapper(ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the observations by normalizing the pose to [-1, 1].

        Args:
            observation: The observation to normalize

        Returns:
            The normalized observations
        """
        if not isinstance(observation, dict):
            raise ValueError(f'ResizeDictObservation requires observation to be a dict. Got: {type(observation)}')

        observation = observation.copy()

        # Zero out the velocity / angular velocity components
        # TODO: Hardcoding for now since we're just trying to get a working example right now.
        #       In the simple example we're using velocity isn't needed to solve the task anyway.
        observation['pose'][:6] = 0.0

        # Normalize the pose
        x_max = max((abs(self.env.observation_space['pose'].low[6]), abs(self.env.observation_space['pose'].high[6])))
        y_max = max((abs(self.env.observation_space['pose'].low[7]), abs(self.env.observation_space['pose'].high[7])))
        z_max = max((abs(self.env.observation_space['pose'].low[8]), abs(self.env.observation_space['pose'].high[8])))
        observation['pose'][6] /= x_max
        observation['pose'][7] /= y_max
        observation['pose'][8] /= z_max

        # Normalize the orientation to [-1, 1]
        roll_norm = normalize_angle(np.deg2rad(observation['pose'][9])) / np.pi
        pitch_norm = normalize_angle(np.deg2rad(observation['pose'][10])) / np.pi
        yaw_norm = normalize_angle(np.deg2rad(observation['pose'][11])) / np.pi

        observation['pose'][9] = roll_norm
        observation['pose'][10] = pitch_norm
        observation['pose'][11] = yaw_norm

        # from tabulate import tabulate
        # print(tabulate(observation['pose'].reshape((1, -1)), tablefmt='fancy_grid'))

        return observation


class NormalizeSensorWrapper(ObservationWrapper):
    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs = obs.copy()
        for idx, value in enumerate(obs["sensor"]):
            max_val = max(
                (
                    abs(self.env.observation_space["sensor"].low[idx]),
                    abs(self.env.observation_space["sensor"].high[idx])
                )
            )
            obs["sensor"][idx] = value / max_val
        return obs
