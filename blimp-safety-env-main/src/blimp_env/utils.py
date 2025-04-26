import random
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.ppo import MlpPolicy


_log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR
}


def save_image(img: np.ndarray, filename):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_logger(name: str, level: str = 'info') -> logging.Logger:
    # Configure the logger
    level_val = _log_levels.get(level.lower(), logging.INFO)
    logging.basicConfig(level=level_val,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger(name)
    return logger


class DummyPolicy(MlpPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=torch.nn.Tanh,
                 *args, **kwargs):
        super(DummyPolicy, self).__init__(observation_space, action_space,
                                          lr_schedule, net_arch, activation_fn,
                                          *args, **kwargs)

    def forward(self, obs, deterministic=False):
        # Assuming discrete action space
        num_actions = self.action_space.n
        # Output random actions
        actions = torch.tensor(np.random.randint(0, num_actions, size=(len(obs),)), dtype=torch.long)
        # Dummy values and log probs
        values = torch.zeros((len(obs), 1))
        log_probs = torch.zeros((len(obs), 1))
        return actions, values, log_probs

    def _predict(self, observation, deterministic=False):
        # Assuming discrete action space
        num_actions = self.action_space.n
        # Output random actions
        actions = np.random.randint(0, num_actions, size=len(observation))
        return actions, None

    def evaluate_actions(self, obs, actions):
        # Dummy evaluation, returning zeros for values, action log prob and entropy
        return torch.zeros(len(obs)), torch.zeros(len(obs)), torch.zeros(len(obs))


def autoregressive_infinite_generator():
    """
    Yields integers in the range 0-5 indefinitely. Most of the time it chooses the previous integer,
    but occasionally switches to a new one in the valid range.
    """

    current_value = random.randint(0, 5)
    while True:
        # Most of the time, yield the current value
        if random.random() < 0.9995:  # 80% chance to keep the same number
            yield current_value
        else:
            # Occasionally, switch to a new number in the range 0-5
            current_value = random.randint(0, 5)

            # Let's make going forward most likely
            rv = random.random()
            if rv < 0.5:
                current_value = 1
            elif rv < 0.75:
                current_value = 2
            elif rv < 0.875:
                current_value = 3
            elif rv < 0.9375:
                current_value = 4
            else:
                current_value = 5

            yield current_value
