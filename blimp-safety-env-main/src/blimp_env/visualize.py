from pathlib import Path
from datetime import datetime

import yaml
import click
import gymnasium as gym
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from blimp_env.wrappers import (
    ResizeDictWithImageObservation,
    NormalizePoseWrapper,
    NormalizeSensorWrapper,
)


def save_image(img: np.ndarray, filename):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


@click.command()
@click.option('--config-path', type=click.Path(exists=True), help='Path to YAML configuration file')
@click.option('--ckpt-path', default=None, type=click.Path(exists=True), help='Path to saved model')
def main(config_path, ckpt_path):
    """ Load a checkpoint and run the policy """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if ckpt_path:
        config.update({"ckpt_path": ckpt_path})

    logger.info(f"Running with args:\n{config}")

    model = PPO.load(config['ckpt_path'])

    video_length = 100000
    logdir = Path('logdir/test')
    logdir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    Env = gym.envs.registration.load_env_creator(config.get("environment", "blimp_env:IntoTheFireBasicNavEnv"))
    sensor_env = "FollowTask" in Env.__name__ or "ChaseTask" in Env.__name__
    reset_options = config.get('reset_options', {})
    if reset_options:
        Env.reset_options = reset_options
    img_size = (224, 224)
    env = Env(
        max_episode_steps=config['max_episode_steps'],
        dynamics=config['dynamics_mode'],
        render_mode='rgb_array',
        img_size=img_size,
        debug=config['debug'],
        disable_render=False,
        log_interval=config['log_interval'],
        action_repeat=config['action_repeat']
    )

    env = ResizeDictWithImageObservation(env, img_size)
    env = NormalizePoseWrapper(env)
    if sensor_env:
        env = NormalizeSensorWrapper(env)
    obs, _ = env.reset()

    video_recorder = VideoRecorder(env=env, path=f'{logdir}/video-{ts}.mp4')

    env.reset()
    for i in range(video_length + 1):
        # action = env.action_space.sample()
        obs.pop('img', None)
        action, _ = model.predict(obs)

        # TODO: query the RTA algorithm!
        # 1. Pass action to a ros2 node
        # 2. Wait for a ros2 subscription to show a non-null response
        # Note: this is not realistic for hardware.
        
        # Another option
        # 1. Query (sub) the previous RTA solution that is 1 step in the future
        #    relative to the current time. Use a cost function that perturbs
        #    two nominal actions from model.predict(obs) and feeding this through
        #    the physics engine action_repeat steps.
        # 2. Publish the new nominal and one-future nominal actions to trigger the RTA

        obs, reward, is_terminated, is_truncated, info = env.step(action)

        video_recorder.capture_frame()

        if is_terminated or is_truncated:
            break

    # Save the video
    env.close()
    video_recorder.close()


if __name__ == """__main__""":
    """ Create video of agent performing task using saved checkpoint 
    
    Example usage:
        unzip ./logdir/best_model.zip
        python -m blimp_env.visualize --config-path config/task2.yaml
    
    """
    main()
