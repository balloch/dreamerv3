from pathlib import Path
from argparse import Namespace

import yaml
import torch
import click
import gymnasium as gym
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from blimp_env.feature_extractors import DinoV2Extractor
from blimp_env.wrappers import (
    VideoRecorderCallback,
    ResizeDictWithImageObservation,
    NormalizePoseWrapper,
    NormalizeSensorWrapper,
)


def setup_policy(env, feature_extractor, feature_dim, hidden_dim, disable_render, logdir, device, verbose=1) -> PPO:
    """Make policy for training"""

    if feature_extractor == 'dino':
        # Setup policy with DINO v2 feature extractor
        logger.info("Using DINO v2 feature extractor")
        policy_kwargs = dict(
            features_extractor_class=DinoV2Extractor,
            features_extractor_kwargs=dict(features_dim=feature_dim),
            activation_fn=torch.nn.ReLU,
        )
        model = PPO(
            "MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=logdir,
            device=device,
        )
    elif feature_extractor == 'mlp':
        if not disable_render:
            raise ValueError("MLP policy requires --disable-render flag")
        # Policy with MLP feature extractor
        logger.info("Using MLP feature extractor (No image inputs)")

        policy_kwargs = dict(
            net_arch=dict(pi=[hidden_dim, feature_dim], vf=[hidden_dim, feature_dim]),
            activation_fn=torch.nn.ReLU,
        )

        model = PPO(
            "MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=logdir,
            device=device,
            ent_coef=0.0005,
            vf_coef=0.8,
            clip_range=0.1,
            learning_rate=5e-5,
            gamma=0.9999,
            n_steps=64*1000 #batch-size * factor
        )
    elif feature_extractor == 'cnn':
        # Policy with CNN feature extractor
        logger.info("Using CNN feature extractor")
        model = PPO("MultiInputPolicy", env=env, verbose=verbose, tensorboard_log=logdir, device=device)

    else:
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")

    return model


def setup_env(
    environment,
    reset_options,
    dynamics_mode,
    max_episode_steps,
    debug,
    disable_render,
    log_interval,
    do_env_check,
    img_size=224,
    reward_unit=1e-3,
    action_repeat=1,
) -> gym.Env:
    Env = gym.envs.registration.load_env_creator(environment)
    # TODO: Better way of detecting this needed
    sensor_env = 'FollowTask' in Env.__name__ or "ChaseTask" in Env.__name__

    Env.reset_options = reset_options
    env = Env(
        max_episode_steps=max_episode_steps,
        dynamics=dynamics_mode,
        render_mode='rgb_array' if not disable_render else None,
        img_size=(img_size, img_size),
        debug=debug,
        disable_render=disable_render,
        log_interval=log_interval,
        reward_unit=reward_unit,
        action_repeat=action_repeat,
    )

    env = ResizeDictWithImageObservation(env, img_size)
    env = NormalizePoseWrapper(env)

    if sensor_env:
        env = NormalizeSensorWrapper(env)

    # if disable_render:
    #     if sensor_env:
    #         env = gym.wrappers.TransformObservation(
    #             env,
    #             lambda obs: {'pose': obs['pose'], 'sensor': obs['sensor']}
    #         )
    #     else:
    #         env = gym.wrappers.TransformObservation(env, lambda obs: {'pose': obs['pose']})

    if do_env_check:
        if disable_render:
            logger.warning("Skipping environment check because --disable-render flag is not set")
            return env
        check_env(env, warn=True, skip_render_check=False)

    return env


def setup_callbacks(env, eval_freq, eval_episodes, disable_render, logdir, stochastic_eval):
    eval_callback = EvalCallback(
        env,
        best_model_save_path=logdir,
        log_path=logdir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=stochastic_eval,
        render=disable_render,
    )

    if disable_render:
        callback = CallbackList([eval_callback])
    else:
        video_callback = VideoRecorderCallback(env, render_freq=eval_freq, deterministic=stochastic_eval)
        callback = CallbackList([eval_callback, video_callback])

    return callback


def setup_logging(logdir):
    base_output_dir = Path(logdir)
    base_output_dir.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option('--environment', default="blimp_env:IntoTheFireBasicNavEnv", help='Environment Class')
# @click.option('--env_id', default="blimp_env/IntoTheFireBasicNav-v0", help='Environment ID')
@click.option('--max-train-steps', default=1e8, help='Maximum training steps', type=int)
@click.option('--max-episode-steps', default=10000, help='Maximum episode steps', type=int)
@click.option('--feature-extractor', type=click.Choice(['dino', 'cnn', 'mlp']), default='dino')
@click.option('--feature-dim', default=128, help='Feature dimension', type=int)
@click.option('--hidden-dim', default=128, help='Hidden dimension (applies to MLP policy only)', type=int)
@click.option('--dynamics-mode', type=click.Choice(['simple', 'physics']), default='physics')
@click.option('--disable-render', is_flag=True, default=False)
@click.option('--do-env-check', is_flag=True, default=False)
@click.option('--logdir', default="your_default_log_dir", help='Log directory')
@click.option('--log-interval', default=100, help='Log interval', type=int)
@click.option('--experiment-base-name', default='BLIMP_AGENT', help='Experiment base name')
@click.option('--eval-freq', default=1000, help='Evaluation frequency', type=int)
@click.option('--eval-episodes', default=3, help='Evaluation episodes', type=int)
@click.option('--debug', is_flag=True, default=False)
@click.option('--device', default='cuda:7', help='Device to use')
@click.option('--stochastic-eval', is_flag=True, default=False)
@click.option('--config', type=click.Path(exists=True), help='Path to YAML configuration file')
@click.option("--reward-unit", type=float, default=1e-3, help="Unit of rewards")
@click.option("--action-repeat", type=float, default=1, help="Number of sim-steps to repeat actions (10 recommended for physics)")
def main(
    environment,
    # env_id,
    max_train_steps,
    max_episode_steps,
    feature_extractor,
    feature_dim,
    hidden_dim,
    dynamics_mode,
    disable_render,
    do_env_check,
    logdir,
    log_interval,
    experiment_base_name,
    eval_freq,
    eval_episodes,
    debug,
    device,
    stochastic_eval,
    config,
    reward_unit,
    action_repeat,
):
    """Your CLI entrypoint for the training code

    Example usage:
    python src/blimp_env/train.py --eval-freq 10 --max-episode-steps 5000 --log-interval 100 --eval-episodes 3 --debug --do-env-check --dynamics-mode simple --feature-extractor cnn
    python src/blimp_env/train.py --config config/task2.yaml
    python -m blimp_env.train --config config/task2.yaml --disable-render

    """

    args = {
        "environment": environment,
        # "env_id": env_id,
        "max_train_steps": max_train_steps,
        "max_episode_steps": max_episode_steps,
        "feature_extractor": feature_extractor,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "dynamics_mode": dynamics_mode,
        "disable_render": disable_render,
        "do_env_check": do_env_check,
        "logdir": logdir,
        "log_interval": log_interval,
        "experiment_base_name": experiment_base_name,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "stochastic_eval": stochastic_eval,
        "debug": debug,
        "device": device,
        "reset_options": {},
        "reward_unit": reward_unit,
        "action_repeat": action_repeat
    }

    # Load configuration from YAML file if provided
    # Note: YAML config will override any CLI arguments
    if config:
        with open(config, 'r') as file:
            yaml_config = yaml.safe_load(file)

            # args = ChainMap(yaml_config, args)
            args.update(yaml_config)

    logger.info(f"Running with args:\n{args}")

    args = Namespace(**args)

    setup_logging(args.logdir)
    env = setup_env(
        args.environment,
        args.reset_options,
        args.dynamics_mode,
        args.max_episode_steps,
        args.debug,
        args.disable_render,
        args.log_interval,
        args.do_env_check,
        reward_unit=args.reward_unit,
        action_repeat=args.action_repeat,
    )

    eval_env = setup_env(
        args.environment,
        args.reset_options,
        args.dynamics_mode,
        args.max_episode_steps,
        args.debug,
        args.disable_render,
        args.log_interval,
        args.do_env_check,
        reward_unit=args.reward_unit,
        action_repeat=args.action_repeat,
    )
    callback = setup_callbacks(
        eval_env, args.eval_freq, args.eval_episodes, args.disable_render, args.logdir, args.stochastic_eval
    )
    model = setup_policy(
        env,
        args.feature_extractor,
        args.feature_dim,
        args.hidden_dim,
        args.disable_render,
        args.logdir,
        args.device,
        verbose=1,
    )

    # Train the agent
    model.learn(total_timesteps=int(args.max_train_steps), tb_log_name=args.experiment_base_name, callback=callback)


if __name__ == "__main__":
    main()
