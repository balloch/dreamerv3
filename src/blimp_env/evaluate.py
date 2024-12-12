import json

import click
import gymnasium as gym
from loguru import logger
import pandas as pd
import yaml
from stable_baselines3 import PPO


from blimp_env.wrappers import (
    ResizeDictWithImageObservation,
    NormalizePoseWrapper,
    NormalizeSensorWrapper,
)


def generate_value(start, stop, steps: int):
    """Generate steps equally spaced values between start and stop.

    E.g. multiple balloons [[12, 0, 4], [10, 0, 4]]"""
    if isinstance(start, list):
        result = []
        for start_row, stop_row in zip(start, stop):
            result.append(generate_value(start_row, stop_row, steps))
        yield from map(list, zip(*result))
    else:
        step = abs(stop - start) / (steps - 1)
        factor = 1 if start < stop else -1
        for idx in range(steps):
            yield start + idx * step * factor


@click.command()
@click.option("--trials", default=100, type=int, help="Number of evaluations.")
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    help="Path to YAML configuration file.",
)
@click.option("--model-path", type=click.Path(), help="Path to saved model file.")
@click.option("--output", type=click.Path(), help="Save results.", default="output.parquet")
@click.option("--params", type=click.Path(), help="Save step values.", default="output.json")
def main(trials, config_path, model_path, output, params):
    """Evaluate a model based on a configuration."""

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    steps = config.pop("steps", 1)
    varied = {}
    configs = []
    results = []
    keys = list(config["reset_options"].keys())
    for key in keys:
        if key.endswith("_final"):
            orig_key = key[:-6]
            start = config["reset_options"][orig_key]
            end = config["reset_options"][key]
            varied[orig_key] = list(generate_value(start, end, steps))
            config["reset_options"].pop(key)

    for trial_step in range(steps):
        current_config = {}
        logger.info(f"Working on step: {trial_step}")
        for key, val in varied.items():
            config["reset_options"][key] = val[trial_step]
            current_config[key] = val[trial_step]
        configs.append(current_config)
        logger.info(f"Running with args:\n{config}")
        model = PPO.load(model_path)

        Env = gym.envs.registration.load_env_creator(config.get("environment", "blim_env:IntoTheFireBasicNavEnv"))
        sensor_env = "FollowTask" in Env.__name__ or "ChaseTask" in Env.__name__
        reset_options = config.get("reset_options", {})
        if reset_options:
            Env.reset_options = reset_options
        img_size = (224, 224)
        env = Env(
            max_episode_steps=config['max_episode_steps'],
            dynamics=config['dynamics_mode'],
            render_mode='rgb_array',
            img_size=img_size,
            debug=config['debug'],
            disable_render=True,
            log_interval=config['log_interval'],
        )
        env = ResizeDictWithImageObservation(env, img_size)
        env = NormalizePoseWrapper(env)
        if sensor_env:
            env = NormalizeSensorWrapper(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: {'pose': obs['pose'], 'sensor': obs['sensor']})
        else:
            env = gym.wrappers.TransformObservation(env, lambda obs: {'pose': obs['pose']})

        for episode in range(trials):
            obs, _ = env.reset()
            totals = {
                "reward": 0,
                "from_fire": 0,
                "from_balloon": 0,
            }
            for step in range(config["max_episode_steps"]):
                obs.pop('img', None)
                action, _ = model.predict(obs)
                obs, reward, is_terminated, is_truncated, info = env.step(action)
                for key in totals:
                    totals[key] += info[key]
                if is_terminated or is_truncated:
                    for key in ("step_num", "num_balloons_collected"):
                        totals[key] = info[key]
                    totals["step"] = trial_step
                    break
            results.append(totals)
    logger.info(f"Configuration positions: {configs}")
    df = pd.DataFrame.from_records(results)
    df.to_parquet(output)
    with open(params, "w") as fh:
        json.dump(configs, fh)


if __name__ == "__main__":
    main()
