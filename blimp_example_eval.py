import warnings
from functools import partial as bind
from jax.lib import xla_bridge

import dreamerv3
import embodied

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


def main():
  mode='safety_adaptation'
  check_pt = '/home/general/logdir/20241101T001652-simple-blimp-1/checkpoint.ckpt'
  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size100m'],#['size50m'],
      'logdir': f'~/logdir_{mode}/{embodied.timestamp()}-simple-blimp-1',
      'run.train_ratio': 32,
      'run.mode': mode,
      'run.from_checkpoint': check_pt
  })
  ##DEBUGMODE
  #config = config.update({**dreamerv3.Agent.configs['debug'],})
  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def make_agent(config):
    env = make_env(config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent

  def make_logger(config):
    logdir = embodied.Path(config.logdir)
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandbOutput(logdir.name, config=config),
    ])

  def make_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / 'replay',
        online=config.replay.online)

  def make_env(config, env_id=0):

    import gymnasium as gym
    from embodied.envs import from_gymnasium
    from blimp_env.wrappers import (
      NormalizePoseWrapper,
    )
   
    Env = gym.envs.registration.load_env_creator("blimp_env:FollowTaskEnv")
    Env.reset_options = {
                          'pose': [0, 0, 4, 0, 0, 0],
                          'num_fires': 10,
                          'num_balloons': 10
                        }
    env = Env(dynamics='physics',render_mode='rgb_array',img_size=(64,64),disable_render=False,max_episode_steps=10000,reward_unit=.0001,action_repeat=20)    
    env = NormalizePoseWrapper(env)
    env = from_gymnasium.FromGymnasium(env)
    env = dreamerv3.wrap_env(env, config)
    return env

  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context
  )

  embodied.run.eval_only(
      bind(make_agent, config),
      bind(make_env, config),
      bind(make_logger, config), args)


if __name__ == '__main__':
  print(xla_bridge.get_backend().platform)  #NECESSARY TO GET THE GPU TO PLAY NICE
  main()