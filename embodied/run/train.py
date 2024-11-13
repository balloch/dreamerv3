import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np
import time 

def train(make_agent, make_replay, make_env, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  
  # avoid_replay = make_replay(args.avoid_logdir)
  # investigate_replay = make_replay(args.investigate_logdir)
  logger = make_logger()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  policy_fps = embodied.FPS()
  train_fps = embodied.FPS()

  batch_steps = args.batch_size * (args.batch_length - args.replay_context)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_eval = embodied.when.Clock(args.eval_every)
  should_save = embodied.when.Clock(args.save_every)


  @embodied.timer.section('log_step')
  def log_step(tran, worker):

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    
    if 'mode' not in tran.keys():
      episode.add('rewards', tran['reward'], agg='stack')
    elif tran['mode'][-1] == 0:
      episode.add('avoid_rewards', tran['avoid_reward'], agg='stack')
    elif tran['mode'][-1] == 1:
      episode.add('investigate_rewards', tran['investigate_reward'], agg='stack')
    else:
      episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    if worker < args.log_video_streams:
      for key in args.log_keys_video:
        if key in tran:
          episode.add(f'policy_{key}', tran[key], agg='stack')
    for key, value in tran.items():
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')

      if tran['mode'][-1] == 0:
        avoid_rew = result.pop('avoid_rewards')
        if len(avoid_rew) > 1:
          result['avoid_reward_rate'] = (np.abs(avoid_rew[1:] - avoid_rew[:-1]) >= 0.01).mean()
      elif tran['mode'][-1] == 1:
        investigate_rew = result.pop('investigate_rewards')
        if len(investigate_rew) > 1:
          result['investigate_reward_rate'] = (np.abs(investigate_rew[1:] - investigate_rew[:-1]) >= 0.01).mean()
      else:
        rew = result.pop('rewards')
        if len(rew) > 1:
          result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
       
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.num_envs)]
  driver = embodied.Driver(fns, args.driver_parallel)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(log_step)

  dataset_train = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length)))
  
  dataset_report = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length_eval)))

  
  carry = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def train_step(tran, worker):
    if len(replay) < args.batch_size or step < args.train_fill:
      return
    for _ in range(should_train(step)):
      with embodied.timer.section('dataset_next'):
        batch = next(dataset_train)
      outs, carry[0], mets = agent.train(batch, carry[0])
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      agg.add(mets, prefix='train')
  print('Train step initialize')
  driver.on_step(train_step)


  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay

  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  avoid_policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'avoid')
  investigate_policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'investigate')
  #TODO reacts based on environment.
  multi_skill_policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  
  # avoid_policy = lambda *args: agent.policy(
  #     *args, mode='explore' if should_expl(step) else 'avoid')
  # investigate_policy = lambda *args: agent.policy(
  #     *args, mode='explore' if should_expl(step) else 'investigate')
  
  driver.reset(agent.init_policy)
  # avoid_driver.reset(agent.init_policy)
  # investigate_driver.reset(agent.init_policy)
  step_cycle = 5000 
  should_investigate = embodied.when.Until(step+step_cycle)
  should_be_normal = embodied.when.Until(step+(step_cycle*2))
  should_avoid = embodied.when.Until(step+(step_cycle*3))

  while step < args.steps:
    # if should_investigate(step):
    #   driver(investigate_policy, steps=10)
    # elif should_avoid(step):
    #   driver(avoid_policy, steps=10)
    # elif should_be_normal(step):
    #   driver(policy, steps=10)
    # else:
    #   should_investigate = embodied.when.Until(step+step_cycle)
    #   should_be_normal = embodied.when.Until(step+(step_cycle*2))
    #   should_avoid = embodied.when.Until(step+(step_cycle*3))
    driver(policy,episodes=10)
    
    
    if should_eval(step) and len(replay):
      print('Evaluate Avoid')
      driver(avoid_policy, episodes=100)
      time.sleep(10)
      print('Evaluate Investigate')
      driver(investigate_policy, episodes=100)
      time.sleep(10)
      mets, _ = agent.report(next(dataset_report), carry_report)
      logger.add(mets, prefix='report')

    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.write()

    if should_save(step):
      checkpoint.save()

  logger.close()
