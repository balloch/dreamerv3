import re
from functools import partial as bind

import embodied
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ruamel.yaml as yaml
import torch
from PIL import Image
import time 
# import os
# path = ''
# os.environ['PATH'] += ':'+path

# from .MobileVLM_main.mobilevlm.model.mobilevlm import load_pretrained_model
# from .MobileVLM_main.mobilevlm.conversation import conv_templates, SeparatorStyle
# from .MobileVLM_main.mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
# from .MobileVLM_main.mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}


@jaxagent.Wrapper
class Agent(nj.Module):
  
  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, config):
    self.obs_space = {
        k: v for k, v in obs_space.items() if not k.startswith('log_')}
    
    self.act_space = {
        k: v for k, v in act_space.items() if k != 'reset'}
    self.config = config
    enc_space = {
        k: v for k, v in obs_space.items()
        if k not in ('is_first', 'is_last', 'is_terminal', 'reward', 'avoid_reward', 'investigate_reward') and
        not k.startswith('log_') and re.match(config.enc.spaces, k)}
    dec_space = {
        k: v for k, v in obs_space.items()
        if k not in ('is_first', 'is_last', 'is_terminal', 'reward', 'avoid_reward', 'investigate_reward') and
        not k.startswith('log_') and re.match(config.dec.spaces, k)}
    embodied.print('Encoder:', {k: v.shape for k, v in enc_space.items()})
    embodied.print('Decoder:', {k: v.shape for k, v in dec_space.items()})
    # model_path = "mtgv/MobileVLM-1.7B" # MobileVLM V2
    # model_name =  model_path.split('/')[-1]
    # self.model_name = model_name
    # image_file = "assets/samples/IntoFire.png"
    # prompt_str = "Is there fire?"
    # print('Model Path: ',model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, False, False)
    # self.VLM_model = model
    # self.VLM_tokenizer = tokenizer
    # self.image_processor = image_processor
    # self.context_len = context_len
    # self.prompt = prompt_str
    # print('Model Loaded')

    # World Model
    self.enc = {
        'simple': bind(nets.SimpleEncoder, **config.enc.simple),
    }[config.enc.typ](enc_space, name='enc')
    self.dec = {
        'simple': bind(nets.SimpleDecoder, **config.dec.simple),
    }[config.dec.typ](dec_space, name='dec')
    self.dyn = {
        'rssm': bind(nets.RSSM, **config.dyn.rssm),
    }[config.dyn.typ](name='dyn')
    self.rew = nets.MLP((), **config.rewhead, name='rew')
    self.avoid_rew = nets.MLP((), **config.rewhead, name='avoid_rew')
    self.investigate_rew = nets.MLP((), **config.rewhead, name='investigate_rew')
    self.con = nets.MLP((), **config.conhead, name='con')
    # Actor
    kwargs = {}
    kwargs['shape'] = {
        k: (*s.shape, s.classes) if s.discrete else s.shape
        for k, s in self.act_space.items()}
    kwargs['dist'] = {
        k: config.actor_dist_disc if v.discrete else config.actor_dist_cont
        for k, v in self.act_space.items()}
    self.actor = nets.MLP(**kwargs, **config.actor, name='actor')
    self.avoid_actor = nets.MLP(**kwargs, **config.actor, name='avoid_actor')
    self.investigate_actor = nets.MLP(**kwargs, **config.actor, name='investigate_actor')

    self.retnorm = jaxutils.Moments(**config.retnorm, name='retnorm')
    self.valnorm = jaxutils.Moments(**config.valnorm, name='valnorm')
    self.advnorm = jaxutils.Moments(**config.advnorm, name='advnorm')

    # Critic
    self.critic = nets.MLP((), name='critic', **self.config.critic)
    self.slowcritic = nets.MLP(
        (), name='slowcritic', **self.config.critic, dtype='float32')
    self.updater = jaxutils.SlowUpdater(
        self.critic, self.slowcritic,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='updater')

    self.avoid_critic = nets.MLP((), name='avoid_critic', **self.config.critic)
    self.avoid_slowcritic = nets.MLP(
        (), name='avoid_slowcritic', **self.config.critic, dtype='float32')
    self.avoid_updater = jaxutils.SlowUpdater(
        self.avoid_critic, self.avoid_slowcritic,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='avoid_updater')
    
    self.investigate_critic = nets.MLP((), name='investigate_critic', **self.config.critic)
    self.investigate_slowcritic = nets.MLP(
        (), name='investigate_slowcritic', **self.config.critic, dtype='float32')
    self.investigate_updater = jaxutils.SlowUpdater(
        self.investigate_critic, self.investigate_slowcritic,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='investigate_updater')
    
    # Optimizer
    kw = dict(config.opt)
    lr = kw.pop('lr')
    if config.separate_lrs:
      lr = {f'agent/{k}': v for k, v in config.lrs.items()}
    self.opt = jaxutils.Optimizer(lr, **kw, name='opt')
    # self.modules = [
    #     self.enc, self.dyn, self.dec, self.rew, self.con,
    #     self.actor, self.critic]
    # self.modules = [
    #     self.enc, self.dyn, self.dec, self.rew, self.avoid_rew, self.investigate_rew, self.con,
    #     self.actor, self.critic]
    self.modules = [
        self.enc, self.dyn, self.dec, self.rew, self.avoid_rew, self.investigate_rew, self.con,
        self.actor, self.avoid_actor, self.investigate_actor, self.critic, self.avoid_critic, self.investigate_critic]
    scales = self.config.loss_scales.copy()
    cnn = scales.pop('dec_cnn')
    mlp = scales.pop('dec_mlp')
    scales.update({k: cnn for k in self.dec.imgkeys})
    scales.update({k: mlp for k in self.dec.veckeys})
    self.scales = scales

  @property
  def policy_keys(self):
    return '/(enc|dyn|actor)/'

  @property
  def aux_spaces(self):
    spaces = {}
    spaces['stepid'] = embodied.Space(np.uint8, 20)
    if self.config.replay_context:
      latdtype = jaxutils.COMPUTE_DTYPE
      latdtype = np.float32 if latdtype == jnp.bfloat16 else latdtype
      spaces['deter'] = embodied.Space(latdtype, self.config.dyn.rssm.deter)
      spaces['stoch'] = embodied.Space(np.int32, self.config.dyn.rssm.stoch)
    return spaces

  def init_policy(self, batch_size):
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return (self.dyn.initial(batch_size), prevact)

  def init_train(self, batch_size):
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return (self.dyn.initial(batch_size), prevact)

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  def policy(self, obs, carry, mode='train'):
    self.config.jax.jit and embodied.print(
        'Tracing policy function', color='yellow')
    prevlat, prevact = carry
    obs = self.preprocess(obs)
    
    ####USED FOR VLM######
    # if obs['is_first'][0] == False:
    #   print(obs['img'])
    # self.VLM_response(obs['img'])
    # time.sleep(10)
    ####END OF VLM CODE######

    embed = self.enc(obs, bdims=1)
    prevact = jaxutils.onehot_dict(prevact, self.act_space)
    lat, out = self.dyn.observe(
        prevlat, prevact, embed, obs['is_first'], bdims=1)
    
    outs = {}
    if self.config.replay_context:
      outs.update({k: out[k] for k in self.aux_spaces if k != 'stepid'})
      outs['stoch'] = jnp.argmax(outs['stoch'], -1).astype(jnp.int32)

    if mode == 'avoid': # 0 
      actor = self.avoid_actor(out, bdims=1)
      outs['mode'] = jnp.full_like(outs['stoch'], 0)
    elif mode == 'investigate': # 1
      actor = self.investigate_actor(out, bdims=1)
      outs['mode'] = jnp.full_like(outs['stoch'], 1)
    else: # 2
      outs['mode'] = jnp.full_like(outs['stoch'], 2)
      actor = self.actor(out, bdims=1)
    act = sample(actor)

    outs['finite'] = {
        '/'.join(x.key for x in k): (
            jnp.isfinite(v).all(range(1, v.ndim)),
            v.min(range(1, v.ndim)),
            v.max(range(1, v.ndim)))
        for k, v in jax.tree_util.tree_leaves_with_path(dict(
            obs=obs, prevlat=prevlat, prevact=prevact,
            embed=embed, act=act, out=out, lat=lat,
        ))}

    assert all(
        k in outs for k in self.aux_spaces
        if k not in ('stepid', 'finite', 'is_online')), (
              list(outs.keys()), self.aux_spaces)
    act = {
        k: jnp.nanargmax(act[k], -1).astype(jnp.int32)
        if s.discrete else act[k] for k, s in self.act_space.items()}

    return act, outs, (lat, act)
  
  def train(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing train function', color='yellow')
    data = self.preprocess(data)
    stepid = data.pop('stepid')

    if self.config.replay_context:
      K = self.config.replay_context
      data = data.copy()
      context = {
          k: data.pop(k)[:, :K] for k in self.aux_spaces if k != 'stepid'}
      context['stoch'] = f32(jax.nn.one_hot(
          context['stoch'], self.config.dyn.rssm.classes))
      prevlat = self.dyn.outs_to_carry(context)
      prevact = {k: data[k][:, K - 1] for k in self.act_space}
      carry = prevlat, prevact
      data = {k: v[:, K:] for k, v in data.items()}
      stepid = stepid[:, K:]

    if self.config.reset_context:
      keep = jax.random.uniform(
          nj.seed(), data['is_first'][:, :1].shape) > self.config.reset_context
      data['is_first'] = jnp.concatenate([
          data['is_first'][:, :1] & keep, data['is_first'][:, 1:]], 1)

    mets, (out, carry, metrics) = self.opt(
        self.modules, self.loss, data, carry, has_aux=True)
    metrics.update(mets)
    self.updater()
    outs = {}

    if self.config.replay_context:
      outs['replay'] = {'stepid': stepid}
      outs['replay'].update({
          k: out['replay_outs'][k] for k in self.aux_spaces if k != 'stepid'})
      outs['replay']['stoch'] = jnp.argmax(
          outs['replay']['stoch'], -1).astype(jnp.int32)

    if self.config.replay.fracs.priority > 0:
      bs = data['is_first'].shape
      if self.config.replay.priosignal == 'td':
        priority = out['critic_loss'][:, 0].reshape(bs)
      elif self.config.replay.priosignal == 'model':
        terms = [out[f'{k}_loss'] for k in (
            'rep', 'dyn', *self.dec.veckeys, *self.dec.imgkeys)]
        priority = jnp.stack(terms, 0).sum(0)
      elif self.config.replay.priosignal == 'all':
        terms = [out[f'{k}_loss'] for k in (
            'rep', 'dyn', *self.dec.veckeys, *self.dec.imgkeys)]
        terms.append(out['actor_loss'][:, 0].reshape(bs))
        terms.append(out['critic_loss'][:, 0].reshape(bs))
        priority = jnp.stack(terms, 0).sum(0)
      else:
        raise NotImplementedError(self.config.replay.priosignal)
      assert stepid.shape[:2] == priority.shape == bs
      outs['replay'] = {'stepid': stepid, 'priority': priority}

    return outs, carry, metrics

  
  def loss(self, data, carry, update=True):
    metrics = {}
    prevlat, prevact = carry
    # Replay rollout
    prevacts = {
        k: jnp.concatenate([prevact[k][:, None], data[k][:, :-1]], 1)
        for k in self.act_space}
    prevacts = jaxutils.onehot_dict(prevacts, self.act_space)
    embed = self.enc(data)
    newlat, outs = self.dyn.observe(prevlat, prevacts, embed, data['is_first'])
    rew_feat = outs if self.config.reward_grad else sg(outs)
    dists = dict(
        **self.dec(outs),
        reward=self.rew(rew_feat, training=True),
        avoid_reward=self.avoid_rew(rew_feat, training=True),
        investigate_reward=self.investigate_rew(rew_feat,training=True),
        cont=self.con(outs, training=True))
    losses = {k: -v.log_prob(f32(data[k])) for k, v in dists.items()}

    if self.config.contdisc:
      del losses['cont']
      softlabel = data['cont'] * (1 - 1 / self.config.horizon)
      losses['cont'] = -dists['cont'].log_prob(softlabel)
    dynlosses, mets = self.dyn.loss(outs, **self.config.rssm_loss)
    losses.update(dynlosses)
    metrics.update(mets)
    replay_outs = outs

    # Imagination rollout
    def imgstep(carry, _):
      lat, act = carry
      lat, out = self.dyn.imagine(lat, act, bdims=1)
      out['stoch'] = sg(out['stoch'])
      act = cast(sample(self.actor(out, bdims=1)))
      return (lat, act), (out, act)
    
    # Imagination rollout avoid actor
    def avoid_imgstep(carry, _):
      lat, act = carry
      lat, out = self.dyn.imagine(lat, act, bdims=1)
      out['stoch'] = sg(out['stoch'])
      act = cast(sample(self.avoid_actor(out, bdims=1)))
      return (lat, act), (out, act)
    
    # Imagination rollout investigate actor
    def investigate_imgstep(carry, _):
      lat, act = carry
      lat, out = self.dyn.imagine(lat, act, bdims=1)
      out['stoch'] = sg(out['stoch'])
      act = cast(sample(self.investigate_actor(out, bdims=1)))
      return (lat, act), (out, act)
    
    rew = data['reward']
    avoid_rew = data['avoid_reward']
    investigate_rew = data['investigate_reward']

    con = 1 - f32(data['is_terminal'])
    if self.config.imag_start == 'all':
      B, T = data['is_first'].shape
      startlat = self.dyn.outs_to_carry(treemap(
          lambda x: x.reshape((B * T, 1, *x.shape[2:])), replay_outs))
      startout, startrew, startcon = treemap(
          lambda x: x.reshape((B * T, *x.shape[2:])),
          (replay_outs, rew, con))
      avoid_startrew = treemap(
          lambda x: x.reshape((B * T, *x.shape[2:])),
          (avoid_rew))
      investigate_startrew = treemap(
          lambda x: x.reshape((B * T, *x.shape[2:])),
          (investigate_rew))
    elif self.config.imag_start == 'last':
      startlat = newlat
      startout, startrew, startcon = treemap(
          lambda x: x[:, -1], (replay_outs, rew, con))
      avoid_startrew= treemap(
          lambda x: x[:, -1], (avoid_rew))
      investigate_startrew = treemap(
          lambda x: x[:, -1], (investigate_rew))
      
    if self.config.imag_repeat > 1:
      N = self.config.imag_repeat
      startlat, startout, startrew, startcon = treemap(
          lambda x: x.repeat(N, 0), (startlat, startout, startrew, startcon))
      avoid_startrew = treemap(
          lambda x: x.repeat(N, 0), (avoid_startrew))
      investigate_startrew = treemap(
          lambda x: x.repeat(N, 0), (investigate_startrew))
      
    startact = cast(sample(self.actor(startout, bdims=1)))
    avoid_startact = cast(sample(self.avoid_actor(startout, bdims=1)))
    investigate_startact = cast(sample(self.investigate_actor(startout, bdims=1)))

    _, (outs, acts) = jaxutils.scan(
        imgstep, sg((startlat, startact)),
        jnp.arange(self.config.imag_length), self.config.imag_unroll)
    
    _, (avoid_outs, avoid_acts) = jaxutils.scan(
        avoid_imgstep, sg((startlat, avoid_startact)),
        jnp.arange(self.config.imag_length), self.config.imag_unroll)
    
    _, (investigate_outs, investigate_acts) = jaxutils.scan(
        investigate_imgstep, sg((startlat, investigate_startact)),
        jnp.arange(self.config.imag_length), self.config.imag_unroll)
    
    outs, acts = treemap(lambda x: x.swapaxes(0, 1), (outs, acts))
    outs, acts = treemap(
        lambda first, seq: jnp.concatenate([first, seq], 1),
        treemap(lambda x: x[:, None], (startout, startact)), (outs, acts))
    
    avoid_outs, avoid_acts = treemap(lambda x: x.swapaxes(0, 1), (avoid_outs, avoid_acts))
    avoid_outs, avoid_acts = treemap(
        lambda first, seq: jnp.concatenate([first, seq], 1),
        treemap(lambda x: x[:, None], (startout, avoid_startact)), (avoid_outs, avoid_acts))
    
    investigate_outs, investigate_acts = treemap(lambda x: x.swapaxes(0, 1), (investigate_outs, investigate_acts))
    investigate_outs, investigate_acts = treemap(
        lambda first, seq: jnp.concatenate([first, seq], 1),
        treemap(lambda x: x[:, None], (startout, investigate_startact)), (investigate_outs, investigate_acts))

    # Note that the rewards are given by their actors actions....
    rew = jnp.concatenate([startrew[:, None], self.rew(outs).mean()[:, 1:]], 1)
    avoid_rew = jnp.concatenate([avoid_startrew[:, None], self.avoid_rew(avoid_outs).mean()[:, 1:]], 1)
    investigate_rew = jnp.concatenate([investigate_startrew[:, None], self.investigate_rew(investigate_outs).mean()[:, 1:]], 1)

    con = jnp.concatenate([startcon[:, None], self.con(outs).mean()[:, 1:]], 1)
    acts = sg(acts)
    avoid_acts = sg(avoid_acts)
    investigate_acts = sg(investigate_acts)

    inp = treemap({
        'none': lambda x: sg(x),
        'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1),
        'all': lambda x: x,
    }[self.config.ac_grads], outs)

    avoid_inp = treemap({
        'none': lambda x: sg(x),
        'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1),
        'all': lambda x: x,
    }[self.config.ac_grads], avoid_outs)

    investigate_inp = treemap({
        'none': lambda x: sg(x),
        'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1),
        'all': lambda x: x,
    }[self.config.ac_grads], investigate_outs)

    actor = self.actor(inp)
    critic = self.critic(inp)
    slowcritic = self.slowcritic(inp)

    avoid_actor = self.avoid_actor(avoid_inp)
    avoid_critic = self.avoid_critic(avoid_inp)
    avoid_slowcritic = self.avoid_slowcritic(avoid_inp)

    investigate_actor = self.investigate_actor(investigate_inp)
    investigate_critic = self.investigate_critic(investigate_inp)
    investigate_slowcritic = self.investigate_slowcritic(investigate_inp)
    
   
    voffset, vscale = self.valnorm.stats()

    # vals from critic
    val = critic.mean() * vscale + voffset
    slowval = slowcritic.mean() * vscale + voffset
    tarval = slowval if self.config.slowtar else val
    discount = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    weight = jnp.cumprod(discount * con, 1) / discount

    # vals from avoid_critic
    avoid_val = avoid_critic.mean() * vscale + voffset
    avoid_slowval = avoid_slowcritic.mean() * vscale + voffset
    avoid_tarval = avoid_slowval if self.config.slowtar else avoid_val

    # vals from investigate_critic
    investigate_val = investigate_critic.mean() * vscale + voffset
    investigate_slowval = investigate_slowcritic.mean() * vscale + voffset
    investigate_tarval = investigate_slowval if self.config.slowtar else investigate_val
    
    # Return
    rets = [tarval[:, -1]]
    avoid_rets = [avoid_tarval[:, -1]]
    investigate_rets = [investigate_tarval[:, -1]]

    disc = con[:, 1:] * discount
    lam = self.config.return_lambda

    interm = rew[:, 1:] + (1 - lam) * disc * tarval[:, 1:]
    avoid_interm = avoid_rew[:, 1:] + (1 - lam) * disc * avoid_tarval[:, 1:]
    investigate_interm = investigate_rew[:, 1:] + (1 - lam) * disc * investigate_tarval[:, 1:]

    for t in reversed(range(disc.shape[1])):
      rets.append(interm[:, t] + disc[:, t] * lam * rets[-1])
      avoid_rets.append(avoid_interm[:, t] + disc[:, t] * lam * avoid_rets[-1])
      investigate_rets.append(investigate_interm[:, t] + disc[:, t] * lam * investigate_rets[-1])

    ret = jnp.stack(list(reversed(rets))[:-1], 1)
    avoid_ret = jnp.stack(list(reversed(avoid_rets))[:-1], 1)
    investigate_ret = jnp.stack(list(reversed(investigate_rets))[:-1], 1)

    # Actor
    roffset, rscale = self.retnorm(ret, update)  
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = self.advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale
    logpi = sum([v.log_prob(sg(acts[k]))[:, :-1] for k, v in actor.items()])
    ents = {k: v.entropy()[:, :-1] for k, v in actor.items()}
    actor_loss = sg(weight[:, :-1]) * -(
        logpi * sg(adv_normed) + self.config.actent * sum(ents.values()))
    losses['actor'] = actor_loss

    # Avoid Actor
    avoid_roffset, avoid_rscale = self.retnorm(avoid_ret, update)
    avoid_adv = (avoid_ret - avoid_tarval[:, :-1]) / avoid_rscale
    avoid_aoffset, avoid_ascale = self.advnorm(avoid_adv, update)
    avoid_adv_normed = (avoid_adv - avoid_aoffset) / avoid_ascale
    avoid_logpi = sum([v.log_prob(sg(avoid_acts[k]))[:, :-1] for k, v in avoid_actor.items()])
    avoid_ents = {k: v.entropy()[:, :-1] for k, v in avoid_actor.items()}
    avoid_actor_loss = sg(weight[:, :-1]) * -(
        avoid_logpi * sg(avoid_adv_normed) + self.config.actent * sum(avoid_ents.values()))
    losses['avoid_actor'] = avoid_actor_loss

    # Investigate Actor
    investigate_roffset, investigate_rscale = self.retnorm(investigate_ret, update)
    investigate_adv = (investigate_ret - investigate_tarval[:, :-1]) / investigate_rscale
    investigate_aoffset, investigate_ascale = self.advnorm(investigate_adv, update)
    investigate_adv_normed = (investigate_adv - investigate_aoffset) / investigate_ascale
    investigate_logpi = sum([v.log_prob(sg(investigate_acts[k]))[:, :-1] for k, v in investigate_actor.items()])
    investigate_ents = {k: v.entropy()[:, :-1] for k, v in investigate_actor.items()}
    investigate_actor_loss = sg(weight[:, :-1]) * -(
        investigate_logpi * sg(investigate_adv_normed) + self.config.actent * sum(investigate_ents.values()))
    losses['investigate_actor'] = investigate_actor_loss


    # Critic
    voffset, vscale = self.valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses['critic'] = sg(weight)[:, :-1] * -(
        critic.log_prob(sg(ret_padded)) +
        self.config.slowreg * critic.log_prob(sg(slowcritic.mean())))[:, :-1]
    
    # Avoid Critic
    avoid_voffset, avoid_vscale = self.valnorm(avoid_ret, update)
    avoid_ret_normed = (avoid_ret - avoid_voffset) / avoid_vscale
    avoid_ret_padded = jnp.concatenate([avoid_ret_normed, 0 * avoid_ret_normed[:, -1:]], 1)
    losses['avoid_critic'] = sg(weight)[:, :-1] * -(
        avoid_critic.log_prob(sg(avoid_ret_padded)) +
        self.config.slowreg * critic.log_prob(sg(avoid_slowcritic.mean())))[:, :-1]
    
    # investigate Critic
    investigate_voffset, investigate_vscale = self.valnorm(investigate_ret, update)
    investigate_ret_normed = (investigate_ret - investigate_voffset) / investigate_vscale
    investigate_ret_padded = jnp.concatenate([investigate_ret_normed, 0 * investigate_ret_normed[:, -1:]], 1)
    losses['investigate_critic'] = sg(weight)[:, :-1] * -(
        investigate_critic.log_prob(sg(investigate_ret_padded)) +
        self.config.slowreg * critic.log_prob(sg(investigate_slowcritic.mean())))[:, :-1]
    
   

    if self.config.replay_critic_loss:
      replay_critic = self.critic(
          replay_outs if self.config.replay_critic_grad else sg(replay_outs))
      avoid_replay_critic = self.avoid_critic(
          replay_outs if self.config.replay_critic_grad else sg(replay_outs))
      investigate_replay_critic = self.investigate_critic(
          replay_outs if self.config.replay_critic_grad else sg(replay_outs))
      
      replay_slowcritic = self.slowcritic(replay_outs)
      avoid_replay_slowcritic = self.avoid_slowcritic(replay_outs)
      investigate_replay_slowcritic = self.investigate_slowcritic(replay_outs)

      boot = dict(
          imag=ret[:, 0].reshape(data['reward'].shape),
          critic=replay_critic.mean(),
      )[self.config.replay_critic_bootstrap]

      avoid_boot = dict(
          imag=avoid_ret[:, 0].reshape(data['avoid_reward'].shape),
          critic=avoid_replay_critic.mean(),
      )[self.config.replay_critic_bootstrap]

      investigate_boot = dict(
          imag=investigate_ret[:, 0].reshape(data['investigate_reward'].shape),
          critic=investigate_replay_critic.mean(),
      )[self.config.replay_critic_bootstrap]


      rets = [boot[:, -1]]
      avoid_rets = [avoid_boot[:, -1]]
      investigate_rets = [investigate_boot[:, -1]]

      live = f32(~data['is_terminal'])[:, 1:] * (1 - 1 / self.config.horizon)
      cont = f32(~data['is_last'])[:, 1:] * self.config.return_lambda_replay

      interm = data['reward'][:, 1:] + (1 - cont) * live * boot[:, 1:]
      avoid_interm = data['avoid_reward'][:, 1:] + (1 - cont) * live * avoid_boot[:, 1:]
      investigate_interm = data['investigate_reward'][:, 1:] + (1 - cont) * live * investigate_boot[:, 1:]

      for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
        avoid_rets.append(avoid_interm[:, t] + live[:, t] * cont[:, t] * avoid_rets[-1])
        investigate_rets.append(investigate_interm[:, t] + live[:, t] * cont[:, t] * investigate_rets[-1])

      replay_ret = jnp.stack(list(reversed(rets))[:-1], 1)
      avoid_replay_ret = jnp.stack(list(reversed(avoid_rets))[:-1], 1)
      investigate_replay_ret = jnp.stack(list(reversed(investigate_rets))[:-1], 1)
      
      voffset, vscale = self.valnorm(replay_ret, update)
      avoid_voffset, avoid_vscale = self.valnorm(avoid_replay_ret, update)
      investigate_voffset, investigate_vscale = self.valnorm(investigate_replay_ret, update)
      
      ret_normed = (replay_ret - voffset) / vscale
      avoid_ret_normed = (avoid_replay_ret - avoid_voffset) / avoid_vscale
      investigate_ret_normed = (investigate_replay_ret - investigate_voffset) / investigate_vscale

      ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
      avoid_ret_padded = jnp.concatenate([avoid_ret_normed, 0 * avoid_ret_normed[:, -1:]], 1)
      investigate_ret_padded = jnp.concatenate([investigate_ret_normed, 0 * investigate_ret_normed[:, -1:]], 1)

      losses['replay_critic'] = sg(f32(~data['is_last']))[:, :-1] * -(
          replay_critic.log_prob(sg(ret_padded)) +
          self.config.slowreg * replay_critic.log_prob(
              sg(replay_slowcritic.mean())))[:, :-1]
      
      losses['avoid_replay_critic'] = sg(f32(~data['is_last']))[:, :-1] * -(
          avoid_replay_critic.log_prob(sg(avoid_ret_padded)) +
          self.config.slowreg * avoid_replay_critic.log_prob(
              sg(avoid_replay_slowcritic.mean())))[:, :-1]
      
      losses['investigate_replay_critic'] = sg(f32(~data['is_last']))[:, :-1] * -(
          investigate_replay_critic.log_prob(sg(investigate_ret_padded)) +
          self.config.slowreg * investigate_replay_critic.log_prob(
              sg(investigate_replay_slowcritic.mean())))[:, :-1]

    # Metrics
    metrics.update({f'{k}_loss': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics.update(jaxutils.tensorstats(rew, 'rew'))
    metrics.update(jaxutils.tensorstats(avoid_rew, 'avoid_rew'))
    metrics.update(jaxutils.tensorstats(investigate_rew, 'investigate_rew'))
    metrics.update(jaxutils.tensorstats(weight, 'weight'))
    metrics.update(jaxutils.tensorstats(val, 'val'))
    metrics.update(jaxutils.tensorstats(ret, 'ret'))
    metrics.update(jaxutils.tensorstats(
        (ret - roffset) / rscale, 'ret_normed'))
    
    if self.config.replay_critic_loss:
      metrics.update(jaxutils.tensorstats(replay_ret, 'replay_ret'))
      metrics.update(jaxutils.tensorstats(avoid_replay_ret, 'avoid_replay_ret'))
      metrics.update(jaxutils.tensorstats(investigate_replay_ret, 'investigate_replay_ret'))

    metrics['td_error'] = jnp.abs(ret - val[:, :-1]).mean()
    metrics['ret_rate'] = (jnp.abs(ret) > 1.0).mean()
    
    metrics['avoid_td_error'] = jnp.abs(avoid_ret - avoid_val[:, :-1]).mean()
    metrics['avoid_ret_rate'] = (jnp.abs(avoid_ret) > 1.0).mean()

    metrics['investigate_td_error'] = jnp.abs(investigate_ret - investigate_val[:, :-1]).mean()
    metrics['investigate_ret_rate'] = (jnp.abs(investigate_ret) > 1.0).mean()

    for k, space in self.act_space.items():
      act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
      avoid_act = f32(jnp.argmax(avoid_acts[k], -1) if space.discrete else avoid_acts[k])
      investigate_act = f32(jnp.argmax(investigate_acts[k], -1) if space.discrete else investigate_acts[k])

      metrics.update(jaxutils.tensorstats(f32(act), f'act/{k}'))
      metrics.update(jaxutils.tensorstats(f32(avoid_act), f'act/{k}'))
      metrics.update(jaxutils.tensorstats(f32(investigate_act), f'act/{k}'))

      if hasattr(actor[k], 'minent'):
        lo, hi = actor[k].minent, actor[k].maxent
        rand = ((ents[k] - lo) / (hi - lo)).mean(
            range(2, len(ents[k].shape)))
        metrics.update(jaxutils.tensorstats(rand, f'rand/{k}'))
      metrics.update(jaxutils.tensorstats(ents[k], f'ent/{k}'))

      if hasattr(avoid_actor[k], 'minent'):
        avoid_lo, avoid_hi = avoid_actor[k].minent, avoid_actor[k].maxent
        avoid_rand = ((avoid_ents[k] - avoid_lo) / (avoid_hi - avoid_lo)).mean(
            range(2, len(avoid_ents[k].shape)))
        metrics.update(jaxutils.tensorstats(avoid_rand, f'avoid_rand/{k}'))
      metrics.update(jaxutils.tensorstats(avoid_ents[k], f'avoid_ent/{k}'))

      if hasattr(investigate_actor[k], 'minent'):
        investigate_lo, investigate_hi = investigate_actor[k].minent, investigate_actor[k].maxent
        investigate_rand = ((investigate_ents[k] - investigate_lo) / (investigate_hi - investigate_lo)).mean(
            range(2, len(investigate_ents[k].shape)))
        metrics.update(jaxutils.tensorstats(investigate_rand, f'investigate_rand/{k}'))
      metrics.update(jaxutils.tensorstats(investigate_ents[k], f'investigate_ent/{k}'))

      
    metrics['data_rew/max'] = jnp.abs(data['reward']).max()
    metrics['pred_rew/max'] = jnp.abs(rew).max()
    metrics['data_rew/mean'] = data['reward'].mean()
    metrics['pred_rew/mean'] = rew.mean()
    metrics['data_rew/std'] = data['reward'].std()
    metrics['pred_rew/std'] = rew.std()

    metrics['data_avoid_rew/max'] = jnp.abs(data['avoid_reward']).max()
    metrics['pred_avoid_rew/max'] = jnp.abs(avoid_rew).max()
    metrics['data_avoid_rew/mean'] = data['avoid_reward'].mean()
    metrics['pred_avoid_rew/mean'] = avoid_rew.mean()
    metrics['data_avoid_rew/std'] = data['avoid_reward'].std()
    metrics['pred_avoid_rew/std'] = avoid_rew.std()

    metrics['data_investigate_rew/max'] = jnp.abs(data['investigate_reward']).max()
    metrics['pred_investigate_rew/max'] = jnp.abs(investigate_rew).max()
    metrics['data_investigate_rew/mean'] = data['investigate_reward'].mean()
    metrics['pred_investigate_rew/mean'] = investigate_rew.mean()
    metrics['data_investigate_rew/std'] = data['investigate_reward'].std()
    metrics['pred_investigate_rew/std'] = investigate_rew.std()

    if 'reward' in dists:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'rewstats/{k}': v for k, v in stats.items()})
    if 'avoid_reward' in dists:
      stats = jaxutils.balance_stats(dists['avoid_reward'], data['avoid_reward'], 0.1)
      metrics.update({f'rewstats/{k}': v for k, v in stats.items()})
    if 'investigate_reward' in dists:
      stats = jaxutils.balance_stats(dists['investigate_reward'], data['investigate_reward'], 0.1)
      metrics.update({f'rewstats/{k}': v for k, v in stats.items()})
    if 'cont' in dists:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'constats/{k}': v for k, v in stats.items()})
    metrics['activation/embed'] = jnp.abs(embed).mean()
    # metrics['activation/deter'] = jnp.abs(replay_outs['deter']).mean()

    # Combine
    losses = {k: v * self.scales[k] for k, v in losses.items()}
    loss = jnp.stack([v.mean() for k, v in losses.items()]).sum()
    newact = {k: data[k][:, -1] for k in self.act_space}
    outs = {'replay_outs': replay_outs, 'prevacts': prevacts, 'embed': embed}
    outs.update({f'{k}_loss': v for k, v in losses.items()})
    carry = (newlat, newact)
    return loss, (outs, carry, metrics)
  

  def report(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing report function', color='yellow')
    if not self.config.report:
      return {}, carry
    metrics = {}
    data = self.preprocess(data)

    # Train metrics
    _, (outs, carry_out, mets) = self.loss(data, carry, update=False)
    metrics.update(mets)

    # Open loop predictions
    B, T = data['is_first'].shape
    num_obs = min(self.config.report_openl_context, T // 2)
    # Rerun observe to get the correct intermediate state, because
    # outs_to_carry doesn't work with num_obs<context.
    img_start, rec_outs = self.dyn.observe(
        carry[0],
        {k: v[:, :num_obs] for k, v in outs['prevacts'].items()},
        outs['embed'][:, :num_obs],
        data['is_first'][:, :num_obs])
    img_acts = {k: v[:, num_obs:] for k, v in outs['prevacts'].items()}
    img_outs = self.dyn.imagine(img_start, img_acts)[1]
    rec = dict(
        **self.dec(rec_outs), reward=self.rew(rec_outs),
        cont=self.con(rec_outs))
    img = dict(
        **self.dec(img_outs), reward=self.rew(img_outs),
        cont=self.con(img_outs))

    # Prediction losses
    data_img = {k: v[:, num_obs:] for k, v in data.items()}
    losses = {k: -v.log_prob(data_img[k].astype(f32)) for k, v in img.items()}
    metrics.update({f'openl_{k}_loss': v.mean() for k, v in losses.items()})
    stats = jaxutils.balance_stats(img['reward'], data_img['reward'], 0.1)
    metrics.update({f'openl_reward_{k}': v for k, v in stats.items()})
    stats = jaxutils.balance_stats(img['cont'], data_img['cont'], 0.5)
    metrics.update({f'openl_cont_{k}': v for k, v in stats.items()})

    # Video predictions
    for key in self.dec.imgkeys:
      true = f32(data[key][:6])
      pred = jnp.concatenate([rec[key].mode()[:6], img[key].mode()[:6]], 1)
      error = (pred - true + 1) / 2
      video = jnp.concatenate([true, pred, error], 2)
      metrics[f'openloop/{key}'] = jaxutils.video_grid(video)

    # Grad norms per loss term
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              data, carry, update=False)[1][0][f'{key}_loss'].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    return metrics, carry_out

  def preprocess(self, obs):
    spaces = {**self.obs_space, **self.act_space, **self.aux_spaces}
    result = {}
    for key, value in obs.items():
      if key.startswith('log_') or key in ('reset', 'key', 'id'):
        continue
      space = spaces[key]
      if len(space.shape) >= 3 and space.dtype == jnp.uint8:
        value = cast(value) / 255.0
      result[key] = value
    result['cont'] = 1.0 - f32(result['is_terminal'])
    return result
  
  # def VLM_response(self, obs):
  #   #tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)
  #   conversation_mode = "v1"
  #   temperature = 0.2
  #   top_p = None
  #   num_beams = 1
  #   max_new_tokens = 512


  #   images = [Image.open(obs).convert("RGB")]
  #   images_tensor = process_images(images, self.image_processor, self.VLM_model.config).to(self.VLM_model.device, dtype=torch.float16)

  #   conv = conv_templates[conversation_mode].copy()
  #   conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + self.prompt)
  #   conv.append_message(conv.roles[1], None)
  #   prompt = conv.get_prompt()
  #   stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
  #   # Input
  #   input_ids = (tokenizer_image_token(prompt, self.VLM_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
  #   stopping_criteria = KeywordsStoppingCriteria([stop_str], self.VLM_tokenizer, input_ids)
  #   # Inference
  #   with torch.inference_mode():
  #       output_ids = self.VLM_model.generate(
  #           input_ids,
  #           images=images_tensor,
  #           do_sample=True if temperature > 0 else False,
  #           temperature=temperature,
  #           top_p=top_p,
  #           num_beams=num_beams,
  #           max_new_tokens=max_new_tokens,
  #           use_cache=True,
  #           stopping_criteria=[stopping_criteria],
  #       )
  #   # Result-Decode
  #   input_token_len = input_ids.shape[1]
  #   n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
  #   if n_diff_input_output > 0:
  #       print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
  #   outputs = self.VLM_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
  #   outputs = outputs.strip()
  #   if outputs.endswith(stop_str):
  #       outputs = outputs[: -len(stop_str)]
  #   print(f"🚀 {self.model_name}: {outputs.strip()}\n")