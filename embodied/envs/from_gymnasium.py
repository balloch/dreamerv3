# from_gym.py adapted to work with Gymnasium. Differences:
#
# - gym.* -> gymnasium.*
# - Deals with .step() returning a tuple of (obs, reward, terminated, truncated,
#   info) rather than (obs, reward, done, info).
# - Also deals with .reset() returning a tuple of (obs, info) rather than just
#   obs.
# - Passes render_mode='rgb_array' to gymnasium.make() rather than .render().
# - A bunch of minor/irrelevant type checking changes that stopped pyright from
#   complaining (these have no functional purpose, I'm just a completionist who
#   doesn't like red squiggles).
import matplotlib.pyplot as plt
import functools
from typing import Any, Generic, TypeVar, Union, cast, Dict
import embodied
import gymnasium
import numpy as np
import time
U = TypeVar('U')
V = TypeVar('V')
class FromGymnasium(embodied.Env, Generic[U, V]):
  def __init__(self, env: Union[str, gymnasium.Env[U, V]], obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env: gymnasium.Env[U, V] = gymnasium.make(env, render_mode="rgb_array", **kwargs)
    else:
      assert not kwargs, kwargs
      assert env.render_mode == "rgb_array", f"render_mode must be rgb_array, got {self._env.render_mode}"
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None
    self.skill = None
    self.prev_obs = None
    self.side = -1
  @property
  def info(self):
    return self._info
  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      # cast is here to stop type checkers from complaining (we already check
      # that .spaces attr exists in __init__ as a proxy for the type check)
      obs_space = cast(gymnasium.spaces.Dict, self._env.observation_space)
      spaces = obs_space.spaces
    else:
      spaces = {self._obs_key: self._env.observation_space}

    spaces = {k: self._convert(v) for k, v in spaces.items()}
    print(spaces)
    return {
        **spaces,
        'avoid_reward': embodied.Space(np.float32, shape=()),
        'investigate_reward': embodied.Space(np.float32, shape=()),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }
  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      act_space = cast(gymnasium.spaces.Dict, self._env.action_space)
      spaces = act_space.spaces
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces
  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      # we don't bother setting ._info here because it gets set below, once we
      # take the next .step()
      obs, _ = self._env.reset()
      self.prev_obs = None
      # Swap the values of the two keys
      obs['sensor'], obs['ultra_sonic_sensor'] =  obs['ultra_sonic_sensor'], obs['sensor']
      # print(obs)
      return self._obs(obs, 0.0, 0.0, 0.0, is_first=True)
    if self._act_dict:
      gymnasium_action = cast(V, self._unflatten(action))
    else:
      gymnasium_action = cast(V, action[self._act_key])
    blimp_state = self._env.base.get_blimp_state()
    obs, reward, terminated, truncated, self._info = self._env.step(gymnasium_action)
    self._env.prev_obs = self.prev_obs
    # Swap the values of the two keys
    obs['sensor'], obs['ultra_sonic_sensor'] =  obs['ultra_sonic_sensor'], obs['sensor']

    investigate_reward = 0.0
    avoid_reward = 0.0
    total_dist = 0.0
    
    if self.prev_obs is None:
      if obs['ultra_sonic_sensor'][1] < obs['ultra_sonic_sensor'][4]: #Checking angle.
        object_of_interest = obs['ultra_sonic_sensor'][:3]
        obs['ultra_sonic_sensor'][3:] = obs['ultra_sonic_sensor'][3:]*0.0
        self.side = 0
      else:
        object_of_interest = obs['ultra_sonic_sensor'][3:]
        obs['ultra_sonic_sensor'][:3] = obs['ultra_sonic_sensor'][:3]*0.0
        self.side = 1
    else:
      if self.side == 0:
        object_of_interest = obs['ultra_sonic_sensor'][:3]
        obs['ultra_sonic_sensor'][3:] = obs['ultra_sonic_sensor'][3:]*0.0
      else:
        object_of_interest = obs['ultra_sonic_sensor'][3:]
        obs['ultra_sonic_sensor'][:3] = obs['ultra_sonic_sensor'][:3]*0.0
    
    # Extract sensor values
    x_distance, angle, z_distance = object_of_interest
    
    # Reward for reducing the distance to the object (x and z axes combined)
    distance = (x_distance**2 + z_distance**2)**0.5  # Euclidean distance

    if self.prev_obs is not None:
        prev_distance = (self.prev_obs[0]**2 + self.prev_obs[2]**2)**0.5
        distance_change = prev_distance - distance  # Positive if getting closer
        angle_change = abs(angle) - abs(self.prev_obs[1]) 
        self.prev_obs = object_of_interest
    else:
        self.prev_obs = object_of_interest
        distance_change = 0  # No previous observation
        angle_change = 0
        # print(angle)
        # print(angle_change)
        # Penalize large angles (encourage alignment toward the object)

    angle_bonus = angle_change / 180 # Normalize to [0, 1]

    # Calculate total investigate reward
    investigate_reward = 0
    investigate_reward += 1 * distance_change  # Strong reward for reducing distance
    investigate_reward -= .2 * angle_bonus    # Penalize misalignment
    investigate_reward = max(-5, avoid_reward)
    # Bonus for being very close to the object
    investigate_reward = 0
    if distance < 6:  # Within a threshold (e.g., 0.5 units)
        investigate_reward = 1#+= 2

    # Penalize being too close to the object
    proximity_penalty = max(0, 5 - distance)  # Strong penalty if distance < 1

    # Calculate total reward
    avoid_reward = 0
    avoid_reward -= 1 * distance_change  # Strong reward for increasing distance
    avoid_reward -= 5 * proximity_penalty  # Strong penalty for being too close
    avoid_reward += .2 * angle_bonus        # Bonus for avoiding alignment
    avoid_reward = min(5, avoid_reward)
    

    # print(investigate_reward)
    if reward < -1:
      investigate_reward = reward
      avoid_reward = reward

    visualize = False
    act_taken = f"Action: {action['action']}"
    if visualize:
      plt.imshow(obs['img'])
      plt.title(act_taken)
      plt.show(block=False)
      plt.pause(.01)  # Pause to ensure the plot updates
      time.sleep(.01)
      plt.clf()  # Clear the plot so that the next image replaces this one


    self._done = terminated or truncated
    return self._obs(
        obs, reward, avoid_reward, investigate_reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))
  def _obs(
      self, obs, reward, avoid_reward, investigate_reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    np_obs: Dict[str, Any] = {k: np.asarray(v) for k, v in obs.items()}
    np_obs.update(
        reward=np.float32(reward),
        avoid_reward=np.float32(avoid_reward),
        investigate_reward=np.float32(investigate_reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    
    
    
    return np_obs
  def render(self):
    image = self._env.render()
    assert image is not None
    return image
  def close(self):
    try:
      self._env.close()
    except Exception:
      pass
  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gymnasium.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result
  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result
  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
