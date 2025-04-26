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
import functools
from typing import Any, Generic, TypeVar, Union, cast, Dict
import embodied
import gymnasium
import numpy as np
import time
import os
import matplotlib.pyplot as plt
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
    self.prev_obs = None
    self.prev_side = 0
    self.wrapper_time_step = 0
    self.wrapper_episode_step = 0
    self.steps_since_last_look = 0
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
        'pose': embodied.Space(np.float32, (12,), -180.0, 180.0),
        'sensor': embodied.Space(np.float32, (3,), -180.0, 180.0), #x, angle, z, x, angle, z.
        'ultra_sonic_sensor': embodied.Space(np.float32, (3,), -180.0, 180.0),
        'gray_scale_img': embodied.Space(dtype=np.uint8, shape=(64, 64, 3), low=0, high=255),
        'reward': embodied.Space(np.float32),
        'avoid_reward': embodied.Space(np.float32, shape=()),
        'investigate_reward': embodied.Space(np.float32, shape=()),
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
      self.wrapper_time_step = 0
      self.wrapper_episode_step +=1
      self.steps_since_last_look = 0
      # we don't bother setting ._info here because it gets set below, once we
      # take the next .step()
      obs, _ = self._env.reset()
      obs['ultra_sonic_sensor'] = obs['sensor']
      blimp_state = self._env.base.sensor_mgr.sensors[0].nearby_node = None # Geigh Zollicoffer
      return self._obs(obs, 0.0, 0.0, 0.0, is_first=True)
    if self._act_dict:
      gymnasium_action = cast(V, self._unflatten(action))
    else:
      self.wrapper_time_step +=1
      gymnasium_action = cast(V, action[self._act_key])
    obs, reward, terminated, truncated, self._info = self._env.step(gymnasium_action)
    self._done = terminated or truncated
    ## COMPUTE REWARD HERE ##
    obs['sensor'] = np.round(obs['sensor'], 2)
    obs['ultra_sonic_sensor'] = obs['sensor']
    investigate_reward = 0.0
    avoid_reward = 0.0
    total_dist = 0.0
    object_of_interest = obs['sensor']
    side = 0
     
    # if abs(obs['sensor'][1]) < abs(obs['sensor'][4]):  # Checking angle
    #   object_of_interest = obs['sensor'][:3]
    #   side = 0
    # else:
    #   object_of_interest = obs['sensor'][3:]
    #   side = 1


    # Extract sensor values
    x_distance, angle, z_distance = object_of_interest

    # Reward for reducing the distance to the object (x and z axes combined)
    distance = (x_distance ** 2 + z_distance ** 2) ** 0.5  # Euclidean distance

    if self.prev_obs is not None:
      if self.prev_side == 0:
        prev_distance = (self.prev_obs[0] ** 2 + self.prev_obs[2] ** 2) ** 0.5
      else:
        prev_distance = (self.prev_obs[3] ** 2 + self.prev_obs[5] ** 2) ** 0.5
      distance_change = prev_distance - distance  # Positive if getting closer
      angle_change = abs(angle) - abs(self.prev_obs[1])
      self.prev_obs = obs['sensor']#object_of_interest
      self.prev_side = side
    else:
      self.prev_obs = obs['sensor'] #object_of_interest
      self.prev_side = side
      distance_change = 0  # No previous observation
      angle_change = 0

    # Penalize large angles (encourage alignment toward the object)
    angle_bonus = angle_change / 180  # Normalize to [0, 1]

    # Calculate total investigate reward
    investigate_reward = 0
    investigate_reward += .001 * distance_change  # Strong reward for reducing distance
    investigate_reward -= 0.2 * angle_bonus    # Penalize misalignment
    # reward = investigate_reward
    # if distance < =8:
    #   reward = 20
    # investigate_reward = max(-5, investigate_reward)
    # Periodically reward looking back at the object
    time_since_last_look = self.steps_since_last_look
    looking_back = abs(angle) < 180  # Agent is looking at object if angle is small
    
    if looking_back:
        look_back_reward = min(.1, time_since_last_look * 0.01)  # Reward increases with time
        self.steps_since_last_look = 0
    else:
        look_back_reward = 0
        self.steps_since_last_look += 1
    # Penalize being too close to the object
    proximity_penalty = max(0, 5 - distance)  # Strong penalty if distance < 1

    # Calculate total avoid reward
    avoid_reward = 0
    avoid_reward += min(.05 * distance_change, 2)  # Reward for increasing distance
    # avoid_reward -= .05 * max(0, 1 - distance)  # Penalty for being too close
    avoid_reward += look_back_reward  # Reward for periodic looking back
    if abs(z_distance) > 7:
      avoid_reward -= .001 * abs(z_distance)
    if reward < -10:
      avoid_reward = -100
    reward = avoid_reward 
    # reward = avoid_reward
    # reward += .01
    ### END OF COMPUTE REWARD ###

    visualize = False
    act_taken = f"Action: {action['action']}"
    if visualize:
      plt.imshow(obs['img'])
      # Remove axes and ticks
      plt.axis('off')
      # plt.title(act_taken)
      # step_path = f'/frames/plot_{self.wrapper_time_step}_{self.wrapper_episode_step}.png'
      step_path = f'/logdir/pics/plot_avoid_eval_{self.wrapper_time_step}_{self.wrapper_episode_step}.png'
      full_path = os.path.expanduser(step_path)

      directory = os.path.dirname(full_path)
      os.makedirs(directory, exist_ok=True)

      plt.savefig(full_path)
    # print(avoid_reward)
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
        gray_scale_img=self.rgb_to_grayscale(obs['img']),
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

  def rgb_to_grayscale(self, image):
      """
      Converts a 64x64 RGB image (uint8) to grayscale.
      
      Parameters:
          image (np.ndarray): Input RGB image of shape (64, 64, 3) and dtype uint8.
      
      Returns:
          np.ndarray: Grayscale image of shape (64, 64) and dtype uint8.
      """
      if image.shape != (64, 64, 3) or image.dtype != np.uint8:
          raise ValueError("Input must be a (64, 64, 3) uint8 RGB image.")
      
      # Use the luminosity method for grayscale conversion
      grayscale = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
      grayscale = np.stack([grayscale]*3, axis=-1)
      return grayscale.astype(np.uint8)

  def save_image(self, image, filename="image.png"):
      """
      Saves an image (grayscale or RGB) to disk.
      
      Parameters:
          image (np.ndarray): Image to save. Can be grayscale (H, W) or RGB (H, W, 3).
          filename (str): Name of the file to save, including extension (e.g., 'output.png').
      """
      # Automatically set cmap for grayscale
      cmap = 'gray' if image.ndim == 2 else None
      
      # Ensure the directory exists
      os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(filename) else None
      
      plt.imsave(fname=filename, arr=image, cmap=cmap)
      print(f"Image saved as '{filename}'")