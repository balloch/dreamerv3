import embodied
import virtualhome
from virtualhome.simulation.unity_simulator import comm_unity
from virtualhome.simulation.environment.unity_environment import UnityEnvironment
import numpy as np

class VirtualHome(embodied.Env):
    def __init__(self, env_id, config, size=(64, 64), length=100):
        self._env = UnityEnvironment(**config)
        self.env_id = env_id
        self._env.reset(environment_id=env_id)
        self._size = size
        self._step = 0
        self._done = False
    
    @property
    def obs_space(self):
        return {
            'image': embodied.Space(np.uint8, self._size + (3,)),
            'vector': embodied.Space(np.float32, (7,)),
            'token': embodied.Space(np.int32, (), 0, 256),
            'step': embodied.Space(np.float32, (), 0, self._length),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }
    
    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.int32, (), 0, 5),
            'other': embodied.Space(np.float32, (6,)),
            'reset': embodied.Space(bool),
        }
    
    def step(self, action):
        if action['reset'] or self._done:
            self._step = 0
            self._done = False
            obs = self._env.reset()
        else:
            obs, reward, self._done, info = self._env.step(action['action'])
            self._step += 1
        return {
            'image': obs['image'],
            'vector': obs['vector'],
            'token': obs['token'],
            'step': np.float32(self._step),
            'reward': np.float32(reward),
            'is_first': self._step == 0,
            'is_last': self._done,
            'is_terminal': self._done,
        }
    
    def reset(self):
        self._step = 0
        self._done = False
        obs = self._env.reset()
        return {
            'image': obs['image'],
            'vector': obs['vector'],
            'token': obs['token'],
            'step': np.float32(self._step),
            'reward': np.float32(0),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }