from dataclasses import dataclass
from virtualhome.simulation.environment.unity_environment import UnityEnvironment
import numpy as np
import embodied

# INIT_ROOMS
f1 = open("actions_taken.txt", "w")
f2 = open("reward_received.txt", "w")

NUM_ACTIONS = 3
ROOMS = {
    205: "kitchen",
    335: "livingroom",
    11: "bathroom",
    73: "bedroom"
}
AGENT_ID = 0
ACTIONS = {
    0: "[turnleft]",
    1: "[walkforward]",
    2: "[turnright]"
}
CROP_SIZE = 256
POOL_SIZE = 64

@dataclass
class VirtualHomeConfig:
    env_id: int = 0
    num_agents: int = 1
    max_episode_length: int = 200
    obs_type: str = 'image'

    # Modifies port and location where the environment is hosted
    base_port: int = 8080
    port_id: int = 0

    # Uses command line manually executed Unity environment if True
    use_editor: bool = False

    # Custom
    destination: int = 205

def _average_pool_image(image, pool_size = CROP_SIZE // POOL_SIZE):
    dtype = image.dtype
    image = image.reshape(
        image.shape[0] // pool_size, pool_size,
        image.shape[1] // pool_size, pool_size, 3
    )
    image = image.mean(axis=(1, 3))
    return image.astype(dtype)

class VirtualHome(embodied.Env):
    def __init__(self, config: VirtualHomeConfig, size=(POOL_SIZE, POOL_SIZE, 3), seed=None):
        self.config = config
        self.size = size

        self._env = UnityEnvironment(
            base_port=config.base_port,
            port_id=config.port_id,
            use_editor=config.use_editor,
            num_agents=config.num_agents,
            max_episode_length=config.max_episode_length,
            observation_types=[config.obs_type] * config.num_agents,
            seed=seed if not None else 123,
        )       
        self._env.reset(environment_id=config.env_id)

        self._episode = 0
        self._reward = 0
        self._done = False

        self.destination = config.destination

        print("Number of agents: ", self._env.num_agents)
        print("Number of cameras: ", self._env.num_static_cameras, self._env.num_camera_per_agent)
        
        import json
        json.dump(self._env.get_graph(), open("environment_graph.json", "w"))

    # Be wary of the raw environment as it is not fully implemented
    @property
    def env(self):
        return self._env

    @property
    def obs_space(self):
        return {
            "image": embodied.Space(np.uint8, shape=self.size),
            "reward": embodied.Space(np.float32),
            'avoid_reward': embodied.Space(np.float32),
            'investigate_reward': embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
            "ultra_sonic_sensor": embodied.Space(
                np.float32,
                (6,),
                low=0.0,
                high=1001.0,
            )
        }

    @property
    def act_space(self):
        return {
            "action": embodied.Space(np.int32, (), 0, NUM_ACTIONS),
            "reset": embodied.Space(bool),
        }
    
    def reward(self, obs):
        for e in obs["edges"]:
            if e["relation_type"] == "INSIDE" and \
               e["from_id"] == AGENT_ID + 1 and \
               e["to_id"] in ROOMS.keys():
                return 1, True
        return 0, False

    def step(self, action_dict):
        if action_dict["reset"]:
            return self.reset()
        action_dict = {0: ACTIONS[action_dict["action"]]}
        f1.write(f"{action_dict[0]}\n")
        self._env.step(action_dict)
        obs_full = self._env.get_observation(AGENT_ID, "full")
        reward, self._done = self.reward(obs_full)
        f2.write(f"{reward} {self._done}\n")
        obs_image = self._env.get_observation(AGENT_ID, "image")
        return self._obs(obs_image, reward, 0, 0, False, False, self._done)

    def _obs(self, obs, reward, avoid_reward, investigate_reward,
             is_first, is_last, is_terminal, ultra_sonic_sensor = np.zeros((6,))):
        x1, y1 = (obs.shape[0] - CROP_SIZE) // 2, (obs.shape[1] - CROP_SIZE) // 2
        x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE
        return {
            "image": _average_pool_image(obs[x1:x2, y1:y2, :].astype(np.uint8)),
            "reward": np.float32(reward),
            "avoid_reward": np.float32(avoid_reward),
            "investigate_reward": np.float32(investigate_reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "ultra_sonic_sensor": ultra_sonic_sensor
        }

    def reset(self, env_id=None):
        if env_id is not None:
            self.config.env_id = env_id
        self._env.reset(environment_id=self.config.env_id)
        self._done = False
        self._episode = 0
        return self._obs(self._env.get_observation(AGENT_ID, "image"), 0, 0, 0, True, False, False)

    def close(self):
        self._env.close()
