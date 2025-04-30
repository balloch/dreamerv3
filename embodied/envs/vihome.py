from dataclasses import dataclass
from random import random
from virtualhome.simulation.environment.unity_environment import UnityEnvironment
import numpy as np
import embodied
import abc
from pathlib import Path

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
DISTANCE_THRESHOLD = 0.5

EXEC_DIR = str(Path(__file__).parent.parent.parent.parent / "linux_exec_v2.2.4" / "linux_exec_v2.2.4.x86_64")

f = open("rewards.txt", "w")

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
    size: tuple = (POOL_SIZE, POOL_SIZE, 3)

def _average_pool_image(image, pool_size = CROP_SIZE // POOL_SIZE):
    dtype = image.dtype
    image = image.reshape(
        image.shape[0] // pool_size, pool_size,
        image.shape[1] // pool_size, pool_size, 3
    )
    image = image.mean(axis=(1, 3))
    return image.astype(dtype)

def _get_room(obs):
    for e in obs["edges"]:
        if e["relation_type"] == "INSIDE" and e["from_id"] == AGENT_ID + 1:
            return e["to_id"]
    return random.choice(ROOMS.keys())

class BaseVirtualHome(embodied.Env, abc.ABC):
    def __init__(self, config: VirtualHomeConfig):
        print(EXEC_DIR)
        self._config = config
        self._env = UnityEnvironment(
            base_port=config.base_port,
            port_id=config.port_id,
            use_editor=config.use_editor,
            num_agents=config.num_agents,
            observation_types=[config.obs_type] * config.num_agents,
            seed=42,
            executable_args={
                # "file_name": EXEC_DIR,
                # "no_graphics": True,
            }
        )
        self.episode = 0

    # Be wary of the raw environment as it is not fully implemented
    @property
    def env(self):
        return self._env

    @property
    def obs_space(self):
        return {
            "image": embodied.Space(np.uint8, shape=self._config.size),
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            "action": embodied.Space(np.int32, (), 0, NUM_ACTIONS),
            "reset": embodied.Space(bool),
        }

    def _obs(self, obs, reward, is_first, is_last, is_terminal):
        f.write(f"reward: {reward}, is_first: {is_first}, is_last: {is_last}, is_terminal: {is_terminal}\n")
        x1, y1 = (obs.shape[0] - CROP_SIZE) // 2, (obs.shape[1] - CROP_SIZE) // 2
        x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE
        return {
            "image": _average_pool_image(obs[x1:x2, y1:y2, :].astype(np.uint8)),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def close(self):
        self._env.close()
    
    @abc.abstractmethod
    def reward(self, obs):
        pass

    @abc.abstractmethod
    def step(self, action_dict):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

class GoToKitchen(BaseVirtualHome):

    INIT_ROOMS = ["livingroom", "bedroom", "bathroom"]

    def __init__(self, config: VirtualHomeConfig, destination: int = 205):
        super().__init__(config)
        self._env.reset(
            environment_id=self._config.env_id,
            init_rooms=self.INIT_ROOMS
        )
        self.destination = destination

    def reward(self, obs):
        room = _get_room(obs)
        if room == self.destination:
            return 1, True
        return 0, False

    def step(self, action_dict):
        if action_dict["reset"]:
            return self.reset()
        self._env.step({0: ACTIONS[action_dict["action"]]})
        self.episode += 1
        graph, image = self._env.get_observation(AGENT_ID, "full"), self._env.get_observation(AGENT_ID, "image")
        reward, done = self.reward(graph)
        return self._obs(
            obs=image,
            reward=reward,
            is_first=False,
            is_last=False,
            is_terminal=done
        )

    def reset(self):
        self._env.reset(
            environment_id=self._config.env_id,
            init_rooms=self.INIT_ROOMS
        )
        self.episode = 0
        return self._obs(
            obs=self._env.get_observation(AGENT_ID, "image"),
            reward=0,
            is_first=True,
            is_last=False,
            is_terminal=False
        )

class LeaveRoom(BaseVirtualHome):
    def __init__(self, config: VirtualHomeConfig):
        super().__init__(config)
        self._env.reset(environment_id=self._config.env_id)
        self.state = {
            "room": _get_room(self._env.get_observation(AGENT_ID, "full"))
        }

    def reward(self, obs):
        current_room = _get_room(obs)
        if current_room != self.state["room"]:
            return 1, True
        return 0, False

    def step(self, action_dict):
        if action_dict["reset"]:
            return self.reset()
        self._env.step({0: ACTIONS[action_dict["action"]]})
        self.episode += 1
        graph, image = self._env.get_observation(AGENT_ID, "full"), self._env.get_observation(AGENT_ID, "image")
        reward, done = self.reward(graph)
        self.state["room"] = _get_room(graph)
        return self._obs(
            obs=image,
            reward=reward,
            is_first=False,
            is_last=False,
            is_terminal=done
        )

    def reset(self):
        self._env.reset(environment_id=self._config.env_id)
        self.state["room"] = _get_room(self._env.get_observation(AGENT_ID, "full"))
        self.episode = 0
        return self._obs(
            obs=self._env.get_observation(AGENT_ID, "image"),
            reward=0,
            is_first=True,
            is_last=False,
            is_terminal=False
        )

class SweepAllRooms(BaseVirtualHome):

    def __init__(self, config: VirtualHomeConfig):
        super().__init__(config)
        self._env.reset(environment_id=self._config.env_id)
        self.state = self.init_state()
    
    def init_state(self):
        graph = self._env.get_observation(AGENT_ID, "full")
        object_ids = set()
        for room in ROOMS.keys():
            object_ids.update(SweepAllRooms._objects_in_room(graph, room))
        return {
            "objects": {
                "left": object_ids,
                "completion": 0,
                "total": len(object_ids)
            }
        }

    def _objects_in_room(obs, room):
        object_ids = set()
        for edge in obs["edges"]:
            if edge["relation_type"] == "INSIDE" \
                and edge["to_id"] == room \
                and edge["from_id"] != AGENT_ID + 1:
                object_ids.add(edge["from_id"])
        return object_ids

    def reward(self, obs):
        for edge in obs["edges"]:
            if edge["relation_type"] == "CLOSE" \
                and edge["from_id"] == AGENT_ID + 1 \
                and edge["to_id"] in self.state["objects"]["left"]:
                self.state["objects"]["left"].remove(edge["to_id"])
                total = self.state["objects"]["total"]
                left = len(self.state["objects"]["left"])
                completion = (total - left) / total
                self.state["objects"]["completion"] = completion
        return self.state["objects"]["completion"], self.state["objects"]["completion"] >= 0.975
    
    def step(self, action_dict):
        if action_dict["reset"]:
            return self.reset()
        self._env.step({0: ACTIONS[action_dict["action"]]})
        self.episode += 1
        graph, image = self._env.get_observation(AGENT_ID, "full"), self._env.get_observation(AGENT_ID, "image")
        reward, done = self.reward(graph)
        return self._obs(
            obs=image,
            reward=reward,
            is_first=False,
            is_last=False,
            is_terminal=done
        )

    def reset(self):
        self._env.reset(environment_id=self._config.env_id)
        self.state = self.init_state()
        self.episode = 0
        return self._obs(
            obs=self._env.get_observation(AGENT_ID, "image"),
            reward=0,
            is_first=True,
            is_last=False,
            is_terminal=False
        )

class ScanRooms(BaseVirtualHome):

    def __init__(self, config: VirtualHomeConfig):
        super().__init__(config)
        self._env.reset(environment_id=self._config.env_id)
        self.state = self.init_state()

    def init_state(self):
        return {
            "actions": []
        }
    
    def reward(self, obs):
        if self.state["actions"] == [0] * 12 or self.state["actions"] == [2] * 12:
            return 1, True
        i = 0
        clefts, mlefts = 0, 0
        crights, mrights = 0, 0
        while i < len(self.state["actions"]):
            if self.state["actions"][i] == 0:
                crights = 0
                clefts += 1
                mlefts = max(mlefts, clefts)
            elif self.state["actions"][i] == 2:
                clefts = 0
                crights += 1
                mrights = max(mrights, crights)
            else:
                clefts = 0
                crights = 0
            i += 1
        rotation = max(mlefts, mrights) / 12
        return rotation, False
    
    def step(self, action_dict):
        if action_dict["reset"]:
            return self.reset()
        self._env.step({0: ACTIONS[action_dict["action"]]})
        self.episode += 1
        self.state["actions"].append(action_dict["action"])
        if len(self.state["actions"]) > 12:
            self.state["actions"].pop(0)
        graph, image = self._env.get_observation(AGENT_ID, "full"), self._env.get_observation(AGENT_ID, "image")
        reward, done = self.reward(graph)
        return self._obs(
            obs=image,
            reward=reward,
            is_first=False,
            is_last=False,
            is_terminal=done
        )
    
    def reset(self):
        self._env.reset(environment_id=self._config.env_id)
        self.state = self.init_state()
        self.episode = 0
        return self._obs(
            obs=self._env.get_observation(AGENT_ID, "image"),
            reward=0,
            is_first=True,
            is_last=False,
            is_terminal=False
        )

class FindObject(BaseVirtualHome):

    def __init__(self, config: VirtualHomeConfig):
        super().__init__(config)
        self._env.reset(environment_id=self._config.env_id)
        self.rug_ids = FindObject._get_objects(self._env.get_observation(AGENT_ID, "full"), "rug")
    
    def _get_objects(self, obs, class_name):
        object_ids = set()
        for node in obs["nodes"]:
            if node["class_name"] == class_name:
                object_ids.add(node["id"])
        return object_ids
    
    def reward(self, obs):
        for edge in obs["edges"]:
            if edge["relation_type"] == "ON" \
                and edge["from_id"] == AGENT_ID + 1 \
                and edge["to_id"] in self.rug_ids:
                return 1, True
        return 0, False
    
    def step(self, action_dict):
        if action_dict["reset"]:
            return self.reset()
        self._env.step({0: ACTIONS[action_dict["action"]]})
        self.episode += 1
        graph, image = self._env.get_observation(AGENT_ID, "full"), self._env.get_observation(AGENT_ID, "image")
        reward, done = self.reward(graph)
        return self._obs(
            obs=image,
            reward=reward,
            is_first=False,
            is_last=False,
            is_terminal=done
        )
    
    def reset(self):
        self._env.reset(environment_id=self._config.env_id)
        self.rug_ids = FindObject._get_objects(self._env.get_observation(AGENT_ID, "full"), "rug")
        self.episode = 0
        return self._obs(
            obs=self._env.get_observation(AGENT_ID, "image"),
            reward=0,
            is_first=True,
            is_last=False,
            is_terminal=False
        )


# SENSOR CHANGES
# Randomly dropping frames to the agent
# Gaussian noise (across all sensors)
# Gaussian noise (across all sensors) + random dropping frames

# SENSORS
# Averaging pooled image (control)
# Max pool image
# First person POV

# Need to show that performance increases when sensor is affected
