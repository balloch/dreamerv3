"""Chase Nav."""

from typing import Any, Dict, List, Tuple

import numpy as np
from panda3d.core import LPoint3f, LVecBase3f
from stable_baselines3 import PPO

from blimp_env.tasks.into_the_fire_basic_nav import action_int_to_vec, do_run
from blimp_env.tasks.follow_task import FollowTaskEnv


class NPCMgr:
    """Container for movable objects."""

    def __init__(self):
        """Construct NPC Manager."""
        self.npcs = []

    def add_npc(self, npc):
        """Track an NPC for movement."""
        self.npcs.append(npc)

    def remove_npc(self, id_):
        """Untrack an NPC."""
        found = False
        for idx, npc in enumerate(self.npcs):
            if npc.id == id_:
                found = True
                break
        if found:
            del self.npcs[idx]

    def step(self, blimp_pos: np.array):
        """Execute movement of all NPCs."""
        for npc in self.npcs:
            npc.step(blimp_pos=blimp_pos)

    def reset(self):
        """Clear all previous NPCs."""
        self.npcs = []


class NPC:
    """Movable object."""

    def __init__(self, id_, obj, movement):
        """Construct NPC fire or balloon."""
        self.id = id_
        self.obj = obj
        self.movement = movement

    def step(self, blimp_pos: np.array):
        """Execute movement for an NPC."""
        pos = self.obj.get_pos()
        rot = self.obj.get_node().get_hpr()
        new_pos, new_hpr = self.movement.move(pos=pos, hpr=rot, blimp=blimp_pos)
        self.obj.get_node().set_pos(new_pos)
        self.obj.get_node().set_hpr(new_hpr)


class Movement:
    """Bouncy way for an NPC to move."""

    UPPER = [27, 17, 17]
    LOWER = [-27, -17, 3]

    def __init__(self, scale=0.05):
        """Configure speed."""
        self.scale = scale
        self.direction = [1, 1, 1]

    def move(self, pos: LPoint3f, hpr: LVecBase3f, blimp: Tuple[float, float, float]):
        """Move according to policy."""
        for idx in range(3):
            if pos[idx] >= self.UPPER[idx]:
                self.direction[idx] = -1
            if pos[idx] <= self.LOWER[idx]:
                self.direction[idx] = 1
        delta = tuple([self.scale * direction for direction in self.direction])
        return pos + delta, hpr


class SmartMovement(Movement):
    """Model driven way for an NPC to move."""

    CENTER = LPoint3f(*[0, 0, 10])

    def __init__(self, path, normalizers):
        super().__init__(scale=0.25)
        self.model = PPO.load(path)
        self.normalizers = normalizers
        self.memory = 0

    def _make_obs(self, pos, hpr, blimp):
        pose = np.zeros(12)
        pose[6:9] = pos
        pose[9:12] = hpr
        sensor = np.zeros(6)
        sensor[0:3] = blimp - pos  # stay away from blimp
        # heat = LVecBase3f(*sensor[0:3]).length()
        # print(heat)
        if (
                pos[0] >= Movement.UPPER[0] or
                pos[1] >= Movement.UPPER[1] or
                pos[2] >= Movement.UPPER[2] or
                pos[0] <= Movement.LOWER[0] or
                pos[1] <= Movement.LOWER[1] or
                pos[2] <= Movement.LOWER[2]
        ):
            self.memory = 100
        if self.memory > 0:
            self.memory -= 1
            sensor[3:6] = self.CENTER - pos  # go to center
        return {
            "sensor": sensor,
            "pose": pose,
            # "img": None,
        }

    def _action_to_delta(self, action: int, yaw: float) -> np.array:
        vec = action_int_to_vec(int(action))  # [forward, 0, z, yaw]
        scaled = self.scale * vec
        dxy, _, dz, dh = scaled
        dx = dxy * np.cos(np.deg2rad(yaw))
        dy = dxy * np.sin(np.deg2rad(yaw))
        return np.array([dx, dy, dz, dh])

    def _safe_move(self, pos, delta):
        return LVecBase3f(*(pos + delta).tolist())

    def move(self, pos, hpr, blimp):
        """Move according to policy."""
        obs = self._make_obs(pos, hpr, blimp)
        yaw = obs["pose"][11]
        for normalizer in self.normalizers:
            obs = normalizer(obs)
        action, _ = self.model.predict(obs)
        # print(f"Balloon:  S: {[round(x) for x in obs['sensor'].tolist()[0:3]]}  P: {[round(x) for x in obs['pose'].tolist()[6:9]]}  A: {action}")
        # print(f"Balloon: {[round(x, 5) for x in obs['pose'].tolist()[9:12]]} -> {action}")
        delta = self._action_to_delta(action, yaw)
        new_pos = self._safe_move(pos, delta[0:3])
        new_hpr = hpr + LVecBase3f(0, 0, delta[3])
        return new_pos, new_hpr


class ChaseTaskEnv(FollowTaskEnv):
    """Environment with information about rewards and hazards."""

    def __init__(self, *args, **kwargs):
        """Add NPCs to follow environment."""
        chase_config = kwargs.pop("chase_config", {})
        super().__init__(*args, **kwargs)
        self.chase_config = chase_config
        npc_mgr = NPCMgr()
        self.base.npc_mgr = npc_mgr

    def step(self, action=None) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step with hook for NPC movement."""
        self._set_action(action)
        action = self._get_action()
        self._apply_action(action)
        obs = self._get_obs()
        pos = obs["pose"][6:9]
        # rot = obs[9:12]
        self.base.npc_mgr.step(blimp_pos=np.array(pos))
        self._get_action_effect()

        self.step_num += 1
        self.base.step()

        obs, info = self._obs, self._info
        is_terminated = info['is_terminated']
        is_truncated = info['is_truncated']
        reward = info['reward']

        self.total_reward += reward

        if (self.step_num - 1) % self.log_interval == 0 or is_terminated or is_truncated:
            self.pprint(obs, action, reward, self.total_reward, is_terminated, is_truncated, info)

        return obs, reward, is_terminated, is_truncated, info

    def reset(self, *args, **kwargs):
        """Override reset to track balloons."""
        result = super().reset(*args, **kwargs)
        self.base.npc_mgr.reset()
        for id_, balloon in self.base.balloon_mgr.items():
            movement = SmartMovement(self.chase_config["agent_path"], self.chase_config["normalizers"])
            npc = NPC(id_, balloon, movement)
            self.base.npc_mgr.add_npc(npc)
        for id_, fire in self.base.fire_mgr.items():
            movement = Movement()
            npc = NPC(id_, fire, movement)
            self.base.npc_mgr.add_npc(npc)
        return result

    def collect_balloon(self, balloon_id):
        self.base.npc_mgr.remove_npc(balloon_id)
        return super().collect_balloon(balloon_id)

    def _apply_action(self, action: int) -> None:
        for t in range(self.action_repeat):
            if self.dynamics == "simple":
                self.simulate_simple(action, scale=0.05)
            elif self.dynamics == "physics":
                self._simulate_physics(action)


if __name__ == "__main__":
    # Use for manual control
    render_mode, disable_render, debug = "human", False, True
    # Use for RL agent
    # render_mode, disable_render, debug = "rgb_array", False, False

    env = ChaseTaskEnv(
        render_mode=render_mode,
        debug=debug,
        dynamics="simple",
        disable_render=disable_render,
        max_episode_steps=10000
    )
    do_run(env, render_mode)
