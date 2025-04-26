"""Follow Nav."""

import itertools
import math

import gymnasium as gym
import numpy as np
import panda3d.core
from panda3d.core import Vec3
from loguru import logger
from tabulate import tabulate
from typing import Tuple, List, Any, Dict
from copy import deepcopy
from blimp_env.tasks.into_the_fire_basic_nav import IntoTheFireBasicNavEnv, do_run

INF = float("inf")


class Sensor:
    """Track elements in the environment."""

    def __init__(self, feature_manager):
        """Set up fire or balloon tracking."""
        self.manager = feature_manager
        self._observation_range = None
        self.nearby_node = None

    def nearest_feature(self, node):
        """Return node of the nearest tracked feature."""
        pos = node.get_pos()
        nearest = None
        feature_ = None
        for _id, feature in self.manager.items():
            loc = feature.get_pos()
            dist = np.linalg.norm(pos - loc)
            if nearest is None or dist < nearest:
                nearest = dist
                feature_ = feature.get_node()
        if feature_ is None:
            logger.warning("Sensor is not tracking any items")
        return feature_

    def nearest_feature_front(self, node):
        """Return node of the nearest tracked feature in front that's not too high above."""
        # Get the position and forward direction of the node
        pos = node.get_pos()
        # Get the forward vector in the node's local space (typically +Y in Panda3D)
        forward_vec = node.get_quat().get_right()
        # Get the up vector (typically +Z in Panda3D)
        up_vec = node.get_quat().get_up()
        
        nearest = None
        feature_ = None
        
        # Configure these parameters based on your needs
        max_height_difference = 10.0  # Maximum acceptable height difference
        height_penalty_factor = 2.0  # Factor to penalize height differences when calculating effective distance
        
        for _id, feature in self.manager.items():
            loc = feature.get_pos()
            
            # Vector from node to feature
            to_feature = loc - pos
            
            # Calculate distance
            dist = np.linalg.norm(to_feature)
            
            # Calculate vertical difference (projection onto the up vector)
            height_diff = np.dot(to_feature, up_vec)
            
            # Normalize the vector
            if dist > 0:
                to_feature_normalized = to_feature / dist
                
                # Calculate dot product to check if feature is in front
                dot_product = np.dot(forward_vec, to_feature_normalized)
                
                # Only consider features that are in front (dot product > 0)
                # and not too high above (height_diff < max_height_difference)
                if dot_product > 0 and abs(height_diff) < max_height_difference:
                    # Calculate "effective distance" that penalizes height differences
                    effective_dist = dist + (abs(height_diff) * height_penalty_factor)
                    
                    if nearest is None or effective_dist < nearest:
                        nearest = effective_dist
                        feature_ = feature.get_node()
        # logger.warning("Sensor is tracking suitable items in front")
        if feature_ is None:
            logger.warning("Sensor is not tracking any suitable items in front")
        
        return feature_

    def _get_observation_range(self, boundary, boundary_scale):
        # set self._observation_range = Tuple[List[float], List[float]]
        raise NotImplementedError()

    def observe(self, node):
        """Take an observation measurement."""
        mins, maxes = self._observation_range
        return maxes


class PositionSensor(Sensor):
    """Track relative position of nearest feature."""

    def _get_observation_range(self, boundary, boundary_scale):
        x_delta = (boundary.x_max - boundary.x_min) * boundary_scale
        y_delta = (boundary.y_max - boundary.y_min) * boundary_scale
        z_delta = (boundary.z_max - boundary.z_min) * boundary_scale
        self._observation_range = (
            [-x_delta, -y_delta, -z_delta], [x_delta, y_delta, z_delta]
        )
        return self._observation_range

    def observe(self, node: panda3d.core.NodePath):
        """Sensor reading based on position."""
        node_ = self.nearest_feature(node)
        if node_:
            return [val for val in (node_.get_pos() - node.get_pos())]
        else:
            _mins, maxes = self._observation_range
            return maxes

class TrackSensor(Sensor):
    """Track bearing to nearest feature.

    0 - delta x, y
    1 - delta yaw
    2 - delta height
    """
    def _get_observation_range(self, boundary, boundary_scale):
        a_min, a_max = -180, 180
        d_min = 0
        d_max = math.sqrt(
            pow((boundary.x_max - boundary.x_min) * boundary_scale, 2) +
            pow((boundary.y_max - boundary.y_min) * boundary_scale, 2)
            # + pow((boundary.z_max - boundary.z_min) * boundary_scale, 2)
        )
        h_max = (boundary.z_max - boundary.z_min) * boundary_scale
        h_min = -h_max
        self._observation_range = [
            [d_min, a_min, h_min], [d_max, a_max, h_max]
        ]
        return self._observation_range

    def observe(self, node):
        """Sensor reading based on relative direction."""
        if self.nearby_node is None:
            self.nearby_node = deepcopy(self.nearest_feature_front(node))
        nearby_node = self.nearby_node
        if nearby_node:
            delta = nearby_node.get_pos() - node.get_pos()
            distance = delta.get_xy().length()
            height = delta.get_z()
            #direction = node.getQuat().getForward().get_xy() + panda3d.core.LVector2f(1, -1)
            #angle = direction.signedAngleDeg(delta.get_xy())

            direction = panda3d.core.LVector2f(0, 0).signed_angle_deg(delta.get_xy())
            angle = direction - node.get_h()
            if angle > 180:
                angle = 360 - angle
            elif angle < -180:
                angle = 360 + angle

            return [distance, angle, height]
        else:
            _mins, maxes = self._observation_range
            return maxes

class DirectionSensor(Sensor):
    """Track bearing to nearest feature.

    0 - delta x, y
    1 - delta yaw
    2 - delta height
    """
    # x, angle, z?

    def _get_observation_range(self, boundary, boundary_scale):
        a_min, a_max = -180, 180
        d_min = 0
        d_max = math.sqrt(
            pow((boundary.x_max - boundary.x_min) * boundary_scale, 2) +
            pow((boundary.y_max - boundary.y_min) * boundary_scale, 2)
            # + pow((boundary.z_max - boundary.z_min) * boundary_scale, 2)
        )
        h_max = (boundary.z_max - boundary.z_min) * boundary_scale
        h_min = -h_max
        self._observation_range = [
            [d_min, a_min, h_min], [d_max, a_max, h_max]
        ]
        return self._observation_range

    def observe(self, node):
        """Sensor reading based on relative direction."""
        nearby_node = self.nearest_feature(node)
        if nearby_node:
            delta = nearby_node.get_pos() - node.get_pos()
            distance = delta.get_xy().length()
            height = delta.get_z()
            #direction = node.getQuat().getForward().get_xy() + panda3d.core.LVector2f(1, -1)
            #angle = direction.signedAngleDeg(delta.get_xy())

            direction = panda3d.core.LVector2f(1, 0).signed_angle_deg(delta.get_xy())
            angle = direction - node.get_h()
            if angle > 180:
                angle = 360 - angle
            elif angle < -180:
                angle = 360 + angle

            return [distance, angle, height]
        else:
            _mins, maxes = self._observation_range
            return maxes


class SensorManager:
    """Manage all the sensors."""

    def __init__(self, boundary, boundary_scale=0.95, dtype=np.float32):
        """Set up baseline environment properties."""
        self.boundary = boundary
        self.boundary_scale = boundary_scale
        self.dtype = dtype
        self.sensors = []

    def add_sensor(self, sensor):
        """Track a sensor."""
        self.sensors.append(sensor)

    def get_observation_space(self):
        """Derive obs space for tracked sensors."""
        mins = []
        maxes = []
        for sensor in self.sensors:
            sensor_mins, sensor_maxes = sensor._get_observation_range(
                self.boundary, self.boundary_scale
            )
            mins.extend(sensor_mins)
            maxes.extend(sensor_maxes)
        sensor_space = gym.spaces.Box(
            low=np.array(mins),
            high=np.array(maxes),
            shape=(len(mins), ),
            dtype=self.dtype,
        )
        return sensor_space

    def observe(self, node):
        """Get sensor readings for pose."""
        return np.array(
            list(itertools.chain(*[sensor.observe(node) for sensor in self.sensors])),
            dtype=self.dtype,
        )


class FollowTaskEnv(IntoTheFireBasicNavEnv):
    """Environment with information about rewards and hazards."""

    def __init__(self, *args, **kwargs):
        """Do setup of environment with sensors."""
        super().__init__(*args, **kwargs)
        sensor_mgr = SensorManager(self.base.boundary)
        fire_sensor = DirectionSensor(self.base.fire_mgr)
        balloon_sensor = DirectionSensor(self.base.balloon_mgr)
        # object_mgr = self.base.object_mgr#{**dict(self.base.fire_mgr._fires.items()), **dict(self.base.balloon_mgr._balloons.items())}
        front_sensor = TrackSensor(self.base.object_mgr)
        # sensor_mgr.add_sensor(fire_sensor)
        # sensor_mgr.add_sensor(balloon_sensor)
        sensor_mgr.add_sensor(front_sensor)
        self.base.sensor_mgr = sensor_mgr
        self.observation_space = self._augment_observation_space(self.observation_space)


    @staticmethod
    def pprint(obs, action, reward, total_reward, is_terminated, is_truncated, info):
        """Pretty-print the environment state to the console."""
        step_num = info['step_num']
        x, y, z = obs["pose"][6:9]
        roll, pitch, yaw = obs["pose"][9:12]
        vx, vy, vz = obs["pose"][0:3]
        v_roll, v_pitch, v_yaw = obs["pose"][3:6]
        num_balloons_collected = info['num_balloons_collected']
        # fire_a, fire_b, fire_c, bln_a, bln_b, bln_c = obs["sensor"]
        fire_a, fire_b, fire_c = obs["sensor"]
        bln_a, bln_b, bln_c = obs["sensor"]
        in_fire_zone = info['in_fire_zone']
        in_reward_zone = info['in_reward_zone']

        data = [
            step_num,
            f'{action}',#str(int(action)),
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            # vx,
            # vy,
            # vz,
            # v_roll,
            # v_pitch,
            # v_yaw,
            reward,
            total_reward,
            in_fire_zone,
            in_reward_zone,
            num_balloons_collected,
            # is_terminated,
            # is_truncated,
            fire_a,
            fire_b,
            fire_c,
            bln_a,
            bln_b,
            bln_c,
        ]
        headers = [
            'step',
            'last_action',
            'x',
            'y',
            'z',
            'roll',
            'pitch',
            'yaw',
            # 'vx',
            # 'vy',
            # 'vz',
            # 'v_roll',
            # 'v_pitch',
            # 'v_yaw',
            'reward',
            'total_reward',
            'in_fire_zone',
            'in_reward_zone',
            '# collected',
            # 'is_terminated',
            # 'is_truncated',
            'fire xy',
            'fire angle',
            'fire z',
            'bln xy',
            'bln angle',
            'bln z',
        ]
        print(
            tabulate(
                [data], headers=headers, tablefmt='fancy_grid', floatfmt='.4f'
            ), flush=True)

    def _augment_observation_space(self, observation_space):
        # Cannot override _setup_observation_space if we want sensor_mgr on base
        observation_space["sensor"] = self.base.sensor_mgr.get_observation_space()
        return observation_space

    def _get_obs(self):
        obs = super()._get_obs()
        obs["sensor"] = self.base.sensor_mgr.observe(self.base.blimp)
        self._obs = obs
        return obs
    


if __name__ == "__main__":
    # Use for manual control
    render_mode, disable_render, debug = "human", False, True
    # Use for RL agent
    # render_mode, disable_render, debug = "rgb_array", False, False

    env = FollowTaskEnv(
        render_mode=render_mode,
        debug=debug,
        dynamics="simple",
        disable_render=disable_render,
        max_episode_steps=10000
    )
    do_run(env, render_mode)
