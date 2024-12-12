"""
TODO LIST:
- [x] Add physics-based dynamics
- [x] World
    - [X] Add in stadium model
    - [X] Add in fire/smoke animations
    - [X] Add boundaries/collision detection
    - [X] Fix collision detection
- [X] Reset method
    - [X] Add random initialization for the blimp's pose
    - [X] Add random initialization for the fire/smoke model
    - [X] Add method for clearing the scene graph when resetting
- [X] Add action space
- [X] Add observation space
- [X] Add 3D model for blimp
- [X] Termination condition
    - [X] Add max episode steps
    - [X] Add collision detection with stadium walls
- [X] Reward model
    - [X] Add -1 penalty for each timestep
    - [X] Add +10 reward when a balloon is collected (i.e. when the blimp is within a certain distance of the balloon)
    - [X] If all balloons are collected, then add reward of 1000 + (max_allowed_steps - current_step) and terminate the episode
    - [X] Add -1000 penalty and terminate the episode for touching the stadium/ground/ceiling
    - [X] Add -10 penalty every step inside the fire hazard danger zone
    - [X] Add -100 penalty and terminate the episode for touching the fire hazard (i.e. when the blimp is within a certain distance of the fire hazard)
- [X] Fix null image observations
- [X} Fix env reset bug
"""
import sys
import uuid
import tempfile
from threading import Lock
from dataclasses import dataclass
from typing import Tuple, List, Any, Dict
import math
import io
import cv2
import numpy as np
from PIL import Image
import gymnasium as gym
from loguru import logger
from direct.particles.ParticleEffect import ParticleEffect
from tabulate import tabulate
import panda3d.core
from panda3d.core import (
    NodePath,
    loadPrcFileData,
    Vec3,
    Vec4,
    TextureStage,
    TexGenAttrib,
    AmbientLight,
    GraphicsOutput,
    Texture,
    PointLight,
    CollisionPlane,
    Point3,
    CollisionNode,
    CollisionTraverser,
    CollisionHandlerPusher,
    CollisionSphere,
    Material,
    TransparencyAttrib,
    MemoryUsage
)
from direct.showbase.Loader import *
from direct.showbase.ShowBase import ShowBase, ClockObject

from blimp_env import settings as cfg
from blimp_env.dynamics import BlimpSim
import time


INF = float("inf")
DEFAULT_SCREEN_WIDTH = 224   # 600
DEFAULT_SCREEN_HEIGHT = 224  # 600
loadPrcFileData('', 'sync-video false')
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "win-size {} {}".format(DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT))
# graphicsWindow.setFilmSize(width, height)
# width, height = base.graphicsWindow.getFilmSize()


# <editor-fold desc="********** General Utilities **********">
@dataclass
class BoundingBox3D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


def clamp(value: float, min_: float, max_: float) -> float:
    """ Clamp a value between a min and max value. """
    return max(min_, min(value, max_))


# noinspection PyUnresolvedReferences
def load_model(model_path, texture_path: str = None, **kwargs) -> NodePath:
    """ Load the model and texture data. """
    scale_x, scale_y, scale_z = kwargs.get('scale', (1, 1, 1))
    r_x, r_y, r_z = kwargs.get('rotation', (0, 0, 0))
    x, y, z = kwargs.get('position', (0, 0, 0))
    targetObj = loader.loadModel(model_path)
    targetObj.setScale(scale_x, scale_y, scale_z)
    targetObj.setPos(x, y, z)
    targetObj.setHpr(r_x, r_y, r_z)

    if texture_path is not None:
        tex = loader.loadTexture(texture_path)
        targetObj.setTexture(tex)

    return targetObj


def find_intersection(plane1, plane2, plane3):
    # Assuming planes are given as tuples of (A, B, C, D)
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    A3, B3, C3, D3 = plane3

    # Create a matrix and vector from the coefficients
    matrix = np.array([[A1, B1, C1], [A2, B2, C2], [A3, B3, C3]])
    vector = -np.array([D1, D2, D3])

    # Solve the system of equations
    try:
        intersection_point = np.linalg.solve(matrix, vector)
        return tuple(intersection_point)
    except np.linalg.LinAlgError:
        # The planes do not meet at a single point
        return None


def action_int_to_vec(action: int) -> np.ndarray:
    if action == 0:
        return np.array([0, 0, 0, 0])
    elif action == 1:
        return np.array([1, 0, 0, 0])
    elif action == 2:
        return np.array([0, 0, 0, +1])
    elif action == 3:
        return np.array([0, 0, 0, -1])
    elif action == 4:
        return np.array([0, 0, +1, 0])
    elif action == 5:
        return np.array([0, 0, -1, 0])


def action_str_to_int(action: str) -> int:
    if action == "no-op":
        return 0
    elif action == "move-forward":
        return 1
    elif action == "turn-left":
        return 2
    elif action == "turn-right":
        return 3
    elif action == "move-up":
        return 4
    elif action == "move-down":
        return 5


class HumanControls:

    def __init__(self, base: ShowBase):
        self.key_map = {
            "no-op": False,
            "move-forward": False,
            "turn-left": False,
            "turn-right": False,
            "move-up": False,
            "move-down": False,
        }

        # base.accept("0", self.update_key_map, ["no-op", True])
        base.accept("1", self.update_key_map, ["move-forward", True])
        base.accept("1-up", self.update_key_map, ["move-forward", False])
        base.accept("2", self.update_key_map, ["turn-left", True])
        base.accept("2-up", self.update_key_map, ["turn-left", False])
        base.accept("3", self.update_key_map, ["turn-right", True])
        base.accept("3-up", self.update_key_map, ["turn-right", False])
        base.accept("4", self.update_key_map, ["move-up", True])
        base.accept("4-up", self.update_key_map, ["move-up", False])
        base.accept("5", self.update_key_map, ["move-down", True])
        base.accept("5-up", self.update_key_map, ["move-down", False])
        base.accept("escape", self._on_exit)

        self._action = 0

    @property
    def action(self):
        return self._action

    @staticmethod
    def _on_exit():
        logger.info("Received exit signal from user. Exiting...")
        sys.exit()

    def update_key_map(self, control_name, control_state):
        self.key_map[control_name] = control_state
        action = action_str_to_int("no-op")
        for name, state in self.key_map.items():
            if state:
                action = action_str_to_int(name)
                break
        self._action = action
        # print("Action:", action)
# </editor-fold>


# <editor-fold desc="********** Balloon/Fire Reward Object Stuff **********">
class FireParticleObject(object):
    """
    Class to encapsulate the fire/smoke particle effects.
    Basically keep any of the Panda3D-specific stuff here,
    and tie the animations to a single dummy node so that
    we can manipulate the fire/smoke as a single object.
    """

    # Note: It's a little ugly to have the particle effect configs defined
    # as class attributes rather than loading from .ptf files, but this
    # was the easiest way to allow dynamically modifying the paths
    # to the particle textures.
    FIRE_CONFIG = f"""
# Source https://github.com/panda3d/panda3d/tree/master/samples/particles
self.reset()
self.setPos(0.000, 0.000, 0.000)
self.setHpr(0.000, 0.000, 0.000)
self.setScale(1.000, 1.000, 1.000)
p0 = Particles.Particles('particles-1')
# Particles parameters
p0.setFactory("PointParticleFactory")
p0.setRenderer("SpriteParticleRenderer")
p0.setEmitter("DiscEmitter")
p0.setPoolSize(1024)
p0.setBirthRate(0.0200)
p0.setLitterSize(10)
p0.setLitterSpread(0)
p0.setSystemLifespan(1200.0000)
p0.setLocalVelocityFlag(1)
p0.setSystemGrowsOlderFlag(0)
# Factory parameters
p0.factory.setLifespanBase(0.5000)
p0.factory.setLifespanSpread(0.0000)
p0.factory.setMassBase(1.0000)
p0.factory.setMassSpread(0.0000)
p0.factory.setTerminalVelocityBase(400.0000)
p0.factory.setTerminalVelocitySpread(0.0000)
# Point factory parameters
# Renderer parameters
p0.renderer.setAlphaMode(BaseParticleRenderer.PRALPHAOUT)
p0.renderer.setUserAlpha(0.22)
# Sprite parameters
p0.renderer.setTexture(loader.loadTexture('{str(cfg.RESOURCES_DIR / 'animated/fire/sparkle.png')}'))
p0.renderer.setColor(LVector4(1.00, 1.00, 1.00, 1.00))
p0.renderer.setXScaleFlag(1)
p0.renderer.setYScaleFlag(1)
p0.renderer.setAnimAngleFlag(0)
p0.renderer.setInitialXScale(0.0050)
p0.renderer.setFinalXScale(0.0300)
p0.renderer.setInitialYScale(0.0100)
p0.renderer.setFinalYScale(0.0400)
p0.renderer.setNonanimatedTheta(0.0000)
p0.renderer.setAlphaBlendMethod(BaseParticleRenderer.PPNOBLEND)
p0.renderer.setAlphaDisable(0)
# Emitter parameters
p0.emitter.setEmissionType(BaseParticleEmitter.ETRADIATE)
p0.emitter.setAmplitude(1.0000)
p0.emitter.setAmplitudeSpread(0.0000)
p0.emitter.setOffsetForce(LVector3(0.0000, 0.0000, 3.0000))
p0.emitter.setExplicitLaunchVector(LVector3(1.0000, 0.0000, 0.0000))
p0.emitter.setRadiateOrigin(LPoint3(0.0000, 0.0000, 0.0000))
# Disc parameters
p0.emitter.setRadius(0.5000)
self.addParticles(p0)
    """

    SMOKE_CONFIG = f"""
self.reset()
self.setPos(0.000, 0.000, 3.000)
self.setHpr(0.000, 0.000, 0.000)
self.setScale(1.000, 1.000, 1.000)
p0 = Particles.Particles('particles-1')
# Particles parameters
p0.setFactory("PointParticleFactory")
p0.setRenderer("SpriteParticleRenderer")
p0.setEmitter("SphereSurfaceEmitter")
p0.setPoolSize(10000)
p0.setBirthRate(0.0500)
p0.setLitterSize(10)
p0.setLitterSpread(0)
p0.setSystemLifespan(0.0000)
p0.setLocalVelocityFlag(1)
p0.setSystemGrowsOlderFlag(0)
# Factory parameters
p0.factory.setLifespanBase(2.0000)
p0.factory.setLifespanSpread(0.2500)
p0.factory.setMassBase(2.0000)
p0.factory.setMassSpread(0.0100)
p0.factory.setTerminalVelocityBase(400.0000)
p0.factory.setTerminalVelocitySpread(0.0000)
# Point factory parameters
# Renderer parameters
p0.renderer.setAlphaMode(BaseParticleRenderer.PRALPHAOUT)
p0.renderer.setUserAlpha(0.13)
# Sprite parameters
p0.renderer.setTexture(loader.loadTexture('{str(cfg.RESOURCES_DIR / 'animated/fire/smoke.png')}'))
p0.renderer.setColor(LVector4(1.00, 1.00, 1.00, 1.00))
p0.renderer.setXScaleFlag(0)
p0.renderer.setYScaleFlag(0)
p0.renderer.setAnimAngleFlag(0)
p0.renderer.setInitialXScale(0.0100)
p0.renderer.setFinalXScale(0.0200)
p0.renderer.setInitialYScale(0.0200)
p0.renderer.setFinalYScale(0.300)
p0.renderer.setNonanimatedTheta(0.0000)
p0.renderer.setAlphaBlendMethod(BaseParticleRenderer.PPBLENDLINEAR)
p0.renderer.setAlphaDisable(0)
# Emitter parameters
p0.emitter.setEmissionType(BaseParticleEmitter.ETRADIATE)
p0.emitter.setAmplitude(1.0000)
p0.emitter.setAmplitudeSpread(0.0000)
p0.emitter.setOffsetForce(LVector3(0.0000, 0.0000, 0.0000))
p0.emitter.setExplicitLaunchVector(LVector3(1.0000, 0.0000, 0.0000))
p0.emitter.setRadiateOrigin(LPoint3(0.0000, 0.0000, 0.0000))
# Sphere Surface parameters
p0.emitter.setRadius(0.0100)
self.addParticles(p0)
f0 = ForceGroup.ForceGroup('gravity')
# Force parameters
self.addForceGroup(f0)
"""

    def __init__(self, danger_radius: float = 1.0, debug: bool = False):

        self.danger_radius = danger_radius
        self.debug = debug

        fire_id = str(uuid.uuid4())  # Generate a unique ID for the fire hazard)
        self._node = NodePath(fire_id)

        # self._setup_collisions()
        self._setup_animation()

        if self.debug:
            self._setup_danger_visualizer()

    def __del__(self):
        self.stop()

    def _setup_danger_visualizer(self):
        """ Use a semi-transparent sphere to visualize the danger radius of the fire hazard. """
        danger_radius_node = load_model("models/misc/sphere")
        danger_radius_node.reparent_to(self._node)
        danger_radius_node.set_scale(self.danger_radius)

        mat = Material()
        mat.set_diffuse(Vec4(1, 0, 0, 0.4))
        danger_radius_node.setMaterial(mat)
        danger_radius_node.setTransparency(TransparencyAttrib.MAlpha)

    def _setup_animation(self):

        self._fire = ParticleEffect('fire-particle')
        self._smoke = ParticleEffect('smoke-particle')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ptf', delete=True) as temp_ptf:
            temp_ptf.write(self.FIRE_CONFIG)
            temp_ptf.flush()
            temp_ptf.seek(0)
            temp_ptf_path = str(temp_ptf.name)
            self._fire.loadConfig(temp_ptf_path)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ptf', delete=True) as temp_ptf:
            temp_ptf.write(self.SMOKE_CONFIG)
            temp_ptf.flush()
            temp_ptf.seek(0)
            temp_ptf_path = str(temp_ptf.name)
            self._smoke.loadConfig(temp_ptf_path)

        self._fire.reparent_to(self._node)
        self._smoke.reparent_to(self._node)

    def _setup_collisions(self, debug: bool = False):
        """ Setup collision detection for the fire hazard.

        Not currently using this code, but leaving it in here in case we want to use it later.
        """
        # Attach the collision node to the blimp
        collision_sphere = CollisionSphere(0, 0, 0, self.danger_radius)
        collision_node = CollisionNode('fire-collision')
        collision_node.addSolid(collision_sphere)
        fire_collision_node = self._node.attachNewNode(collision_node)
        if debug: fire_collision_node.show()
        # self.handler.addCollider(fire_collision_node, self, parent.drive.node())
        # self.traverser.addCollider(fire_collision_node, self.handler)
        # fire_collision_node.reparent_to(parent)

    def get_pos(self) -> Tuple[float, float, float]:
        return self._node.get_pos()

    def move(self, location: Tuple[float, float, float]) -> None:
        self._node.set_pos(location)

    def reparent(self, parent: NodePath) -> None:
        self._node.reparent_to(parent)

    def start(self, parent: NodePath) -> None:
        self._fire.start(parent=self._node, renderParent=parent)
        self._smoke.start(parent=self._node, renderParent=parent)

    def stop(self) -> None:
        self._fire.disable()
        self._fire.cleanup()
        self._smoke.disable()
        self._smoke.cleanup()
        self._node.detach_node()
        self._node.remove_node()


class FireHazard:
    """
    Class to encapsulate the fire hazard setup and basic logic.
    """

    def __init__(self,
                 parent: NodePath,
                 fire_id: str,
                 location: Tuple[float, float, float],
                 danger_radius: float = 1.0,
                 debug: bool = False):

        self.fire_id = fire_id
        self.danger_radius = danger_radius
        self.fire_obj = FireParticleObject(danger_radius=danger_radius, debug=debug)
        self.fire_obj.reparent(parent)
        self.fire_obj.move(location)
        self.fire_obj.start(parent=parent)

    def get_pos(self) -> Tuple[float, float, float]:
        return self.fire_obj.get_pos()

    def get_node(self) -> NodePath:
        return self.fire_obj._node


class FireManager:
    """
    Class to manage the fire hazards in the scene.

    This class is responsible for creating, removing, and updating the fire hazards.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._fires = {}

    def items(self):
        return self._fires.items()

    def add_fire(self, fire_id: str, fire_hazard: FireHazard) -> None:
        self._fires[fire_id] = fire_hazard

    def remove_fire(self, fire_id: str):
        fire = self._fires.pop(fire_id, None)
        if fire:
            fire.get_node().removeNode()
            del fire

    def create_fires(self,
                     parent: NodePath,
                     locations: List[Tuple[float, float, float]],
                     danger_radius: float = 1.0) -> None:
        for loc in locations:
            fire_id = str(uuid.uuid4())
            fire_hazard = FireHazard(parent, fire_id, loc, danger_radius, debug=self.debug)
            self.add_fire(fire_id, fire_hazard)

    def extinguish_fires(self):
        fire_keys = list(self._fires.keys())
        while len(fire_keys) > 0:
            fire_id = fire_keys.pop()
            self.remove_fire(fire_id)


class BalloonObject(object):
    """
    Class to encapsulate the balloon object.
    Keep any of the low-level Panda3D-specific stuff here.
    """

    def __init__(self, reward_zone_radius: float, debug: bool = False):

        self.debug = debug
        self._node = NodePath(str(uuid.uuid4()))

        ballon_model = load_model(cfg.MODELS_DIR / 'misc/beach_ball/scene.gltf', cfg.MODELS_DIR / 'misc/beach_ball/textures/Scene_-_Root_baseColor.jpeg')
        ballon_model.reparent_to(self._node)

        if self.debug:
            self._setup_reward_visualizer(reward_zone_radius)

    def __del__(self):
        self.pop()

    def _setup_reward_visualizer(self, reward_zone_radius: float):
        """ Use a semi-transparent sphere to visualize the reward zone. """
        reward_zone_node = load_model("models/misc/sphere")
        reward_zone_node.reparent_to(self._node)
        reward_zone_node.set_scale(reward_zone_radius)

        mat = Material()
        # Make reward zone visualizer green-ish color
        mat.set_diffuse(Vec4(0, 1, 0, 0.4))
        reward_zone_node.setMaterial(mat)
        reward_zone_node.setTransparency(TransparencyAttrib.MAlpha)

    def get_pos(self) -> Tuple[float, float, float]:
        return self._node.get_pos()

    def pop(self) -> None:
        """ Pop the balloon (i.e. remove from the scene). """
        self._node.detach_node()
        self._node.remove_node()

    def move(self, location: Tuple[float, float, float]) -> None:
        self._node.set_pos(location)

    def reparent(self, parent: NodePath) -> None:
        self._node.reparent_to(parent)


class Balloon:
    """
    Class to encapsulate the balloon setup and basic logic.
    """

    def __init__(self,
                 parent: NodePath,
                 balloon_id: str,
                 location: Tuple[float, float, float],
                 reward_zone_radius: float = 1.0,
                 debug: bool = False):

        self.balloon_id = balloon_id
        self.reward_zone_radius = reward_zone_radius
        self.balloon_obj = BalloonObject(reward_zone_radius=reward_zone_radius, debug=debug)
        self.balloon_obj.reparent(parent)
        self.balloon_obj.move(location)

    def get_pos(self) -> Tuple[float, float, float]:
        return self.balloon_obj.get_pos()

    def get_node(self) -> NodePath:
        return self.balloon_obj._node


class BalloonManager:
    """
    Class to manage the balloons in the scene.

    This class is responsible for creating, removing, and updating the balloons.
    It implements the logic necessary to visually indicate when a balloon has been collected
    by the agent. We've also included some utility methods helpful for calculating the reward
    later on in the code.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._balloons = {}
        self._collected_balloons: List[str] = []    # List of collected balloon IDs

    @property
    def collected_balloons(self) -> List[str]:
        return self._collected_balloons

    @property
    def num_balloons(self) -> int:
        return len(self.list_balloons_ids())

    def items(self):
        return self._balloons.items()

    def list_balloons_ids(self) -> List[str]:
        return list(self._balloons.keys())

    def add_balloon(self, balloon_id: str, balloon: Balloon) -> None:
        self._balloons[balloon_id] = balloon

    def create_balloons(self,
                     parent: NodePath,
                     locations: List[Tuple[float, float, float]],
                     reward_zone_radius: float = 1.0) -> None:
        for loc in locations:
            balloon_id = str(uuid.uuid4())
            balloon = Balloon(parent, balloon_id, loc, reward_zone_radius, debug=self.debug)
            self.add_balloon(balloon_id, balloon)

    def get_balloon(self, balloon_id: str) -> Balloon:
        return self._balloons[balloon_id]

    def pop_balloon(self, balloon_id: str):
        balloon = self._balloons.pop(balloon_id, None)
        if balloon:
            balloon.get_node().removeNode()
            del balloon

    def pop_balloons(self):
        balloon_keys = list(self._balloons.keys())
        while len(balloon_keys) > 0:
            balloon_id = balloon_keys.pop()
            self.pop_balloon(balloon_id)

    def collect_balloon(self, balloon_id: str) -> None:
        self.pop_balloon(balloon_id)
        self._collected_balloons.append(balloon_id)

class Singleton:
    """
    A lock-protected singleton base class.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Another thread could have created the instance
                # before we acquired the lock. So check that the
                # instance is still nonexistent.
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance


class ShowBaseSingleton(Singleton, ShowBase):
    """
    A singleton ShowBase class.

    Panda3D is a pain in the butt when it comes to creating multiple instances of ShowBase,
    so we use this singleton pattern to ensure that only one instance of ShowBase is created.
    (this is necessary when trying to run multiple instances of the environment in parallel,
    e.g. when using Stable Baselines3 for training)

    """
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'initialized'):
            ShowBase.__init__(self, *args, **kwargs)
            self.initialized = True


class Base(ShowBaseSingleton):
    """
    Handles the basic setup of the environment.
    """

    debug = False
    window_type = 'onscreen'

    def __init__(self):

        ShowBaseSingleton.__init__(self, windowType=self.window_type)

        self.controls = HumanControls(base=self)

        self.tasks = []

        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        self.blimp = None
        self.fire_mgr = None
        self.balloon_mgr = None
        self._has_collided_with_wall = False
        self.npc_mgr = None

        self._setup()

    @property
    def screen_size(self) -> Tuple[int, int]:
        return self.win.get_y_size(), self.win.get_x_size()

    @property
    def has_collided_with_wall(self) -> bool:
        return self._has_collided_with_wall

    @property
    def action(self):
        return self.controls.action

    def _setup_world(self):
        """ Setup skybox, lighting, etc. """

        def create_light(type_: str,
                         color: Tuple[float, float, float, float],
                         pos: Tuple[float, float, float] = None,
                         hpr: Tuple[float, float, float] = None) -> None:
            if type_ == 'ambient':
                plight = AmbientLight(type_)
            elif type_ == 'plight':
                plight = PointLight(type_)
            else:
                raise ValueError(f'{type_} is not a valid light type.')
            plight.setColor(color)
            plnp = self.render.attachNewNode(plight)
            if pos is not None:
                plnp.setPos(pos)
            if hpr is not None:
                plnp.setHpr(hpr)
            self.render.setLight(plnp)

        # General setup stuff
        self.disableMouse()
        self.enableParticles()

        # Stadium setup
        self.stadium = load_model(cfg.MODELS_DIR / 'buildings/stadium/scene.gltf', scale=(1, 1, 1), rotation=(0, 90, 0))
        self.stadium.reparent_to(self.render)

        # self.test = self.loader.loadModel("models/environment")
        # self.test.reparentTo(self.render)

        # skybox
        skybox = self.loader.loadModel(cfg.RESOURCES_DIR / 'models/skybox.gltf')
        skybox.setTexGen(TextureStage.getDefault(), TexGenAttrib.MWorldPosition)
        skybox.setTexProjector(TextureStage.getDefault(), self.render, skybox)
        skybox.setTexScale(TextureStage.getDefault(), 1)
        skybox.setScale(100)
        # skybox.setHpr(0, -90, 0)

        tex = self.loader.loadCubeMap(cfg.RESOURCES_DIR / 'textures/s_#.jpg')
        skybox.setTexture(tex)
        skybox.reparentTo(self.render)
        self.render.setTwoSided(True)

        # Lights
        create_light('plight', (1.0, 1.0, 1.0, 1), (20, 20, 10))
        create_light('ambient', (0.7, 0.7, 0.7, 1))

        # Set up offscreen buffer for capturing images
        self.texture = Texture()

    def _setup_clock(self):
        self.clock.setMode(1)  # C++: M_NON_REAL_TIME
        self.clock.dt = 1.0 / 60
        self.clock = ClockObject.get_global_clock()
        self.dt = self.clock.getDt()

    def _setup_global_frame(self) -> NodePath:
        global_frame = self.loader.loadModel("zup-axis")
        global_frame.reparentTo(self.render)
        return global_frame

    # noinspection PyUnresolvedReferences
    def _setup_world_boundary(self):
        # Setup collision bounding box to prevent travel outside stadium
        # We do this before setting up the observation space so that we can leverage the collision bbox
        # for the valid range of the observation space
        # NOTE: This will NOT prevent collision with visible objects (e.g. the stadium); those sorts of calculations
        # are computationally expensive, so Pandas3D doesn't have an easy way of enabling them. It assumes reliance
        # on primitive collision shapes (e.g. spheres, boxes, planes, etc.)
        collision_ground_plane = CollisionPlane((Vec3(0, 0, 1), Point3(0, 0, 0)))  # xy plane
        collision_xz_plane_pos = CollisionPlane((Vec3(0, -1, 0), Point3(0, 20, 0)))  # xz plane
        collision_xz_plane_neg = CollisionPlane((Vec3(0, 1, 0), Point3(0, -20, 0)))  # xz plane
        collision_yz_plane_pos = CollisionPlane((Vec3(-1, 0, 0), Point3(30, 0, 0)))  # yz plane
        collision_yz_plane_neg = CollisionPlane((Vec3(1, 0, 0), Point3(-30, 0, 0)))  # yz plane
        collision_ceiling_plane = CollisionPlane((Vec3(0, 0, -1), Point3(0, 0, 20)))  # xy plane

        collision_node = CollisionNode('boundary')
        collision_node.addSolid(collision_ground_plane)
        collision_node.addSolid(collision_xz_plane_pos)
        collision_node.addSolid(collision_xz_plane_neg)
        collision_node.addSolid(collision_yz_plane_pos)
        collision_node.addSolid(collision_yz_plane_neg)
        collision_node.addSolid(collision_ceiling_plane)
        self.boundary_np = self.render.attachNewNode(collision_node)
        if self.debug: self.boundary_np.show()

        # Create a collision traverser and a collision handler
        self.traverser = CollisionTraverser()
        base.cTrav = self.traverser
        self.handler = CollisionHandlerPusher()
        if self.debug: self.traverser.showCollisions(self.render)

        # self.handler.addCollider(self.blimp.collision_node, self.blimp, self.render.drive.node())
        # self.traverser.addCollider(self.blimp.collision_node, self.handler)

        # Find vertices of the intersection of collision planes, so we can create the position space accordingly
        # Lower left corner of stadium boundaries
        v1 = find_intersection(collision_ground_plane.get_plane(), collision_xz_plane_pos.get_plane(),
                               collision_yz_plane_pos.get_plane())
        # Lower right corner of stadium boundaries
        v2 = find_intersection(collision_ground_plane.get_plane(), collision_xz_plane_neg.get_plane(),
                               collision_yz_plane_neg.get_plane())
        # Upper right corner of stadium boundaries
        v3 = find_intersection(collision_ceiling_plane.get_plane(), collision_xz_plane_neg.get_plane(),
                               collision_yz_plane_pos.get_plane())
        # Upper left corner of stadium boundaries
        v4 = find_intersection(collision_ceiling_plane.get_plane(), collision_xz_plane_pos.get_plane(),
                               collision_yz_plane_pos.get_plane())

        x_min, x_max = min(v1[0], v2[0], v3[0], v4[0]), max(v1[0], v2[0], v3[0], v4[0])
        y_min, y_max = min(v1[1], v2[1], v3[1], v4[1]), max(v1[1], v2[1], v3[1], v4[1])
        z_min, z_max = min(v1[2], v2[2], v3[2], v4[2]), max(v1[2], v2[2], v3[2], v4[2])

        bbox = BoundingBox3D(x_min, x_max, y_min, y_max, z_min, z_max)
        self.boundary = bbox

    def _setup_offscreen_buffer(self):
        self.texture = Texture()
        self.win.addRenderTexture(self.texture, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)

    def _setup(self):

        self._setup_world()
        self._setup_clock()
        if self.debug:
            self.global_frame = self._setup_global_frame()
        self._setup_world_boundary()
        if self.window_type != "none":
            self._setup_offscreen_buffer()

        self.fire_mgr = FireManager(debug=self.debug)
        self.balloon_mgr = BalloonManager(debug=self.debug)

    def add_task(self, task, name, extra_args=None):
        extra_args = [] if extra_args is None else extra_args
        task_ = self.task_mgr.add(task, name, extraArgs=extra_args)
        self.tasks.append(task_)

    def get_blimp_state(self) -> Dict[str, Any]:
        x, y, z = self.blimp.get_pos()
        h, p, r = self.blimp.get_hpr()
        roll, pitch, yaw = r, p, h
        velocity = self.velocity
        angular_velocity = self.angular_velocity
        current_pose = np.array([x, y, z, roll, pitch, yaw])
        return {
            "pose": current_pose,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
        }

    def move_blimp(self, dx, dy, dz):
        self.blimp.set_y(self.blimp, dy)
        self.blimp.set_x(self.blimp, dx)
        self.blimp.set_z(self.blimp, dz)
        self.velocity = np.array([dx, dy, dz]) / self.dt

    def rotate_blimp(self, droll, dpitch, dyaw):
        self.blimp.set_h(self.blimp, dyaw)
        self.blimp.set_p(self.blimp, dpitch)
        self.blimp.set_r(self.blimp, droll)
        self.angular_velocity = np.array([droll, dpitch, dyaw]) / self.dt

    def capture_screenshot(self, img_size: Tuple[int, int] = (DEFAULT_SCREEN_HEIGHT, DEFAULT_SCREEN_WIDTH)) -> np.ndarray:
        """
        Capture a screenshot from the offscreen buffer.

        Note: The weird resizing hack here is to avoid conflicts between Panda3D
        and the gym observation space. This method at least guarantees that the
        output from the game engine will always be of the global DEFAULT_SCREEN_WIDTH x DEFAULT_SCREEN_HEIGHT.

        """

        # Get screenshot from offscreen buffer
        tex = self.texture
        screen_width, screen_height = tex.get_x_size(), tex.get_y_size()

        target_height, target_width = img_size
        try:
            data = self.win.get_screenshot().get_ram_image_as("RGBA")
            # data = io.BytesIO(data).read() #Geigh Addition
            image = Image.frombytes(
                "RGBA",
                (screen_width, screen_height),   # Is this supposed to be (width, height) or (height, width)?
                data,
                "raw",
                "RGBA"
            )
            image = image.convert("RGB")
            image.thumbnail((target_width, target_height), Image.LANCZOS)
            image_np = np.array(image, dtype=np.uint8)
            image_np = np.flipud(image_np)
        except ValueError:
            logger.warning("Could not get screenshot from offscreen buffer.")
            image_np = np.zeros((target_width, target_height, 3), dtype=np.uint8)

        return image_np

    def _on_handle_into_wall_collision(self, entry):
        # managing the state of the collision flag. It will be reset when the environment is reset.
        self._has_collided_with_wall = True

    def _reset_blimp(self, pose: np.ndarray):
        # Create a sphere to represent the blimp
        self.blimp = self.loader.load_model("models/misc/sphere")
        # self.blimp = self.loader.load_model(cfg.MODELS_DIR / 'vehicles/blimp_02/scene.glb')

        # 3d model has weird alignment in blender. Hack job here to get it to align
        # with the forward direction of the agent.
        model_3d = self.loader.load_model(cfg.MODELS_DIR / 'vehicles/blimp_02/scene.glb')
        model_bounds = model_3d.getBounds().center
        model_com = model_3d.get_pos() + model_bounds
        xm, ym, zm = self.blimp.get_pos() - model_com
        ym = 0.25
        model_3d.set_pos(xm, ym, zm)
        model_3d.set_hpr(self.blimp.get_hpr())
        model_3d.set_p(90)
        model_3d.set_h(-25)
        model_3d.reparent_to(self.blimp)

        # Attach the collision node to the blimp
        blimp_radius = 1.5  # 1.0  # Adjust this value to match the size of your blimp
        collision_sphere = CollisionSphere(0, 0, 0, blimp_radius)
        collision_node = CollisionNode('blimp')
        collision_node.addSolid(collision_sphere)
        blimp_collision_node = self.render.attachNewNode(collision_node)
        if self.debug: blimp_collision_node.show()
        self.handler.addCollider(blimp_collision_node, self.blimp, self.drive.node())
        self.traverser.addCollider(blimp_collision_node, self.handler)

        self.handler.addInPattern('%fn-into-%in')
        # Note: Names below must match the names of the collision nodes, i.e. 'blimp' and 'boundary'
        self.accept('blimp-into-boundary', self._on_handle_into_wall_collision)

        blimp_collision_node.reparent_to(self.blimp)

        self.blimp.reparent_to(self.render)
        self.blimp.set_scale(0.5)  # Adjust the size of the sphere as needed

        # Initial state of the blimp (default to slight z-offset so we don't clip through floor)
        x, y, z = pose[:3]
        roll, pitch, yaw = pose[3:]
        self.move_blimp(x, y, z)
        self.rotate_blimp(roll, pitch, yaw)

        # Reset velocity and angular velocity
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        # Set up the camera to follow the blimp
        if self.window_type != "none":
            self.camera.reparent_to(self.blimp)
            self.camera.set_pos(-20, 0, 3)  # (0, 0, 0)      # (0, -10, 3) # Position the camera behind and above the blimp
            self.camera.setH(-90)

        self._has_collided_with_wall = False

    def _clear_scene(self):
        logger.info('Clearing scene graph...')
        if self.blimp:
            self.blimp.detach_node()
            self.blimp.remove_node()

        if self.fire_mgr:
            self.fire_mgr.extinguish_fires()

        if self.balloon_mgr:
            self.balloon_mgr.pop_balloons()

    def _reset_fire(self, locations: List[Tuple[float, float, float]], danger_radius: float) -> None:
        self.fire_mgr.create_fires(parent=self.render, locations=locations, danger_radius=danger_radius)

    def _reset_balloons(self, locations: List[Tuple[float, float, float]], reward_zone_radius: float) -> None:
        self.balloon_mgr._collected_balloons = []
        self.balloon_mgr.create_balloons(parent=self.render, locations=locations, reward_zone_radius=reward_zone_radius)

    def reset(self,
              pose: np.ndarray,
              fire_locations: List[Tuple[float, float, float]] | None = None,
              fire_danger_radius: float = 1.0,
              balloon_locations: List[Tuple[float, float, float]] | None = None,
              balloon_reward_zone_radius: float = 1.0) -> None:

        assert pose.shape[0] == 6, "Pose must be a 6-dimensional vector [x, y, z, roll, pitch, yaw]"

        self._clear_scene()

        # If no locations are provided, then assume user doesn't want any fire hazards
        fire_locations = [] if fire_locations is None else fire_locations

        self._reset_blimp(pose=pose)
        self._reset_fire(locations=fire_locations, danger_radius=fire_danger_radius)
        self._reset_balloons(locations=balloon_locations, reward_zone_radius=balloon_reward_zone_radius)

    def step(self):
        self.task_mgr.step()


class IntoTheFireBasicNavEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    reset_options = None

    def __init__(
            self,
            max_episode_steps=1000,
            render_mode='rgb_array',
            log_interval=1000,
            dynamics="physics",
            debug=False,
            disable_render=False,
            img_size=(DEFAULT_SCREEN_HEIGHT, DEFAULT_SCREEN_WIDTH),
            reward_unit=1E-3,
            action_repeat=10,
    ):
        """ Initialize the environment. """
        super().__init__()

        if not disable_render:
            assert render_mode in ["human", "rgb_array"], (
                "Invalid mode. Must be one of 'human' or 'rgb_array'. Got: {}".format(render_mode))

        assert dynamics in ["simple", "physics"], (
            "Invalid dynamics type. Must be one of 'simple' or 'physics'. Got: {}".format(dynamics))

        self.max_episode_steps = max_episode_steps
        self.step_num = 0
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.dynamics = dynamics
        self.debug = debug
        self.disable_render = disable_render
        self.img_size = img_size
        self.action_repeat = action_repeat

        # Configure and setup Panda3D ShowBase instance
        if render_mode == "rgb_array":
            loadPrcFileData("", "window-type offscreen")
            Base.window_type = 'offscreen'
        if render_mode is None:
            Base.window_type = "none"

        Base.debug = debug
        self.base = Base()
        self.blimp = BlimpSim()

        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(6)
        self.previous_dist_balloon = INF
        self.prev_bln_angle = INF
        self.signed_angle = 0
        self.last_dist = INF

        # Game state vars
        self.total_reward = 0
        self._action = 0
        self._obs = None
        self._info = None

        self.reward_unit = reward_unit
        self.reward_factor = 1.0

    @staticmethod
    def pprint(obs, action, reward, total_reward, is_terminated, is_truncated, info):
        """ Pretty-print the environment state to the console in a tabular format. """
        step_num = info['step_num']
        x, y, z = obs["pose"][6:9]
        roll, pitch, yaw = obs["pose"][9:12]
        vx, vy, vz = obs["pose"][0:3]
        v_roll, v_pitch, v_yaw = obs["pose"][3:6]
        num_balloons_collected = info['num_balloons_collected']

        in_fire_zone = info['in_fire_zone']
        in_reward_zone = info['in_reward_zone']

        data = [step_num, str(int(action)), x, y, z, roll, pitch, yaw, vx, vy, vz, v_roll, v_pitch, v_yaw, reward, int(round(total_reward)), in_fire_zone, in_reward_zone, num_balloons_collected, is_terminated, is_truncated]
        headers = ['step', 'last_action', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'v_roll', 'v_pitch', 'v_yaw', 'reward', 'total_reward', 'in_fire_zone', 'in_reward_zone', '# collected', 'is_terminated', 'is_truncated']
        print(tabulate([data], headers=headers, tablefmt='fancy_grid', floatfmt='.4f'), flush=True)

    def _setup_observation_space(self):

        # Spatial boundaries calculated from rough bbox of stadium model
        bbox = self.base.boundary
        roll_min, roll_max = -180, 180
        pitch_min, pitch_max = -180, 180
        yaw_min, yaw_max = -180, 180
        boundary_scale = 0.95   # Shrink the boundaries a bit to avoid clipping through, e.g. stadium seats, etc.

        # Setup observation space
        # We define the pose as a 6-dimensional vector [x, y, z, roll, pitch, yaw]
        # Task 2 needs a full 12-d state vector though that includes velocity and angular velocity,
        # so we'll adapt the observation space to include those as well
        vx_min = vy_min = vz_min = -100    # Arbitrary values
        vx_max = vy_max = vz_max = +100    # Arbitrary values
        v_roll_min = v_pitch_min = v_yaw_min = -180    # Arbitrary values
        v_roll_max = v_pitch_max = v_yaw_max = +180    # Arbitrary values
        pose_space = gym.spaces.Box(
            low=np.array([vx_min,
                          vy_min,
                          vz_min,
                          v_roll_min,
                          v_pitch_min,
                          v_yaw_min,
                          bbox.x_min * boundary_scale,
                          bbox.y_min * boundary_scale,
                          bbox.z_min * boundary_scale,
                          roll_min,
                          pitch_min,
                          yaw_min]),
            high=np.array([
                           vx_max,
                           vy_max,
                           vz_max,
                           v_roll_max,
                           v_pitch_max,
                           v_yaw_max,
                           bbox.x_max * boundary_scale,
                           bbox.y_max * boundary_scale,
                           bbox.z_max * boundary_scale,
                           roll_max,
                           pitch_max,
                           yaw_max]),
            shape=(12,),
            dtype=np.float32
        )

        # RGB image, with dimensions (height, width, 3)
        height, width = self.img_size
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8
        )

        ultra_sonic_sensor_space = gym.spaces.Box(
            low=0.0,
            high=1001.0,
            shape=(6,),
            dtype=np.float32
        )

        if not self.disable_render:
            observation_space = gym.spaces.Dict([
                ('img', image_space),
                ('pose', pose_space),
                ('ultra_sonic_sensor', ultra_sonic_sensor_space) ## (near_dist_f, near_dist_b, near_dist_r, near_dist_l, near_dist_a, near_dist_u)
            ])
        else:
            # dummy_img_space = gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
            observation_space = gym.spaces.Dict([
                # ('img', dummy_img_space),
                ('pose', pose_space),
                ('ultra_sonic_sensor', ultra_sonic_sensor_space) # (near_dist_f, near_dist_b, near_dist_r, near_dist_l, near_dist_a, near_dist_u)
            ])

        return observation_space

    def reset(self, *,
              seed: int | None = None,
              options: dict[str, Any] | None = None):
        """ Reset the environment.

        Follows the signature of the OpenAI Gym reset method.

        """
        super().reset(seed=seed)
        options_from_cls = self.reset_options or {}
        options = {} if options is None else options
        options = {**options_from_cls, **options}

        # Unpack options
        pose = options.get("pose", None)
        num_fires = options.get("num_fires", 1)
        fire_danger_radius = options.get("fire_danger_radius", 5)
        num_balloons = options.get("num_balloons", 1)
        balloon_reward_zone_radius = options.get("balloon_reward_zone_radius", 5)
        fire_locations = options.get("fire_locations", None)  # List of (x, y, z) tuples
        balloon_locations = options.get("balloon_locations", None)  # List of (x, y, z) tuples

        # Initialize blimp state
        # If pose isn't explicitly provided, initialize randomly
        if pose is None:
            x, y, _, _, _, yaw = self.observation_space.sample()["pose"][6:12]
            z = 3.0   # Start slightly above the ground to avoid clipping through the floor
            roll = pitch = 0
            pose = np.array([x, y, z, roll, pitch, yaw])
        elif isinstance(pose, (list, tuple)):
            pose = np.array(pose)

        pose = pose.astype(self.observation_space["pose"].dtype)
        # If 12-d pose vector is provided (e.g. from task 2), then extract the 6-d pose vector
        if pose.shape[0] == 12:
            pose = pose[:6]

        # Setup the fire hazard locations
        if fire_locations is None:
            fire_locations = []
            for _ in range(num_fires):
                x, y, z = self.observation_space.sample()["pose"][6:9]
                # z = 0.1     # Fires should be on the ground (assume floor is level)
                fire_locations.append((x, y, z))
        fire_locations = [Vec3(*loc) for loc in fire_locations]

        # Setup the balloon locations
        if balloon_locations is None:
            balloon_locations = []
            for _ in range(num_balloons):
                x, y, z = self.observation_space.sample()["pose"][6:9]
                # Balloons should be floating above the ground, we'll arbitrarily set the minimum height to 3
                z = max((3, z))
                balloon_locations.append((x, y, z))
        balloon_locations = [Vec3(*loc) for loc in balloon_locations]
        self.previous_dist_balloon = INF
        self.last_dist = INF



        # Reset the game state
        self.base.reset(
            pose=pose,
            fire_locations=fire_locations,
            fire_danger_radius=fire_danger_radius,
            balloon_locations=balloon_locations,
            balloon_reward_zone_radius=balloon_reward_zone_radius,
        )


        self.step_num = 0
        self.total_reward = 0
        
        obs = self._get_obs()
        info = {"step_num": self.step_num}

        return obs, info

    def _simulate_simple(self, action, scale=0.5):
        """ Simulate the blimp's dynamics using a simple model.

        Simple discrete update, i.e. move the blimp by a fixed
        amount in the direction of the action. Useful for debugging.

        """
        # scale = 0.5   # 0.05
        if action == 0:
            self.base.move_blimp(dx=0, dy=0, dz=0)
            self.base.rotate_blimp(droll=0, dpitch=0, dyaw=0)
        elif action == 1:
            # self.base.move_blimp(dx=0, dy=1 * scale, dz=0)
            self.base.move_blimp(dx=1 * scale, dy=0, dz=0)
            self.base.rotate_blimp(droll=0, dpitch=0, dyaw=0)
        elif action == 2:
            self.base.move_blimp(dx=0, dy=0, dz=0)
            self.base.rotate_blimp(droll=0, dpitch=0, dyaw=1 * scale)
        elif action == 3:
            self.base.move_blimp(dx=0, dy=0, dz=0)
            self.base.rotate_blimp(droll=0, dpitch=0, dyaw=-1 * scale)
        elif action == 4:
            self.base.move_blimp(dx=0, dy=0, dz=1 * scale)
            self.base.rotate_blimp(droll=0, dpitch=0, dyaw=0)
        elif action == 5:
            self.base.move_blimp(dx=0, dy=0, dz=-1 * scale)
            self.base.rotate_blimp(droll=0, dpitch=0, dyaw=0)

    def _simulate_physics(self, action: int):
        """ Simulate the blimp's dynamics using a physics-based model. """
    
        # Get the current state of the blimp and format it into
        # a 12-dimensional state vector needed by the physics model
        # State vector: [vx, vy, vz, v_roll, v_pitch, v_yaw, x, y, z, roll, pitch, yaw]
        blimp_state = self.base.get_blimp_state()
        pose = blimp_state["pose"]
        velocity = blimp_state["velocity"]
        angular_velocity = blimp_state["angular_velocity"]
        X = np.concatenate([velocity, angular_velocity, pose])

        # Scale the action vector to the appropriate range
        # Action vector: [vx, vy, vz, v_yaw]
        # (the scaling is arbitrary and we-ve manually tuned it to keep
        # the dynamics model numerically stable)
        U_scale = np.array([0.01, 0.01, 0.025, 0.002])
        U = action_int_to_vec(action)
        U = U * U_scale

        # Do the simulation step using Luke's dynamics model
        sim_out = self.blimp.step(X, U, self.base.dt)

        # Extract the state variables from the output of the dynamics model
        # We've had too many issues trying to use the state vector directly,
        # so we're just going to extract the velocity and angular velocity
        # and use those to update the blimp's position and orientation
        vx, vy, vz = sim_out[0:3]
        v_roll, v_pitch, v_yaw = sim_out[3:6]

        dx = vx * self.base.dt
        dy = vy * self.base.dt
        dz = vz * self.base.dt
        droll = v_roll * self.base.dt
        dpitch = v_pitch * self.base.dt
        dyaw = v_yaw * self.base.dt

        # Update the blimp's position and orientation
        self.base.move_blimp(dx, dy, dz)
        self.base.rotate_blimp(droll, dpitch, dyaw)

    def _set_action(self, action: int) -> None:
        """ Set action to be applied on next step.

        Affords using manual controls (e.g. keyboard) or an agent (e.g. RL algorithm).

        """
        # action = self.base.controls.action if action is None else action
        if self.render_mode == "human":
            action = self.base.controls.action
        self._action = action

    def _get_action(self) -> int:
        """ Get the action that was applied on the last step. """
        return self._action

    def _apply_action(self, action: int) -> None:
        for t in range(int(self.action_repeat)):
            if self.dynamics == "simple":
                self._simulate_simple(action)
            elif self.dynamics == "physics":
                self._simulate_physics(action)

    def _get_obs(self):
        # Get 12-D pose vector needed for Task 2
        blimp_state = self.base.get_blimp_state()
        x, y, z, roll, pitch, yaw = blimp_state["pose"]
        vx, vy, vz = blimp_state["velocity"]
        v_roll, v_pitch, v_yaw = blimp_state["angular_velocity"]
        pose_for_task_2 = np.array([vx, vy, vz, v_roll, v_pitch, v_yaw, x, y, z, roll, pitch, yaw])
        pose_for_task_2 = pose_for_task_2.astype(self.observation_space["pose"].dtype)

        # Get RGB image
        # Rendering makes things run wayyy slower, so we can disable it if we want (very useful for dev/debugging)
        obs = {}
        if not self.disable_render:
            image = self.base.capture_screenshot(img_size=self.img_size)

            # Sometimes observations get returned with shape (1, 0, 3) instead of (height, width, 3)
            # I think it is some sort of anomaly from the render engine during the initial
            # env setup. We'll just ignore those observations and return an empty image instead.
            if image.shape == (1, 0, 3) or image.shape == (0, 1, 3):
                logger.warning(f"WARNING: Got image with shape {image.shape}. Returning empty image instead.")
                image = np.zeros((self.base.screen_size[0], self.base.screen_size[1], 3), dtype=np.uint8)
            obs["img"] = image
        # else:
        #     if self.step_num == 0:

        #         logger.warning(f"WARNING: Rendering is disabled. Returning empty image instead.")
        #     image = np.zeros((self.base.screen_size[0], self.base.screen_size[1], 3), dtype=np.uint8)

        obs["pose"] = pose_for_task_2
        objects = []
        for key, value in self.base.balloon_mgr._balloons.items():
            #print(value.get_pos())
            objects.append(value.get_pos())

        for key, value in self.base.fire_mgr._fires.items():
            #print(value.get_pos())
            objects.append(value.get_pos())
        
        # (near_dist_f, near_dist_b, near_dist_r, near_dist_l, near_dist_a, near_dist_u)
        obs["ultra_sonic_sensor"] = self.compute_ultra_sonic_sensor(x_p=x,y_p=y,z_p=z,roll=roll,pitch=pitch,yaw=yaw,objects=objects)
        #print(obs['ultra_sonic_sensor'])
        #print(obs["ultra_sonic_sensor"].shape)

        return obs
    
    def compute_ultra_sonic_sensor(self, x_p, y_p, z_p, roll, pitch, yaw, objects):
        #x_o, y_o, z_o
        #
        """
        Determine if an object is in front of the plane and compute the distance to the front of the plane.
        Not perfect, takes distance from center of blimp.
        
        Parameters:
        x_p, y_p, z_p : float
            Position of the plane (x, y, z coordinates).
        roll, pitch, yaw : float
            Orientation of the plane (in degrees) as roll, pitch, and yaw.
        x_o, y_o, z_o : float
            Position of the object (x, y, z coordinates).
            
        Returns:
        tuple
            A tuple containing a boolean (True if the object is in front of the plane, False otherwise) 
            and the distance from the object to the front of the plane (float).
        """
        
        # Convert angles from degrees to radians
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Define rotation matrices
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)]
        ])
        
        R_pitch = np.array([
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        R_yaw = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])
        
        # Combine the rotation matrices
        R = R_yaw @ R_pitch @ R_roll
        
        # The forward direction vector before rotation (assuming the plane faces along the z-axis initially)
        d_front_initial = np.array([1, 0, 0])
        d_back_initial = np.array([-1, 0, 0])
        d_right_initial = np.array([0, 1, 0])
        d_left_initial = np.array([0, -1, 0])
        d_above_initial = np.array([0, 0, 1])
        d_under_initial = np.array([0, 0, -1])

        #d_initial = np.array([0, 0, 1])
        
        # Apply rotation to the initial forward vector to get the new direction of the plane
        d_front = R @ d_front_initial
        d_back = R @ d_back_initial
        d_right = R @ d_right_initial
        d_left = R @ d_left_initial
        d_above = R @ d_above_initial
        d_under = R @ d_under_initial
        
        near_dist_f, near_dist_b, near_dist_r, near_dist_l, near_dist_a, near_dist_u = 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0
        for o in objects:
            # Vector from plane to object
            # v = np.array([o['x'] - x_p, o['y'] - y_p, o['z'] - z_p])
            v = np.array([o[0] - x_p, o[1] - y_p, o[2] - z_p])
            
            # Compute the dot product to determine if the object is in front of the plane
            dot_product_front = np.dot(d_front, v)
            dot_product_back = np.dot(d_back, v)
            dot_product_right = np.dot(d_right, v)
            dot_product_left = np.dot(d_left, v)
            dot_product_above = np.dot(d_above, v)
            dot_product_under = np.dot(d_under, v)
            
            # Magnitude of the direction vector
            d_front_magnitude = np.linalg.norm(d_front)
            d_back_magnitude = np.linalg.norm(d_back)
            d_right_magnitude = np.linalg.norm(d_right)
            d_left_magnitude = np.linalg.norm(d_left)
            d_above_magnitude = np.linalg.norm(d_above)
            d_under_magnitude = np.linalg.norm(d_under)
            
            # Calculate the perpendicular distance from the object to the plane
            distance_front = np.abs(dot_product_front) / d_front_magnitude
            distance_back = np.abs(dot_product_back) / d_back_magnitude
            distance_right = np.abs(dot_product_right) / d_right_magnitude
            distance_left = np.abs(dot_product_left) / d_left_magnitude
            distance_above = np.abs(dot_product_above) / d_above_magnitude
            distance_under = np.abs(dot_product_under) / d_under_magnitude
            
            # If the dot product is positive, the object is in front of the plane
            if dot_product_front > 0:
                near_dist_f = min(distance_front,near_dist_f)
            if dot_product_back > 0:
                near_dist_b = min(distance_back,near_dist_b)
            if dot_product_right > 0:
                near_dist_r = min(distance_right,near_dist_r)
            if dot_product_left > 0:
                near_dist_l = min(distance_left,near_dist_l)
            if dot_product_above > 0:
                near_dist_a = min(distance_above,near_dist_a)
            if dot_product_under > 0:
                near_dist_u = min(distance_under,near_dist_u)
            
            # in_back = dot_product_back > 0
            # in_right = dot_product_right > 0
            # in_left = dot_product_left > 0
            # in_above = dot_product_above > 0
            # in_under = dot_product_under > 0
            
            
        return np.array([near_dist_f, near_dist_b, near_dist_r, near_dist_l, near_dist_a, near_dist_u],dtype=np.float32)

    def _get_reward_and_episode_status(self) -> Dict[str, Any]:
        """
        - Reward model
            - SUCCESS > |FAILURE| + |TOTAL_COST|
            - |TOTAL_COST| < |FAILURE|
            - TOTAL_COST = TIME_LIMIT x COST
            - COST = -2x reward unit
            - SHAPING = discounted COST
        """
        eps = 2.0  # Epsilon for checking if the blimp is **touching** a fire hazard or balloon

        info = {
            'is_terminated': False,
            'is_truncated': False,
            'termination_condition': None,
            'step_num': self.step_num,
            'reward': 0,
            'from_fire': 0,
            'from_balloon': 0,
            'in_fire_zone': False,
            'in_reward_zone': False,
            'num_balloons_collected': 0,
        }
        COST = -2*self.reward_unit
        TOTAL_COST = self.max_episode_steps * COST
        SHAPING = self.reward_unit
        FAILURE = 2 * TOTAL_COST
        SUCCESS = 10*(abs(FAILURE) + abs(TOTAL_COST))

        # 1   # Base COST for each timestep
        info['reward'] += COST

        # Calculate penalties for being inside fire hazard zones
        for fire_id, fire_hazard in self.base.fire_mgr.items():
            # Get the distance from the fire hazard; if it's within the epsilon,
            # then immediately terminate the episode with a large penalty
            dist = np.linalg.norm(self.base.blimp.get_pos() - fire_hazard.get_pos())
            if dist < eps:
                info['reward'] += FAILURE
                info['from_fire'] += FAILURE
                info['is_terminated'] = True
                info["termination_condition"] = "you melted inside the fire!"
                logger.info("Got too close to the fire - you melted! Episode terminated.")
                break

            # Otherwise, if the blimp is within the danger radius, then add a penalty
            if dist < fire_hazard.danger_radius:
                info['in_fire_zone'] = True
                r_fire = -SHAPING
                info['reward'] += r_fire
                info['from_fire'] += r_fire

        # Calculate rewards for being inside balloon reward zones
        balloon_ids = self.base.balloon_mgr.list_balloons_ids()
        num_balloons = len(balloon_ids)

        closest_balloon_dist = INF
        angle = INF

        while balloon_ids:
            balloon_id = balloon_ids.pop()
            balloon = self.base.balloon_mgr.get_balloon(balloon_id)
            dist = np.linalg.norm(self.base.blimp.get_pos() - balloon.get_pos())

            if dist < closest_balloon_dist:
                closest_balloon_dist = dist

                #Check to see if we are turning towards it
                delta = balloon.get_pos() - self.base.blimp.get_pos()
                #direction = self.base.blimp.getQuat().getForward().get_xy() + panda3d.core.LVector2f(1, -1)
                #direction,_,_ = self.base.blimp.get_hpr()
                direction = panda3d.core.LVector2f(1, 0).signed_angle_deg(delta.get_xy())
                self.signed_angle = direction - self.base.blimp.get_h()
                if self.signed_angle > 180:
                    self.signed_angle = 360 - self.signed_angle
                elif self.signed_angle < -180:
                    self.signed_angle = 360 + self.signed_angle

                angle = abs(self.signed_angle)
                delta.get_xy().length()
                # angle = abs(direction.signedAngleDeg(delta.get_xy()))
                # self.signed_angle = direction.signedAngleDeg(delta.get_xy())


            # If blimp is within the epsilon of the balloon, then collect the balloon
            if dist < eps:
                info['reward'] += SUCCESS
                info['from_balloon'] += SUCCESS

                # Collect the balloon (i.e. pop it and receive the reward)
                self.collect_balloon(balloon_id)
                logger.info(f"Congratulations! You collected balloon {balloon_id}. "
                            f"Collected {len(self.base.balloon_mgr.collected_balloons)}/{num_balloons} remaining balloons.")

                # Check if all balloons collected; if so terminate immediately with reward and time-based bonus
                if self.base.balloon_mgr.num_balloons == 0:
                    msg = f"Nice Work! You collected all the balloons! Episode finished."
                    logger.info(msg)
                    info['is_terminated'] = True
                    info['termination_condition'] = msg
                    info['num_balloons_collected'] = len(self.base.balloon_mgr.collected_balloons)
                    break
        # reward 1 unit if distance between current pos and nearest balloon is less than previous or we've turned towards the balloon
        if closest_balloon_dist < self.previous_dist_balloon:
            info["reward"] += SHAPING
            info["from_balloon"] += SHAPING
        elif closest_balloon_dist > self.previous_dist_balloon or angle > self.prev_bln_angle:
            info["reward"] += -SHAPING
            info["from_balloon"] += -SHAPING

        self.previous_dist_balloon = closest_balloon_dist
        self.prev_bln_angle = angle

        info['num_balloons_collected'] = len(self.base.balloon_mgr.collected_balloons)

        # Check if we've exceeded the max number of steps
        if self.step_num >= self.max_episode_steps:
            info['is_truncated'] = True
            info['termination_condition'] = "max steps exceeded"
            logger.info("Max number of steps exceeded. Episode terminated.")

        # Check if the blimp has collided with the wall
        if self.base.has_collided_with_wall:
            info['reward'] += FAILURE
            info['is_terminated'] = True
            info["termination_condition"] = "wall collision"
            logger.info("Wall collision! Episode terminated.")

        return info

    def collect_balloon(self, balloon_id):
        self.base.balloon_mgr.collect_balloon(balloon_id)

    def _get_action_effect(self):

        obs = self._get_obs()
        info = self._get_reward_and_episode_status()
        self._obs = obs
        self._info = info   # Reward is already included in info

    def step(self, action=None) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:

        self._set_action(action)
        action = self._get_action()
        self._apply_action(action)
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

    def render(self):
        if self.render_mode == 'human':
            pass
        elif self.render_mode == 'rgb_array':
            # return self._obs['img']
            if not self._obs or self._obs['img'] is None:
                print('Error')
                logger.warning("WARNING: Got empty image. Returning empty image instead.")
                img = np.zeros((self.base.screen_size[0], self.base.screen_size[1], 3), dtype=np.uint8)
                return {"img": img, "pose": np.zeros(12, dtype=np.float32)}

            return self._obs['img']
        else:
            raise ValueError("Invalid render mode. Must be one of 'human' or 'rgb_array'.")

    def close(self):
        # self.base.destroy()
        pass


def do_run(env, render_mode):
    """ Quick test. """
    from datetime import datetime
    from stable_baselines3.common.env_checker import check_env
    from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
    from blimp_env.utils import autoregressive_infinite_generator, save_image

    logdir = cfg.DEFAULT_LOG_DIR / "test"
    logdir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    action_generator = autoregressive_infinite_generator()

    reset_options = {
        "pose": np.array([0, 0, 3, 0, 0, 0]),
        "fire_locations": [(12, 0, 0.1)],
        "fire_danger_radius": 4.0,
        "balloon_locations": [(12, 0, 7)],
        "balloon_reward_zone_radius": 20.0,
    }
    check_env(env)

    video_recorder = None
    if render_mode == "rgb_array":
        video_recorder = VideoRecorder(env=env, path=f'{logdir}/video-{ts}.mp4')

    obs, _ = env.reset(options=reset_options)
    while True:
        action = next(action_generator)
        # action = None
        obs, reward, is_terminated, is_truncated, info = env.step(action=action)

        if video_recorder:
            video_recorder.capture_frame()

        if is_terminated or is_truncated:
            break

    if video_recorder:
        video_recorder.close()


def run():
    render_mode, disable_render, debug = "human", False, True        # Use for manual control
    #render_mode, disable_render, debug = "rgb_array", False, False   # Use for RL agent
    env = IntoTheFireBasicNavEnv(
        render_mode=render_mode,
        debug=debug,
        dynamics="simple",
        disable_render=disable_render,
        max_episode_steps=10000
    )
    do_run(env, render_mode)


if __name__ == "__main__":
    run()
