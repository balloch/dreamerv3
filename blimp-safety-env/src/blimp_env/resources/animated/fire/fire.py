from direct.particles.ParticleEffect import ParticleEffect

from blimp_env.settings import RESOURCES_DIR


class FireAnimation(object):

    def __init__(self, pos=(0, 0, 0)):
        self.pos = pos

        self._fire = ParticleEffect()
        self._fire.loadConfig(RESOURCES_DIR / 'animated/fire/fire.ptf')

        self._smoke = ParticleEffect()
        self._smoke.loadConfig(RESOURCES_DIR / 'animated/fire/smoke.ptf')

    def start(self, render):
        """ Ignite the fire (i.e. render it)"""
        self._fire.start(parent=render, renderParent=render)
        self._fire.setPos(self.pos)

        x, y, z = self.pos
        self._smoke.start(parent=render, renderParent=render)
        self._smoke.setPos((x, y, z + 5))

