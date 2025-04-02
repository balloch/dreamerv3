from embodied.envs import vihome

virtualhome = vihome.VirtualHome(vihome.VirtualHomeConfig())

observations = virtualhome.env.get_observations()
print(len(observations.keys()))