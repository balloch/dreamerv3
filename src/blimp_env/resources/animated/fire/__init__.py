from panda3d.core import loadPrcFileData

from blimp_env.settings import RESOURCES_DIR

# TODO: Add in any other necessary paths here
loadPrcFileData("", f"model-path {str(RESOURCES_DIR / 'animated/fire')}")
