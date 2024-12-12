import os
from pathlib import Path
try:
    from importlib.resources import files
except ImportError:
    # importlib added the files function in Python 3.9 - use backport if using earlier version
    from importlib_resources import files


ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_LOG_DIR = ROOT_DIR / 'logdir'
PACKAGE_NAME = 'blimp_env'
PACKAGE_DIR = Path(files(PACKAGE_NAME)).absolute()
RESOURCES_DIR = PACKAGE_DIR / 'resources'
MODELS_DIR = RESOURCES_DIR / 'models'
