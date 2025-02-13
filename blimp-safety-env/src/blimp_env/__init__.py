# SPDX-FileCopyrightText: 2023-present Barker, Daniel C. <daniel.barker@gtri.gatech.edu>
#
# SPDX-License-Identifier: MIT
from gymnasium import register

from blimp_env.tasks.into_the_fire_basic_nav import IntoTheFireBasicNavEnv
from blimp_env.tasks.follow_task import FollowTaskEnv
from blimp_env.tasks.chase_task import ChaseTaskEnv


register(
    id="blimp_env/IntoTheFireBasicNav-v0",
    entry_point="blimp_env.tasks.into_the_fire_basic_nav:IntoTheFireBasicNavEnv",
)
register(
    id="blimp_env/FollowTask-v0",
    entry_point="blimp_env.tasks.follow_task:FollowTaskEnv",
)
register(
    id="blimp_env/ChaseTask-v0",
    entry_point="blimp_env.tasks.chase_task:ChaseTaskEnv",
)

__version__ = "0.1.0"
__all__ = [
    "IntoTheFireBasicNavEnv",
    "FollowTaskEnv",
    "ChaseTaskEnv",
    "__version__"
]
