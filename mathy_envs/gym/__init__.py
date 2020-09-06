try:
    import gym  # noqa
except ImportError:
    raise ImportError(
        """
The "gym" library must be installed to use mathy_envs.gym submodule. Please try:

    pip install mathy_envs[gym]

"""
    )
from .gym_binomial_distribute import *  # noqa
from .gym_complex_simplify import *  # noqa
from .gym_poly_blockers import *  # noqa
from .gym_poly_combine_in_place import *  # noqa
from .gym_poly_commute_like_terms import *  # noqa
from .gym_poly_grouping import *  # noqa
from .gym_poly_haystack_like_terms import *  # noqa
from .gym_poly_simplify import *  # noqa
from .masked_discrete import *  # noqa
from .mathy_gym_env import *  # noqa
