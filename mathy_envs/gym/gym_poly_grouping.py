from typing import Any

from ..envs.poly_grouping import PolyGroupLikeTerms
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv, safe_register

#
# Group like terms
#


class GymPolynomialGrouping(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymPolynomialGrouping, self).__init__(
            env_class=PolyGroupLikeTerms,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialGroupingEasy(GymPolynomialGrouping):
    def __init__(self, **kwargs: Any):
        super(PolynomialGroupingEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialGroupingNormal(GymPolynomialGrouping):
    def __init__(self, **kwargs: Any):
        super(PolynomialGroupingNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialGroupingHard(GymPolynomialGrouping):
    def __init__(self, **kwargs: Any):
        super(PolynomialGroupingHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


safe_register(
    id="mathy-poly-grouping-easy-v0",
    entry_point="mathy_envs.gym:PolynomialGroupingEasy",
)
safe_register(
    id="mathy-poly-grouping-normal-v0",
    entry_point="mathy_envs.gym:PolynomialGroupingNormal",
)
safe_register(
    id="mathy-poly-grouping-hard-v0",
    entry_point="mathy_envs.gym:PolynomialGroupingHard",
)
