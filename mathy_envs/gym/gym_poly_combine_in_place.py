from typing import Any

from ..envs.poly_combine_in_place import PolyCombineInPlace
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv, safe_register

#
# Combine like terms without commuting
#


class GymPolynomialCombineInPlace(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymPolynomialCombineInPlace, self).__init__(
            env_class=PolyCombineInPlace,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialCombineInPlaceEasy(GymPolynomialCombineInPlace):
    def __init__(self, **kwargs: Any):
        super(PolynomialCombineInPlaceEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialCombineInPlaceNormal(GymPolynomialCombineInPlace):
    def __init__(self, **kwargs: Any):
        super(PolynomialCombineInPlaceNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialCombineInPlaceHard(GymPolynomialCombineInPlace):
    def __init__(self, **kwargs: Any):
        super(PolynomialCombineInPlaceHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


safe_register(
    id="mathy-poly-combine-easy-v0",
    entry_point="mathy_envs.gym:PolynomialCombineInPlaceEasy",
)
safe_register(
    id="mathy-poly-combine-normal-v0",
    entry_point="mathy_envs.gym:PolynomialCombineInPlaceNormal",
)
safe_register(
    id="mathy-poly-combine-hard-v0",
    entry_point="mathy_envs.gym:PolynomialCombineInPlaceHard",
)
