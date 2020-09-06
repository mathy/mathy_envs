from typing import Any

from ..envs.complex_simplify import ComplexSimplify
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .mathy_gym_env import MathyGymEnv, safe_register


class GymComplexTerms(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymComplexTerms, self).__init__(
            env_class=ComplexSimplify,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class ComplexTermsEasy(GymComplexTerms):
    def __init__(self, **kwargs: Any):
        super(ComplexTermsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class ComplexTermsNormal(GymComplexTerms):
    def __init__(self, **kwargs: Any):
        super(ComplexTermsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class ComplexTermsHard(GymComplexTerms):
    def __init__(self, **kwargs: Any):
        super(ComplexTermsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


safe_register(id="mathy-complex-easy-v0", entry_point="mathy_envs.gym:ComplexTermsEasy")
safe_register(
    id="mathy-complex-normal-v0", entry_point="mathy_envs.gym:ComplexTermsNormal"
)
safe_register(id="mathy-complex-hard-v0", entry_point="mathy_envs.gym:ComplexTermsHard")
