from typing import Any

from ..envs.complex_simplify import ComplexSimplify
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .gym_goal_env import MathyGymGoalEnv
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


class GymGoalComplexSimplify(MathyGymGoalEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymGoalComplexSimplify, self).__init__(
            env_class=ComplexSimplify,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class ComplexTermsGoalEasy(GymGoalComplexSimplify):
    def __init__(self, **kwargs: Any):
        super(ComplexTermsGoalEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class ComplexTermsGoalNormal(GymGoalComplexSimplify):
    def __init__(self, **kwargs: Any):
        super(ComplexTermsGoalNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class ComplexTermsGoalHard(GymGoalComplexSimplify):
    def __init__(self, **kwargs: Any):
        super(ComplexTermsGoalHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


# Goal envs
safe_register(
    id="mathy-goal-complex-easy-v0", entry_point="mathy_envs.gym:ComplexTermsGoalEasy"
)
safe_register(
    id="mathy-goal-complex-normal-v0",
    entry_point="mathy_envs.gym:ComplexTermsGoalNormal",
)
safe_register(
    id="mathy-goal-complex-hard-v0", entry_point="mathy_envs.gym:ComplexTermsGoalHard"
)
