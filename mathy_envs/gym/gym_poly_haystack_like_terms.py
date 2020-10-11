from typing import Any

from ..envs.poly_haystack_like_terms import PolyHaystackLikeTerms
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .gym_goal_env import MathyGymGoalEnv
from .mathy_gym_env import MathyGymEnv, safe_register

#
# Identify like terms in a haystack
#


class GymPolynomialLikeTermsHaystack(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymPolynomialLikeTermsHaystack, self).__init__(
            env_class=PolyHaystackLikeTerms,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialLikeTermsHaystackEasy(GymPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs: Any):
        super(PolynomialLikeTermsHaystackEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialLikeTermsHaystackNormal(GymPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs: Any):
        super(PolynomialLikeTermsHaystackNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialLikeTermsHaystackHard(GymPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs: Any):
        super(PolynomialLikeTermsHaystackHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


safe_register(
    id="mathy-poly-like-terms-haystack-easy-v0",
    entry_point="mathy_envs.gym:PolynomialLikeTermsHaystackEasy",
)
safe_register(
    id="mathy-poly-like-terms-haystack-normal-v0",
    entry_point="mathy_envs.gym:PolynomialLikeTermsHaystackNormal",
)
safe_register(
    id="mathy-poly-like-terms-haystack-hard-v0",
    entry_point="mathy_envs.gym:PolynomialLikeTermsHaystackHard",
)


class GymGoalPolynomialLikeTermsHaystack(MathyGymGoalEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymGoalPolynomialLikeTermsHaystack, self).__init__(
            env_class=PolyHaystackLikeTerms,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialLikeTermsHaystackGoalEasy(GymGoalPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs: Any):
        super(PolynomialLikeTermsHaystackGoalEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialLikeTermsHaystackGoalNormal(GymGoalPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs: Any):
        super(PolynomialLikeTermsHaystackGoalNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialLikeTermsHaystackGoalHard(GymGoalPolynomialLikeTermsHaystack):
    def __init__(self, **kwargs: Any):
        super(PolynomialLikeTermsHaystackGoalHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


# Goal envs
safe_register(
    id="mathy-goal-complex-easy-v0",
    entry_point="mathy_envs.gym:PolynomialLikeTermsHaystackGoalEasy",
)
safe_register(
    id="mathy-goal-complex-normal-v0",
    entry_point="mathy_envs.gym:PolynomialLikeTermsHaystackGoalNormal",
)
safe_register(
    id="mathy-goal-complex-hard-v0",
    entry_point="mathy_envs.gym:PolynomialLikeTermsHaystackGoalHard",
)
