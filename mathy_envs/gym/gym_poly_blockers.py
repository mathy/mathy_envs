from typing import Any

from ..envs.poly_simplify_blockers import PolySimplifyBlockers
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .gym_goal_env import MathyGymGoalEnv
from .mathy_gym_env import MathyGymEnv, safe_register

#
# Commute + simplify with blockers
#


class GymPolynomialBlockers(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymPolynomialBlockers, self).__init__(
            env_class=PolySimplifyBlockers,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolynomialBlockersEasy(GymPolynomialBlockers):
    def __init__(self, **kwargs: Any):
        super(PolynomialBlockersEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolynomialBlockersNormal(GymPolynomialBlockers):
    def __init__(self, **kwargs: Any):
        super(PolynomialBlockersNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolynomialBlockersHard(GymPolynomialBlockers):
    def __init__(self, **kwargs: Any):
        super(PolynomialBlockersHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


safe_register(
    id="mathy-poly-blockers-easy-v0",
    entry_point="mathy_envs.gym:PolynomialBlockersEasy",
)
safe_register(
    id="mathy-poly-blockers-normal-v0",
    entry_point="mathy_envs.gym:PolynomialBlockersNormal",
)
safe_register(
    id="mathy-poly-blockers-hard-v0",
    entry_point="mathy_envs.gym:PolynomialBlockersHard",
)


class GymGoalPolynomialBlockers(MathyGymGoalEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymGoalPolynomialBlockers, self).__init__(
            env_class=PolySimplifyBlockers,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class PolyBlockersGoalEasy(GymGoalPolynomialBlockers):
    def __init__(self, **kwargs: Any):
        super(PolyBlockersGoalEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class PolyBlockersGoalNormal(GymGoalPolynomialBlockers):
    def __init__(self, **kwargs: Any):
        super(PolyBlockersGoalNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class PolyBlockersGoalHard(GymGoalPolynomialBlockers):
    def __init__(self, **kwargs: Any):
        super(PolyBlockersGoalHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


# Goal envs
safe_register(
    id="mathy-goal-poly-blockers-easy-v0",
    entry_point="mathy_envs.gym:PolyBlockersGoalEasy",
)
safe_register(
    id="mathy-goal-poly-blockers-normal-v0",
    entry_point="mathy_envs.gym:PolyBlockersGoalNormal",
)
safe_register(
    id="mathy-goal-poly-blockers-hard-v0",
    entry_point="mathy_envs.gym:PolyBlockersGoalHard",
)
