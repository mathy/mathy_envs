from typing import Any

from ..envs.binomial_distribute import BinomialDistribute
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .gym_goal_env import MathyGymGoalEnv
from .mathy_gym_env import MathyGymEnv, safe_register


class GymBinomialDistribution(MathyGymEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymBinomialDistribution, self).__init__(
            env_class=BinomialDistribute,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class BinomialsEasy(GymBinomialDistribution):
    def __init__(self, **kwargs: Any):
        super(BinomialsEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class BinomialsNormal(GymBinomialDistribution):
    def __init__(self, **kwargs: Any):
        super(BinomialsNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class BinomialsHard(GymBinomialDistribution):
    def __init__(self, **kwargs: Any):
        super(BinomialsHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


# Gym envs
safe_register(id="mathy-binomial-easy-v0", entry_point="mathy_envs.gym:BinomialsEasy")
safe_register(
    id="mathy-binomial-normal-v0", entry_point="mathy_envs.gym:BinomialsNormal"
)
safe_register(id="mathy-binomial-hard-v0", entry_point="mathy_envs.gym:BinomialsHard")


class GymGoalBinomialDistribution(MathyGymGoalEnv):
    def __init__(self, difficulty: MathyEnvDifficulty, **kwargs: Any):
        super(GymGoalBinomialDistribution, self).__init__(
            env_class=BinomialDistribute,
            env_problem_args=MathyEnvProblemArgs(difficulty=difficulty),
            **kwargs
        )


class BinomialsGoalEasy(GymGoalBinomialDistribution):
    def __init__(self, **kwargs: Any):
        super(BinomialsGoalEasy, self).__init__(
            difficulty=MathyEnvDifficulty.easy, **kwargs
        )


class BinomialsGoalNormal(GymGoalBinomialDistribution):
    def __init__(self, **kwargs: Any):
        super(BinomialsGoalNormal, self).__init__(
            difficulty=MathyEnvDifficulty.normal, **kwargs
        )


class BinomialsGoalHard(GymGoalBinomialDistribution):
    def __init__(self, **kwargs: Any):
        super(BinomialsGoalHard, self).__init__(
            difficulty=MathyEnvDifficulty.hard, **kwargs
        )


# Goal envs
safe_register(
    id="mathy-goal-binomial-easy-v0", entry_point="mathy_envs.gym:BinomialsGoalEasy"
)
safe_register(
    id="mathy-goal-binomial-normal-v0", entry_point="mathy_envs.gym:BinomialsGoalNormal"
)
safe_register(
    id="mathy-goal-binomial-hard-v0", entry_point="mathy_envs.gym:BinomialsGoalHard"
)
