from typing import Any, List, Optional, Type, Union

import gym
import numpy as np
from gym import spaces
from mathy_core.rule import ExpressionChangeRule

from ..env import MathyEnv
from ..state import MathyEnvState, MathyObservation
from ..time_step import is_terminal_transition
from ..types import ActionType, MathyEnvProblemArgs
from .masked_discrete import MaskedDiscrete


class MathyGoals:
    """Multi-goal environment goal identifiers. Used to represent
    various goals as an input to the model and for goal-substitution
    using HER."""

    ENV_GOAL = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    SIMPLIFY_GOAL = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    EXPAND_GOAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    INVALID_MOVE_GOAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    @staticmethod
    def same(
        check: Union[np.ndarray, List[int]], against: Union[np.ndarray, List[int]]
    ) -> bool:
        return (np.array(check) == np.array(against)).all()


class MathyGymGoalEnv(gym.GoalEnv):
    """A small wrapper around Mathy envs to allow them to work with OpenAI
    gym.Goal envs."""

    mathy: MathyEnv
    state: Optional[MathyEnvState]
    _challenge: Optional[MathyEnvState]
    env_class: Type[MathyEnv]
    env_problem_args: Optional[MathyEnvProblemArgs]

    def __init__(
        self,
        env_class: Type[MathyEnv] = MathyEnv,
        env_problem_args: Optional[MathyEnvProblemArgs] = None,
        env_problem: Optional[str] = None,
        env_max_moves: int = 64,
        np_observation: bool = True,
        repeat_problem: bool = False,
        **env_kwargs: Any,
    ):
        self.state = None
        self.repeat_problem = repeat_problem
        self.np_observation = np_observation
        self.mathy = env_class(**env_kwargs)
        self.env_class = env_class
        self.env_problem_args = env_problem_args
        if env_problem is not None:
            self._challenge = MathyEnvState(
                problem=env_problem, max_moves=env_max_moves
            )
        else:
            self._challenge, _ = self.mathy.get_initial_state(env_problem_args)

        self.action_space = MaskedDiscrete(self.action_size, [1] * self.action_size)
        mask = len(self.mathy.rules) * self.mathy.max_seq_len
        values = self.mathy.max_seq_len
        nodes = self.mathy.max_seq_len
        type = 4
        time = 1
        obs_size = mask + values + nodes + type + time
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.MultiBinary(16),
                achieved_goal=spaces.MultiBinary(16),
                observation=spaces.Box(
                    low=0, high=1, shape=(obs_size,), dtype="float32"
                ),
            )
        )

    @property
    def action_size(self) -> int:
        return self.mathy.action_size

    def step(
        self, action: Union[int, ActionType]
    ) -> Union[MathyObservation, np.ndarray]:
        assert self.state is not None, "call reset() before stepping the environment"
        self.state, transition, change = self.mathy.get_next_state(self.state, action)
        done = is_terminal_transition(transition)
        info = {
            "transition": transition,
            "done": done,
            "valid": change.result is not None,
        }
        if done:
            info["win"] = transition.reward > 0.0
        return self._observe(self.state), transition.reward, done, info

    def _observe(self, state: MathyEnvState) -> dict:
        """Observe the environment at the given state, updating the observation
        space and action space for the given state."""
        action_mask = self.mathy.get_valid_moves(state)
        observation = self.mathy.state_to_observation(
            state, max_seq_len=self.mathy.max_seq_len
        )
        flat_mask = np.array(action_mask).reshape(-1)
        self.action_space.update_mask(flat_mask)
        nodes = np.array(observation.nodes, dtype="float32").reshape(-1)
        values = np.array(observation.values, dtype="float32").reshape(-1)
        mask = np.array(action_mask, dtype="float32").reshape(-1)
        np_observation = np.concatenate(
            [observation.type, observation.time, nodes, values, mask], axis=-1
        )

        # Determine the achieved expand/simplify goal
        curr_text = state.agent.problem
        last_text = state.agent.history[-1].raw
        achieved = (
            MathyGoals.EXPAND_GOAL
            if len(curr_text) > len(last_text)
            else MathyGoals.SIMPLIFY_GOAL
        )

        output = {
            "observation": np_observation,
            "desired_goal": np.array(MathyGoals.ENV_GOAL),
            "achieved_goal": np.array(achieved),
        }
        return output

    def reset(self) -> Union[MathyObservation, np.ndarray]:
        if self.state is not None:
            self.mathy.finalize_state(self.state)
        if self.repeat_problem:
            assert self._challenge is not None
            self.state = MathyEnvState.copy(self._challenge)
        else:
            self.state, self.problem = self.mathy.get_initial_state(
                self.env_problem_args
            )
        return self._observe(self.state)

    def reset_with_input(
        self, problem_text: str, max_moves: int = 16
    ) -> Union[MathyObservation, np.ndarray]:
        # If the episode is being reset because it ended, assert the validity
        # of the last problem outcome
        if self.state is not None:
            self.mathy.finalize_state(self.state)
        self.reset()
        self.state = MathyEnvState(problem=problem_text, max_moves=max_moves)
        return self._observe(self.state)

    def render(
        self,
        last_action: ActionType = (-1, -1),
        last_reward: float = 0.0,
        last_change: Optional[ExpressionChangeRule] = None,
    ) -> None:
        assert self.state is not None, "call reset() before rendering the env"
        action_name = "initial"
        token_index = -1
        if last_action != (-1, -1):
            action_index, token_index = last_action
            action_name = self.mathy.rules[action_index].name
        else:
            print(f"Problem: {self.state.agent.problem}")
        self.mathy.print_state(
            self.state,
            action_name[:25].lower(),
            token_index,
            change=last_change,
            change_reward=last_reward,
        )
