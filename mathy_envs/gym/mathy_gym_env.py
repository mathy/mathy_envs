from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import error as gym_error
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.spaces.space import MaskNDArray
from mathy_core.rule import ExpressionChangeRule
from numpy.typing import NDArray

from ..env import MathyEnv
from ..state import MathyEnvState, MathyMessagePassingObservation, ObservationType
from ..time_step import is_terminal_transition
from ..types import ActionType, MathyEnvProblemArgs
from .masked_discrete import MaskedDiscrete


class MathyGymEnv(gym.Env[Any, np.int64]):
    """A small wrapper around Mathy envs to allow them to work with OpenAI Gym. The
    agents currently use this env wrapper, but it could be dropped in the future."""

    mathy: MathyEnv
    state: Optional[MathyEnvState]
    action_space: MaskedDiscrete
    _challenge: Optional[MathyEnvState]
    env_class: Type[MathyEnv]
    env_problem_args: Optional[MathyEnvProblemArgs]
    mask_as_probabilities: bool

    def __init__(
        self,
        env_class: Type[MathyEnv] = MathyEnv,
        env_problem_args: Optional[MathyEnvProblemArgs] = None,
        env_problem: Optional[str] = None,
        env_max_moves: int = 64,
        mask_as_probabilities: bool = False,
        repeat_problem: bool = False,
        obs_type: ObservationType = ObservationType.FLAT,
        **env_kwargs: Any,
    ):
        self.state = None
        self.obs_type = obs_type
        self.mask_as_probabilities = mask_as_probabilities
        self.repeat_problem = repeat_problem
        self.mathy = env_class(**env_kwargs)
        self.env_class = env_class
        self.env_problem_args = env_problem_args
        if env_problem is not None:
            self._challenge = MathyEnvState(
                problem=env_problem, max_moves=env_max_moves
            )
        else:
            self._challenge, _ = self.mathy.get_initial_state(env_problem_args)

        # Setup action space
        self.action_size = self.mathy.action_size
        # Create the masked discrete action space
        self.action_space = MaskedDiscrete(  # type:ignore
            self.action_size, np.array([1] * self.action_size)
        )

        # Define observation space based on observation type
        if obs_type == ObservationType.FLAT:
            # Original flat observation space
            mask = len(self.mathy.rules) * self.mathy.max_seq_len
            values = self.mathy.max_seq_len
            nodes = self.mathy.max_seq_len
            type_dim = 4
            time_dim = 1
            obs_size = mask + values + nodes + type_dim + time_dim
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(obs_size,), dtype=np.float32
            )
        else:
            # For structured observations, use consistent max_nodes from max_seq_len
            max_nodes = self.mathy.max_seq_len
            max_edges = max_nodes * 2  # For message passing

            if obs_type == ObservationType.GRAPH:
                self.observation_space = spaces.Dict(
                    {
                        "node_features": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(max_nodes, 2),
                            dtype=np.float32,
                        ),
                        "adjacency": spaces.Box(
                            low=0,
                            high=1,
                            shape=(max_nodes, max_nodes),
                            dtype=np.float32,
                        ),
                        "action_mask": spaces.Box(
                            low=0, high=1, shape=(self.action_size,), dtype=np.float32
                        ),
                        "num_nodes": spaces.Discrete(max_nodes + 1),
                    }
                )
            elif obs_type == ObservationType.MESSAGE_PASSING:
                # Get a sample observation to determine feature dimension
                dummy_env = env_class(**env_kwargs)
                dummy_state, _ = dummy_env.get_initial_state(env_problem_args)
                dummy_obs = dummy_env.state_to_observation(
                    dummy_state, max_seq_len=dummy_env.max_seq_len, obs_type=obs_type
                )
                assert isinstance(
                    dummy_obs, MathyMessagePassingObservation
                ), "Expected MathyMessagePassingObservation type"
                feature_dim = (
                    dummy_obs.node_features.shape[1]
                    if dummy_obs.node_features.size > 0
                    else 3
                )

                max_nodes = self.mathy.max_seq_len
                max_edges = max_nodes * 2

                self.observation_space = spaces.Dict(
                    {
                        "node_features": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                max_nodes,
                                feature_dim,
                            ),  # Use actual feature dimension
                            dtype=np.float32,
                        ),
                        "edge_index": spaces.Box(
                            low=0,
                            high=max_nodes - 1,
                            shape=(2, max_edges),
                            dtype=np.int64,
                        ),
                        "edge_types": spaces.Box(
                            low=0,
                            high=10,  # Assuming max 10 edge types
                            shape=(max_edges,),
                            dtype=np.int64,
                        ),
                        "action_mask": spaces.Box(
                            low=0, high=1, shape=(self.action_size,), dtype=np.float32
                        ),
                        "num_nodes": spaces.Discrete(max_nodes + 1),
                        "num_edges": spaces.Discrete(max_edges + 1),
                    }
                )
            elif obs_type == ObservationType.HIERARCHICAL:
                self.observation_space = spaces.Dict(
                    {
                        "node_features": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(max_nodes, 2),
                            dtype=np.float32,
                        ),
                        "level_indices": spaces.Box(
                            low=0,
                            high=max_nodes // 4,  # Max depth assumption
                            shape=(max_nodes,),
                            dtype=np.int32,
                        ),
                        "action_mask": spaces.Box(
                            low=0, high=1, shape=(self.action_size,), dtype=np.float32
                        ),
                        "max_depth": spaces.Discrete(max_nodes // 4 + 1),
                        "num_nodes": spaces.Discrete(max_nodes + 1),
                    }
                )
            else:
                raise ValueError(f"Unsupported observation type: {obs_type}")

    def step(
        self, action: Union[int, np.int64]
    ) -> Tuple[Union[NDArray[Any], Dict[str, Any]], float, bool, bool, Dict[str, Any]]:
        assert self.state is not None, "call reset() before stepping the environment"

        mask_sum = np.sum(self.action_space.mask)
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} is not valid. Action mask sum: {mask_sum}. "
                f"Action space mask: {self.action_space.mask}"
            )

        self.state, transition, change = self.mathy.get_next_state(self.state, action)
        terminated = is_terminal_transition(transition)
        truncated = self.state.agent.moves_remaining <= 0
        info = {
            "transition": transition,
            "done": terminated,
            "truncated": truncated,
            "change": change,
            "valid": change.result is not None,
        }
        if terminated or truncated:
            info["win"] = transition.reward > 0.0

        obs, _ = self._observe(self.state)
        return obs, transition.reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[NDArray[Any], Dict[str, Any]], Dict[Any, Any]]:
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
    ) -> Tuple[Union[NDArray[Any], Dict[str, Any]], Dict[Any, Any]]:
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

    def _observe(
        self, state: MathyEnvState
    ) -> Tuple[Union[NDArray[Any], Dict[str, Any]], Dict[Any, Any]]:
        """Observe the environment at the given state, updating the observation
        space and action space for the given state."""

        observation = self.mathy.state_to_observation(
            state, max_seq_len=self.mathy.max_seq_len, obs_type=self.obs_type
        )

        if isinstance(observation, np.ndarray):
            # Flat observation - extract action mask from the end of the array
            mask_start_idx = 5 + (
                2 * self.mathy.max_seq_len
            )  # type_time + nodes + values
            flat_mask: MaskNDArray = observation[mask_start_idx:].astype(np.int8)
            self.action_space.update_mask(flat_mask)
            return observation, {}
        else:
            # Structured observation - extract mask and convert to dict
            flat_mask = observation.action_mask.astype(np.int8)
            self.action_space.update_mask(flat_mask)
            return observation.to_dict(), {}


def safe_register(id: str, **kwargs: Any) -> None:
    """Ignore re-register errors."""
    try:
        register(id, **kwargs)
    except gym_error.Error:
        pass
