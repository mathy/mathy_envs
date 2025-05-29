from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import error as gym_error
from gymnasium import spaces
from gymnasium.envs.registration import register
from mathy_core.rule import ExpressionChangeRule
from numpy.typing import NDArray

from ..env import MathyEnv
from ..state import (
    MathyEnvState,
    MathyGraphObservation,
    MathyHierarchicalObservation,
    MathyMessagePassingObservation,
    MathyObservation,
    ObservationType,
)
from ..time_step import is_terminal_transition
from ..types import ActionType, MathyEnvProblemArgs
from .masked_discrete import MaskedDiscrete


class MathyGymEnv(gym.Env[NDArray[Any], np.int64]):
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

        self.action_space = MaskedDiscrete(
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

    @property
    def action_size(self) -> int:
        return self.mathy.action_size

    def step(
        self, action: Union[int, np.int64, ActionType]
    ) -> Tuple[np.ndarray, Any, bool, bool, Dict[str, object]]:
        assert self.state is not None, "call reset() before stepping the environment"
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
    ) -> Tuple[Any, Dict[Any, Any]]:
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
    ) -> Tuple[np.ndarray, dict]:
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
    ) -> Tuple[Union[np.ndarray, Dict[str, Any]], dict]:
        """Observe the environment at the given state, updating the observation
        space and action space for the given state."""
        action_mask = self.mathy.get_valid_moves(state)
        observation = self.mathy.state_to_observation(
            state, max_seq_len=self.mathy.max_seq_len, obs_type=self.obs_type
        )
        flat_mask = np.array(action_mask).reshape(-1)

        # Ensure mask matches expected action space size
        expected_action_size = self.action_space.n
        if len(flat_mask) != expected_action_size:
            # print(
            #     f"Warning: Mask length {len(flat_mask)} != expected {expected_action_size}"
            # )
            # Pad or truncate mask to match expected size
            if len(flat_mask) < expected_action_size:
                # Pad with zeros (invalid actions)
                padded_mask = np.zeros(expected_action_size, dtype=flat_mask.dtype)
                padded_mask[: len(flat_mask)] = flat_mask
                flat_mask = padded_mask
            else:
                # Truncate to expected size
                flat_mask = flat_mask[:expected_action_size]

        # Always update action space mask (used by MaskedDiscrete regardless of obs type)
        self.action_space.update_mask(flat_mask)

        if self.mask_as_probabilities:
            mask_sum = np.sum(flat_mask)
            if mask_sum > 0.0:
                flat_mask = flat_mask / mask_sum

        if isinstance(observation, MathyObservation):
            # Original flat observation logic
            nodes = np.array(observation.nodes, dtype="float32").reshape(-1)
            values = np.array(observation.values, dtype="float32").reshape(-1)
            obs = np.concatenate(
                np.array(
                    [observation.type, observation.time, nodes, values, flat_mask],
                    dtype="object",
                ),
                axis=-1,
                dtype="float32",
            )
            return obs, {}

        elif isinstance(observation, MathyGraphObservation):
            # For graph observations, pad to maximum expected size
            max_nodes = self.mathy.max_seq_len  # Should match the space definition

            node_features = np.array(observation.node_features, dtype="float32")
            adjacency = np.array(observation.adjacency, dtype="float32")
            actual_nodes = node_features.shape[0]

            # Pad node features to max_nodes
            padded_features = np.zeros((max_nodes, 2), dtype="float32")
            padded_features[:actual_nodes] = node_features

            # Pad adjacency matrix to max_nodes x max_nodes
            padded_adjacency = np.zeros((max_nodes, max_nodes), dtype="float32")
            padded_adjacency[:actual_nodes, :actual_nodes] = adjacency

            obs_dict = {
                "node_features": padded_features,
                "adjacency": padded_adjacency,
                "action_mask": flat_mask,
                "num_nodes": actual_nodes,
            }
            return obs_dict, {}

        elif isinstance(observation, MathyMessagePassingObservation):
            # For message passing observations, pad to maximum expected sizes
            max_nodes = self.mathy.max_seq_len
            max_edges = self.mathy.max_seq_len * 2

            node_features = np.array(observation.node_features, dtype="float32")
            edge_index = np.array(observation.edge_index, dtype=np.int64)
            edge_types = np.array(observation.edge_types, dtype=np.int64)

            actual_nodes = node_features.shape[0]
            actual_edges = edge_index.shape[1] if edge_index.size > 0 else 0

            # Get the actual feature dimension from the observation
            actual_feature_dim = node_features.shape[1] if node_features.size > 0 else 2

            # Pad node features - use actual feature dimension
            padded_features = np.zeros((max_nodes, actual_feature_dim), dtype="float32")
            padded_features[:actual_nodes] = node_features

            # Pad edge information
            padded_edge_index = np.zeros((2, max_edges), dtype=np.int64)
            padded_edge_types = np.zeros((max_edges,), dtype=np.int64)

            if actual_edges > 0:
                padded_edge_index[:, :actual_edges] = edge_index
                padded_edge_types[:actual_edges] = edge_types

            obs_dict = {
                "node_features": padded_features,
                "edge_index": padded_edge_index,
                "edge_types": padded_edge_types,
                "action_mask": flat_mask,
                "num_nodes": actual_nodes,
                "num_edges": actual_edges,
            }
            return obs_dict, {}
        else:
            # For hierarchical observations, flatten the levels
            max_nodes = self.mathy.max_seq_len
            max_depth = (
                self.mathy.max_seq_len // 2
            )  # Assuming max depth is half the max seq length

            # Collect all nodes and their level information
            all_node_features = []
            level_indices = []

            for level, level_data in observation.levels.items():
                level_features = level_data["features"]  # Assuming this exists
                all_node_features.extend(level_features)
                level_indices.extend([level] * len(level_features))

            # Convert to numpy arrays
            node_features = (
                np.array(all_node_features, dtype="float32")
                if all_node_features
                else np.zeros((0, 2), dtype="float32")
            )
            level_indices = (
                np.array(level_indices, dtype=np.int32)
                if level_indices
                else np.zeros((0,), dtype=np.int32)
            )
            actual_nodes = len(all_node_features)

            # Pad to maximum sizes
            padded_features = np.zeros((max_nodes, 2), dtype="float32")
            padded_levels = np.zeros((max_nodes,), dtype=np.int32)

            if actual_nodes > 0:
                padded_features[:actual_nodes] = node_features
                padded_levels[:actual_nodes] = level_indices

            obs_dict = {
                "node_features": padded_features,
                "level_indices": padded_levels,
                "action_mask": flat_mask,
                "max_depth": observation.max_depth,
                "num_nodes": actual_nodes,
            }
            return obs_dict, {}


def safe_register(id: str, **kwargs: Any) -> None:
    """Ignore re-register errors."""
    try:
        register(id, **kwargs)
    except gym_error.Error:
        pass
