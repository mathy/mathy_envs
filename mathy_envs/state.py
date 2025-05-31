from enum import Enum, IntEnum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from zlib import adler32

import numpy as np
from mathy_core.expressions import ConstantExpression, MathExpression, MathTypeKeys
from mathy_core.parser import ExpressionParser
from numpy.typing import NDArray

from .types import ActionType
from .util import pad_array

PROBLEM_TYPE_HASH_BUCKETS = 128

NodeIntList = List[int]
NodeValuesFloatList = List[float]
NodeMaskIntList = List[List[int]]
ProblemTypeIntList = List[int]
TimeFloatList = List[float]


WindowNodeIntList = List[NodeIntList]
WindowNodeMaskIntList = List[NodeMaskIntList]
WindowNodeValuesFloatList = List[NodeValuesFloatList]
WindowProblemTypeIntList = List[ProblemTypeIntList]
WindowTimeFloatList = List[TimeFloatList]


# Input type for mathy models
MathyInputsType = Dict[str, Any]


class ObservationType(Enum):
    FLAT = "flat"
    GRAPH = "graph"
    HIERARCHICAL = "hierarchical"
    MESSAGE_PASSING = "message_passing"


class ObservationFeatureIndices(IntEnum):
    nodes = 0
    mask = 1
    values = 2
    type = 3
    time = 4


class MathyObservation(NamedTuple):
    """A featurized observation from an environment state."""

    @classmethod
    def empty(cls, template: "MathyObservation") -> "MathyObservation":
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]],
            values=[0.0],
            mask=template.mask,
            type=[0, 0],
            time=[0.0],
        )

    nodes: NodeIntList
    mask: NodeMaskIntList
    values: NodeValuesFloatList
    type: ProblemTypeIntList
    time: TimeFloatList


# fmt: off
MathyObservation.nodes.__doc__ = "tree node types in the current environment state shape=[n,]"  # noqa
MathyObservation.mask.__doc__ = "0/1 mask where 0 indicates an invalid action shape=[n,]"  # noqa
MathyObservation.values.__doc__ = "tree node value sequences, with non number indices set to 0.0 shape=[n,]"  # noqa
MathyObservation.type.__doc__ = "two column hash of problem environment type shape=[2,]"  # noqa
MathyObservation.time.__doc__ = "float value between 0.0 and 1.0 indicating the time elapsed shape=[1,]"  # noqa
# fmt: on


class MathyGraphObservation(NamedTuple):
    node_features: np.ndarray[Any, np.dtype[np.float32]]
    adjacency: np.ndarray[Any, np.dtype[np.float32]]
    action_mask: np.ndarray[Any, np.dtype[np.float32]]
    num_nodes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by gym observation spaces"""
        return {
            "node_features": self.node_features,
            "adjacency": self.adjacency,
            "action_mask": self.action_mask,
            "num_nodes": self.num_nodes,
        }


class MathyHierarchicalObservation(NamedTuple):
    node_features: np.ndarray[Any, np.dtype[np.float32]]
    level_indices: np.ndarray[Any, np.dtype[np.int32]]
    action_mask: np.ndarray[Any, np.dtype[np.float32]]
    max_depth: int
    num_nodes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by gym observation spaces"""
        return {
            "node_features": self.node_features,
            "level_indices": self.level_indices,
            "action_mask": self.action_mask,
            "max_depth": self.max_depth,
            "num_nodes": self.num_nodes,
        }


class MathyMessagePassingObservation(NamedTuple):
    node_features: np.ndarray[Any, np.dtype[np.float32]]
    edge_index: np.ndarray[Any, np.dtype[np.int64]]
    edge_types: np.ndarray[Any, np.dtype[np.int64]]
    action_mask: np.ndarray[Any, np.dtype[np.float32]]
    num_nodes: int
    num_edges: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by gym observation spaces"""
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_types": self.edge_types,
            "action_mask": self.action_mask,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
        }


# Union type for all observation types
MathyObservationUnion = Union[
    NDArray[Union[np.int32, np.float32]],
    MathyGraphObservation,
    MathyHierarchicalObservation,
    MathyMessagePassingObservation,
]


class MathyEnvStateStep(NamedTuple):
    """Capture summarized environment state for a previous timestep so the
    agent can use context from its history when making new predictions."""

    raw: str
    action: ActionType


# fmt: off
MathyEnvStateStep.raw.__doc__ = "the input text at the timestep"  # noqa
MathyEnvStateStep.action.__doc__ = "a tuple indicating the chosen action and the node it was applied to"  # noqa
# fmt: on


_problem_hash_cache: Optional[Dict[str, List[int]]] = None


class MathyAgentState:
    """The state related to an agent for a given environment state"""

    moves_remaining: int
    problem: str
    problem_type: str
    reward: float
    action: ActionType
    history: List[MathyEnvStateStep]

    def __init__(
        self,
        moves_remaining: int,
        problem: str,
        problem_type: str,
        reward: float = 0.0,
        history: Optional[List[MathyEnvStateStep]] = None,
    ):
        self.moves_remaining = moves_remaining
        self.problem = problem
        self.reward = reward
        self.problem_type = problem_type
        self.history = (
            history[:]
            if history is not None
            else [MathyEnvStateStep(problem, (-1, -1))]
        )

    @classmethod
    def copy(cls, from_state: "MathyAgentState") -> "MathyAgentState":
        return MathyAgentState(
            moves_remaining=from_state.moves_remaining,
            problem=from_state.problem,
            reward=from_state.reward,
            problem_type=from_state.problem_type,
            history=from_state.history,
        )


class MathyEnvState(object):
    """Class for holding environment state and extracting features
    to be passed to the policy/value neural network.

    Mutating operations all return a copy of the environment adapter
    with its own state.

    This allocation strategy requires more memory but removes a class
    of potential issues around unintentional sharing of data and mutation
    by two different sources.
    """

    agent: "MathyAgentState"
    max_moves: int
    num_rules: int

    def __init__(
        self,
        state: Optional["MathyEnvState"] = None,
        problem: Optional[str] = None,
        max_moves: int = 10,
        num_rules: int = 0,
        problem_type: str = "mathy.unknown",
    ):
        self.max_moves = max_moves
        self.num_rules = num_rules
        if problem is not None:
            self.agent = MathyAgentState(max_moves, problem, problem_type)
        elif state is not None:
            self.num_rules = state.num_rules
            self.max_moves = state.max_moves
            self.agent = MathyAgentState.copy(state.agent)

    @classmethod
    def copy(cls, from_state: "MathyEnvState") -> "MathyEnvState":
        return MathyEnvState(state=from_state)

    def clone(self) -> "MathyEnvState":
        return MathyEnvState(state=self)

    def get_out_state(
        self, problem: str, action: ActionType, moves_remaining: int
    ) -> "MathyEnvState":
        """Get the next environment state based on the current one with updated
        history and agent information based on an action being taken."""
        out_state = MathyEnvState.copy(self)
        agent = out_state.agent
        agent.history.append(MathyEnvStateStep(problem, action))
        agent.problem = problem
        agent.action = action
        agent.moves_remaining = moves_remaining
        return out_state

    def get_problem_hash(self) -> ProblemTypeIntList:
        """Return a two element array with hashed values for the current environment
        namespace string.

        # Example

        - `mycorp.envs.solve_impossible_problems` -> `[12375561, -12375561]`

        """
        global _problem_hash_cache
        if _problem_hash_cache is None:
            _problem_hash_cache = {}
        if self.agent.problem_type not in _problem_hash_cache:
            type_str = self.agent.problem_type
            # Generate a normalized hash with values between 0.0 and 1.0
            # The adler32 crc is used in zip files and is determinstic
            # across runs while also being fast to calculate. It is NOT
            # cryptographically secure.
            hashes: Any = np.asarray(  # type:ignore
                [
                    adler32(type_str.encode(encoding="utf-8")),
                    adler32(type_str[::-1].encode(encoding="utf-8")),
                    adler32(type_str.upper().encode(encoding="utf-8")),
                    adler32(type_str.replace(".", "").encode(encoding="utf-8")),
                ],
                dtype=np.float32,
            )
            if hashes.sum() != 0.0:
                hashes = (hashes - min(hashes)) / (max(hashes) - min(hashes))
            values = hashes.tolist()

            _problem_hash_cache[self.agent.problem_type] = values

        return _problem_hash_cache[self.agent.problem_type]

    def _process_action_mask(
        self,
        raw_mask: NodeMaskIntList,
        max_seq_len: int,
        num_rules: Optional[int] = None,
    ) -> NDArray[np.float32]:
        """Convert raw action mask to padded flat format for action space"""
        if num_rules is None:
            num_rules = self.num_rules

        raw_mask_array = np.array(raw_mask, dtype=np.float32)
        num_rules_actual, actual_nodes = raw_mask_array.shape

        # Ensure we pad to the expected total number of rules, not just the actual rules
        # The gym wrapper expects num_rules total, even if some rules have no valid ones
        expected_num_rules = max(num_rules, num_rules_actual)

        # Pad the mask to (expected_num_rules, max_seq_len) then flatten
        padded_mask = np.zeros((expected_num_rules, max_seq_len), dtype=np.float32)
        padded_mask[:num_rules_actual, :actual_nodes] = raw_mask_array
        flat_mask = padded_mask.reshape(
            -1
        )  # Shape: (expected_num_rules * max_seq_len,)

        return flat_mask

    def to_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        hash_type: Optional[ProblemTypeIntList] = None,
        parser: Optional[ExpressionParser] = None,
        normalize: bool = True,
        max_seq_len: Optional[int] = None,
        obs_type: ObservationType = ObservationType.FLAT,
    ) -> MathyObservationUnion:
        """Unified observation method that returns the appropriate type"""
        if obs_type == ObservationType.FLAT:
            return self.to_flat_observation(
                move_mask=move_mask,
                hash_type=hash_type,
                parser=parser,
                normalize=normalize,
                max_seq_len=max_seq_len,
            )
        elif obs_type == ObservationType.GRAPH:
            return self.to_graph_observation(
                move_mask=move_mask,
                hash_type=hash_type,
                parser=parser,
                normalize=normalize,
                max_seq_len=max_seq_len,
            )
        elif obs_type == ObservationType.HIERARCHICAL:
            return self.to_hierarchical_observation(
                move_mask=move_mask,
                parser=parser,
                max_seq_len=max_seq_len,
            )
        elif obs_type == ObservationType.MESSAGE_PASSING:
            return self.to_message_passing_observation(
                move_mask=move_mask,
                parser=parser,
                max_seq_len=max_seq_len,
            )
        else:
            raise ValueError(f"Unknown observation type: {obs_type}")

    def _get_node_features(
        self,
        nodes: List[MathExpression],
        normalize: bool = True,
    ) -> Tuple[List[List[float]], int]:
        """Get consistent node features across all observation types.

        Returns:
            Tuple of (node_features_list, feature_dim)

        Node features: [type_id, value, time, is_leaf]
        """
        # Calculate time feature (same logic as flat observation)
        step = int(self.max_moves - self.agent.moves_remaining)
        time_normalized = step / self.max_moves if self.max_moves > 0 else 0.0

        node_features: List[List[float]] = []
        node_types: List[float] = []
        node_values: List[float] = []

        for node in nodes:
            node_types.append(node.type_id)
            if isinstance(node, ConstantExpression):
                node_values.append(float(node.value) if node.value else 0.0)
            else:
                node_values.append(0.0)

        # Apply normalization if requested
        if normalize and node_types:
            node_types_array = np.array(node_types, dtype=np.float32)
            if node_types_array.sum() != 0.0:
                node_types_array = (node_types_array - node_types_array.min()) / (
                    node_types_array.max() - node_types_array.min() + 1e-32
                )
            node_types = node_types_array.tolist()  # type:ignore

        if normalize and node_values:
            node_values_array = np.array(node_values, dtype=np.float32)
            if node_values_array.sum() != 0.0:
                node_values_array = (node_values_array - node_values_array.min()) / (
                    node_values_array.max() - node_values_array.min() + 1e-32
                )
            node_values = node_values_array.tolist()  # type:ignore

        # Build consistent feature vectors: [type_id, value, time, is_leaf]
        for i, node in enumerate(nodes):
            features = [
                node_types[i],
                node_values[i],
                time_normalized,
                1.0 if node.is_leaf() else 0.0,
            ]
            node_features.append(features)

        return node_features, 4  # feature_dim = 4

    def to_graph_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        hash_type: Optional[ProblemTypeIntList] = None,
        parser: Optional[ExpressionParser] = None,
        normalize: bool = True,
        max_seq_len: Optional[int] = None,
    ) -> MathyGraphObservation:
        """Convert a state into a graph observation for predictive coding"""
        if parser is None:
            parser = ExpressionParser()
        if hash_type is None:
            hash_type = self.get_problem_hash()
        if max_seq_len is None:
            max_seq_len = 128  # Default fallback

        expression = parser.parse(self.agent.problem)
        nodes: List[MathExpression] = expression.to_list()
        n = len(nodes)

        # Get consistent node features
        node_features_list, feature_dim = self._get_node_features(nodes, normalize)

        # Create adjacency matrix using the tree structure
        adjacency = np.zeros((n, n), dtype=np.float32)
        node_to_idx = {id(node): i for i, node in enumerate(nodes)}

        for i, node in enumerate(nodes):
            # Parent -> child edges (top-down prediction)
            if node.left and id(node.left) in node_to_idx:
                left_idx = node_to_idx[id(node.left)]
                adjacency[i][left_idx] = 1.0  # parent predicts left child

            if node.right and id(node.right) in node_to_idx:
                right_idx = node_to_idx[id(node.right)]
                adjacency[i][right_idx] = 1.0  # parent predicts right child

        # Pad node features to max_seq_len
        padded_features = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
        if node_features_list:
            node_features_array = np.array(node_features_list, dtype=np.float32)
            padded_features[:n] = node_features_array

        # Pad adjacency matrix
        padded_adjacency = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)
        padded_adjacency[:n, :n] = adjacency

        # Process action mask
        if move_mask is not None:
            action_mask = self._process_action_mask(move_mask, max_seq_len)
        else:
            action_mask = np.zeros(self.num_rules * max_seq_len, dtype=np.float32)

        return MathyGraphObservation(
            node_features=padded_features,
            adjacency=padded_adjacency,
            action_mask=action_mask,
            num_nodes=n,
        )

    def to_message_passing_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        parser: Optional[ExpressionParser] = None,
        max_seq_len: Optional[int] = None,
        normalize: bool = True,
    ) -> MathyMessagePassingObservation:
        """Format for graph neural networks with explicit edge types"""
        if parser is None:
            parser = ExpressionParser()
        if max_seq_len is None:
            max_seq_len = 128  # Default fallback

        expression = parser.parse(self.agent.problem)
        nodes = expression.to_list()
        actual_nodes = len(nodes)

        # Get consistent node features
        node_features_list, feature_dim = self._get_node_features(nodes, normalize)

        # Pad node features
        padded_features = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
        if node_features_list:
            node_features_array = np.array(node_features_list, dtype=np.float32)
            padded_features[:actual_nodes] = node_features_array

        # Edge list with types
        edges: List[List[int]] = []
        edge_types: List[int] = []
        node_to_idx = {id(node): i for i, node in enumerate(nodes)}

        for i, node in enumerate(nodes):
            # Parent -> Left child
            if node.left and id(node.left) in node_to_idx:
                left_idx = node_to_idx[id(node.left)]
                edges.append([i, left_idx])
                edge_types.append(0)  # 0 = left edge

            # Parent -> Right child
            if node.right and id(node.right) in node_to_idx:
                right_idx = node_to_idx[id(node.right)]
                edges.append([i, right_idx])
                edge_types.append(1)  # 1 = right edge

        # Pad edge information
        max_edges = max_seq_len * 2
        actual_edges = len(edges)

        padded_edge_index = np.zeros((2, max_edges), dtype=np.int64)
        padded_edge_types = np.zeros((max_edges,), dtype=np.int64)

        if actual_edges > 0:
            edge_array = np.array(edges, dtype=np.int64).T  # PyG format (2, num_edges)
            padded_edge_index[:, :actual_edges] = edge_array
            padded_edge_types[:actual_edges] = np.array(edge_types, dtype=np.int64)

        # Process action mask
        if move_mask is not None:
            action_mask = self._process_action_mask(move_mask, max_seq_len)
        else:
            action_mask = np.zeros(self.num_rules * max_seq_len, dtype=np.float32)

        return MathyMessagePassingObservation(
            node_features=padded_features,
            edge_index=padded_edge_index,
            edge_types=padded_edge_types,
            action_mask=action_mask,
            num_nodes=actual_nodes,
            num_edges=actual_edges,
        )

    def to_flat_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        hash_type: Optional[ProblemTypeIntList] = None,
        parser: Optional[ExpressionParser] = None,
        normalize: bool = True,
        max_seq_len: Optional[int] = None,
    ) -> NDArray[np.float32]:
        """Convert a state into a flat observation array"""
        if parser is None:
            parser = ExpressionParser()
        if hash_type is None:
            hash_type = self.get_problem_hash()
        if max_seq_len is None:
            max_seq_len = 128  # Default fallback

        expression = parser.parse(self.agent.problem)
        nodes: List[MathExpression] = expression.to_list()
        vectors: NodeIntList = []
        values: NodeValuesFloatList = []

        for node in nodes:
            vectors.append(node.type_id)
            if isinstance(node, ConstantExpression):
                assert node.value is not None
                values.append(float(node.value))
            else:
                values.append(0.0)

        # The "types" and "values" can be normalized 0-1
        if normalize is True:
            # https://bit.ly/3irAalH
            x: Any = np.asarray(values, dtype=np.float32)
            if x.sum() != 0.0:
                x = (x - min(x)) / (max(x) - min(x) + 1e-32)
            values = x.tolist()
            x = np.asarray(vectors, dtype=np.float32)
            if x.sum() != 0.0:
                x = (x - min(x)) / (max(x) - min(x) + 1e-32)
            vectors = x.tolist()

        # Pass a 0-1 value indicating the relative episode time where 0.0 is
        # the episode start, and 1.0 is the episode end as indicated by the
        # maximum allowed number of actions.
        step = int(self.max_moves - self.agent.moves_remaining)
        time = int(step / self.max_moves * 10)

        # Pad observations to max_seq_len
        values = pad_array(values, max_seq_len, 0.0)
        vectors = pad_array(vectors, max_seq_len, 0.0)

        # Process action mask
        if move_mask is not None:
            flat_mask = self._process_action_mask(move_mask, max_seq_len)
        else:
            flat_mask = np.zeros(self.num_rules * max_seq_len, dtype=np.float32)

        # Build the flat observation array
        nodes_array = np.array(vectors, dtype="float32")
        values_array = np.array(values, dtype="float32")
        type_time = np.array(hash_type + [time], dtype="float32")
        obs = np.concatenate([type_time, nodes_array, values_array, flat_mask], axis=0)
        return obs

    def to_hierarchical_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        parser: Optional[ExpressionParser] = None,
        max_seq_len: Optional[int] = None,
        normalize: bool = True,
    ) -> MathyHierarchicalObservation:
        """Group nodes by tree depth for hierarchical predictive coding"""
        if parser is None:
            parser = ExpressionParser()
        if max_seq_len is None:
            max_seq_len = 128  # Default fallback

        expression = parser.parse(self.agent.problem)
        root = expression  # assuming root is the full expression

        # Group nodes by depth level
        levels: Dict[int, List[MathExpression]] = {}
        node_to_level = {}

        def assign_levels(node: Optional[MathExpression], depth: int = 0) -> None:
            if node is None:
                return

            if depth not in levels:
                levels[depth] = []

            levels[depth].append(node)
            node_to_level[id(node)] = depth

            if hasattr(node, "left") and node.left:
                assign_levels(node.left, depth + 1)
            if hasattr(node, "right") and node.right:
                assign_levels(node.right, depth + 1)

        assign_levels(root)

        # Collect all nodes and their level information
        all_nodes: List[MathExpression] = []
        level_indices: List[int] = []
        max_depth = max(levels.keys()) if levels else 0

        for level, level_nodes in levels.items():
            for node in level_nodes:
                all_nodes.append(node)
                clamped_level = max(0, min(int(level), max_seq_len // 4 - 1))
                level_indices.append(clamped_level)

        # Get consistent node features
        node_features_list, feature_dim = self._get_node_features(all_nodes, normalize)
        actual_nodes = len(all_nodes)

        # Pad features and level indices
        padded_features = np.zeros((max_seq_len, feature_dim), dtype=np.float32)
        padded_levels = np.zeros((max_seq_len,), dtype=np.int32)

        if actual_nodes > 0:
            node_features_array = np.array(node_features_list, dtype=np.float32)
            level_indices_array = np.array(level_indices, dtype=np.int32)

            padded_features[:actual_nodes] = node_features_array
            padded_levels[:actual_nodes] = level_indices_array

        # Process action mask
        if move_mask is not None:
            action_mask = self._process_action_mask(move_mask, max_seq_len)
        else:
            action_mask = np.zeros(self.num_rules * max_seq_len, dtype=np.float32)

        return MathyHierarchicalObservation(
            node_features=padded_features,
            level_indices=padded_levels,
            action_mask=action_mask,
            max_depth=min(max_depth, max_seq_len // 4),
            num_nodes=actual_nodes,
        )

    @classmethod
    def from_string(cls, input_string: str) -> "MathyEnvState":
        """Convert a string representation of state into a state object"""
        sep = "@"
        history_sep = ","
        # remove any padding from string inputs (if added by to_np call)
        input_string = input_string.rstrip()
        inputs = input_string.split(sep)
        state = MathyEnvState()
        state.max_moves = int(inputs[0])
        state.num_rules = int(inputs[1])
        state.agent = MathyAgentState(
            moves_remaining=int(inputs[2]),
            problem=str(inputs[3]),
            problem_type=str(inputs[4]),
            reward=float(inputs[5]),
            history=[],
        )
        history = inputs[6:]
        for step in history:
            raw, action, focus = step.split(history_sep)
            state.agent.history.append(
                MathyEnvStateStep(raw, (int(action), int(focus)))
            )
        return state

    def to_string(
        self,
    ) -> str:
        """Convert a state object into a string representation"""
        sep = "@"
        assert self.agent is not None, "invalid state"
        out = [
            str(self.max_moves),
            str(self.num_rules),
            str(self.agent.moves_remaining),
            str(self.agent.problem),
            str(self.agent.problem_type),
            str(self.agent.reward),
        ]
        for step in self.agent.history:
            out.append(
                ",".join([str(step.raw), str(step.action[0]), str(step.action[1])])
            )
        return sep.join(out)

    @classmethod
    def from_np(cls, input_bytes: NDArray[np.uint8]) -> "MathyEnvState":
        """Convert a numpy object into a state object"""
        input_string = "".join(
            [chr(int(o)) for o in input_bytes.tolist()]  # type:ignore
        )
        state = cls.from_string(input_string)
        return state

    def to_np(self, pad_to: Optional[int] = None) -> NDArray[np.uint8]:
        """Convert a state object into a numpy representation"""
        string = self.to_string()
        if pad_to is not None:
            assert pad_to >= len(string), "input is larger than pad size!"
            to_pad = pad_to - len(string)
            string += " " * to_pad
        return np.array([ord(c) for c in string], dtype=np.uint8)
