from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from mathy_core.expressions import ConstantExpression, MathExpression, MathTypeKeys
from mathy_core.parser import ExpressionParser

from zlib import adler32

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
MathyObservation.nodes.__doc__ = "tree node types in the current environment state shape=[n,]" # noqa
MathyObservation.mask.__doc__ = "0/1 mask where 0 indicates an invalid action shape=[n,]" # noqa
MathyObservation.values.__doc__ = "tree node value sequences, with non number indices set to 0.0 shape=[n,]" # noqa
MathyObservation.type.__doc__ = "two column hash of problem environment type shape=[2,]" # noqa
MathyObservation.time.__doc__ = "float value between 0.0 and 1.0 indicating the time elapsed shape=[1,]" # noqa
# fmt: on


class MathyEnvStateStep(NamedTuple):
    """Capture summarized environment state for a previous timestep so the
    agent can use context from its history when making new predictions."""

    raw: str
    action: ActionType


# fmt: off
MathyEnvStateStep.raw.__doc__ = "the input text at the timestep" # noqa
MathyEnvStateStep.action.__doc__ = "a tuple indicating the chosen action and the node it was applied to" # noqa
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
        problem: str = None,
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
            hashes = np.asfarray(
                [
                    adler32(type_str.encode(encoding="utf-8")),
                    adler32(type_str[::-1].encode(encoding="utf-8")),
                    adler32(type_str.upper().encode(encoding="utf-8")),
                    adler32(type_str.replace(".", "").encode(encoding="utf-8")),
                ]
            )
            if hashes.sum() != 0.0:
                hashes = (hashes - min(hashes)) / (max(hashes) - min(hashes))
            values = hashes.tolist()

            _problem_hash_cache[self.agent.problem_type] = values

        return _problem_hash_cache[self.agent.problem_type]

    def to_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        hash_type: Optional[ProblemTypeIntList] = None,
        parser: Optional[ExpressionParser] = None,
        normalize: bool = True,
        max_seq_len: Optional[int] = None,
    ) -> MathyObservation:
        """Convert a state into an observation"""
        if parser is None:
            parser = ExpressionParser()
        if hash_type is None:
            hash_type = self.get_problem_hash()
        expression = parser.parse(self.agent.problem)
        nodes: List[MathExpression] = expression.to_list()
        vectors: NodeIntList = []
        values: NodeValuesFloatList = []
        if move_mask is None:
            move_mask = np.zeros(len(nodes)).tolist()
        assert move_mask is not None
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
            x = np.asfarray(values)
            if x.sum() != 0.0:
                x = (x - min(x)) / (max(x) - min(x) + 1e-32)
            values = x.tolist()
            x = np.asfarray(vectors)
            if x.sum() != 0.0:
                x = (x - min(x)) / (max(x) - min(x) + 1e-32)
            vectors = x.tolist()

        # Pass a 0-1 value indicating the relative episode time where 0.0 is
        # the episode start, and 1.0 is the episode end as indicated by the
        # maximum allowed number of actions.
        step = int(self.max_moves - self.agent.moves_remaining)
        time = int(step / self.max_moves * 10)

        # Pad observations to max_seq_len if specified
        if max_seq_len is not None:
            values = pad_array(values, max_seq_len, 0.0)
            vectors = pad_array(vectors, max_seq_len, 0.0)
            move_mask = [pad_array(m, max_seq_len, 0) for m in move_mask]

        return MathyObservation(
            nodes=vectors, mask=move_mask, values=values, type=hash_type, time=[time]
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
    def from_np(cls, input_bytes: np.ndarray) -> "MathyEnvState":
        """Convert a numpy object into a state object"""
        input_string = "".join([chr(o) for o in input_bytes.tolist()])
        state = cls.from_string(input_string)
        return state

    def to_np(self, pad_to: int = None) -> np.ndarray:
        """Convert a state object into a numpy representation"""
        string = self.to_string()
        if pad_to is not None:
            assert pad_to >= len(string), "input is larger than pad size!"
            to_pad = pad_to - len(string)
            string += " " * to_pad
        return np.array([ord(c) for c in string])
