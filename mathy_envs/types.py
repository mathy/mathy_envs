import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple, Tuple

# Use typing_extensions for Python < 3.8
if sys.version_info < (3, 8):
    from typing_extensions import Literal  # noqa
    from typing_extensions import get_args  # noqa
else:
    from typing import Literal  # noqa
    from typing import get_args  # noqa


class MathyEnvDifficulty(Enum):
    # The simplest form of problems that demonstrate a task
    easy = "easy"
    # Challenging problems that involve more terms
    normal = "normal"
    # Difficult problems that have intentionally large expressions
    hard = "hard"


MathyEnvDifficultyValue = Literal["easy", "normal", "hard"]

assert set(get_args(MathyEnvDifficultyValue)) == {
    member.value for member in MathyEnvDifficulty
}


@dataclass
class MathyEnvProblemArgs:
    difficulty: MathyEnvDifficulty = MathyEnvDifficulty.easy


class MathyEnvProblem(NamedTuple):
    """Summarize an environment-specific problem that was generated with
    a tuple of (text, complexity, type) where:
     - "text" is the text content of the generated problem
     - "complexity" is an integer value that represents the number of
       terms in the problem text.
     - "type" is a dot namespaced string, e.g. "mathy.poly.simplify"
    """

    text: str
    complexity: int
    type: str


class EnvRewards:
    LOSE = -1.0
    WIN = 1.0
    HELPFUL_MOVE = 0.01
    UNHELPFUL_MOVE = -0.01
    TIMESTEP = -0.01
    PREVIOUS_LOCATION = -0.02
    INVALID_MOVE = -0.5


ActionType = Tuple[int, int]
ActionList = List[ActionType]
RewardList = List[float]
ValueList = List[float]
