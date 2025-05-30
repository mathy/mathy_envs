import random
from typing import Any

import pytest
from mathy_core.rules import AssociativeSwapRule, CommutativeSwapRule

from mathy_envs import MathyEnv, MathyEnvState
from mathy_envs.env import INVALID_ACTION_RESPONSES
from mathy_envs.envs.poly_simplify import PolySimplify
from mathy_envs.time_step import is_terminal_transition
from mathy_envs.types import EnvRewards


def test_env_init():
    env = MathyEnv()
    assert env is not None
    # Default env is abstract and cannot be directly used for problem solving
    with pytest.raises(NotImplementedError):
        env.get_initial_state()
    with pytest.raises(NotImplementedError):
        env.get_env_namespace()


def test_env_init_check_invalid_action_response():
    with pytest.raises(ValueError):
        MathyEnv(invalid_action_response="something_wrong")  # type:ignore
    option: Any
    for option in INVALID_ACTION_RESPONSES:
        assert MathyEnv(invalid_action_response=option) is not None


def test_env_action_masks():
    problem = "4x + 2x"
    env = MathyEnv(invalid_action_response="raise")
    env_state = MathyEnvState(problem=problem, max_moves=35)
    valid_mask = env.get_valid_moves(env_state)
    assert len(valid_mask) == len(env.rules)
    assert len(valid_mask[0]) == len(env.parser.parse(problem).to_list())


def test_env_random_actions():
    env = MathyEnv(invalid_action_response="raise")
    state = MathyEnvState(problem="4x + 2x + 7 + y")
    expression = env.parser.parse(state.agent.problem)
    # Can select random actions of the given type
    action = env.random_action(expression, AssociativeSwapRule)
    env.get_next_state(state, action)

    # Can select random actions from all types
    state = MathyEnvState(problem="4x + 2x + 7 + y")
    expression = env.parser.parse(state.agent.problem)
    action = env.random_action(expression)
    env.get_next_state(state, action)


def test_env_invalid_action_behaviors():
    problem = "4x + 2x"
    env = MathyEnv(invalid_action_response="raise")
    env_state = MathyEnvState(problem=problem, max_moves=35)
    rule_actions = env.get_valid_moves(env_state)
    rule_indices = [i for i, value in enumerate(rule_actions) if 1 not in value]
    random.shuffle(rule_indices)
    rule_nodes = rule_actions[rule_indices[0]]
    node_indices = [i for i, value in enumerate(rule_nodes) if value == 0]
    action = (rule_indices[0], node_indices[0])

    # Raise an error when selecting an invalid action
    env_state = MathyEnvState(problem=problem, max_moves=35)
    with pytest.raises(ValueError):
        env.get_next_state(env_state, action)

    # Penalize the agent for choosing an invalid action
    env = MathyEnv(invalid_action_response="penalize")
    env_state = MathyEnvState(problem=problem, max_moves=35)
    _, transition, _ = env.get_next_state(env_state, action)
    assert transition.reward == EnvRewards.INVALID_MOVE
    assert is_terminal_transition(transition) is False

    # End the episode when choosing an invalid action
    env = MathyEnv(invalid_action_response="terminal")
    env_state = MathyEnvState(problem=problem, max_moves=35)
    _, transition, _ = env.get_next_state(env_state, action)
    # a transition is returned with error_invalid=False
    assert is_terminal_transition(transition) is True


def test_env_terminal_conditions():
    expectations = [
        ("70656 * (x^2 * z^6)", True),
        ("b * (44b^2)", False),
        ("z * (1274z^2)", False),
        ("4x^2", True),
        ("100y * x + 2", True),
        ("10y * 10x + 2", False),
        ("10y + 1000y * (y * z)", False),
        ("4 * (5y + 2)", False),
        ("2", True),
        ("4x * 2", False),
        ("4x * 2x", False),
        ("4x + 2x", False),
        ("4 + 2", False),
        ("3x + 2y + 7", True),
        ("3x^2 + 2x + 7", True),
        ("3x^2 + 2x^2 + 7", False),
    ]

    # Valid solutions but out of scope so they aren't counted as wins.
    #
    # This works because the problem sets exclude this type of > 2 term
    # polynomial expressions
    out_of_scope_valid = []

    env = PolySimplify()
    for text, is_win in expectations + out_of_scope_valid:
        env_state = MathyEnvState(problem=text)
        reward = env.get_state_transition(env_state)
        assert text == text and env.is_terminal_state(env_state) == bool(is_win)
        assert text == text and is_terminal_transition(reward) == bool(is_win)


@pytest.mark.parametrize("pretty", [True, False])
def test_env_print_history(pretty: bool):
    env = PolySimplify()
    env_state = MathyEnvState(problem="4+2")
    for i in range(10):
        env_state = env_state.get_out_state(
            problem="4+2" if i % 2 == 0 else "2+4",
            moves_remaining=10 - i,
            action=(1, 1),
        )
    env.print_history(env_state, pretty=pretty)


def test_env_finalize_state():
    env = PolySimplify()

    env_state = MathyEnvState(problem="4x + 2x").get_out_state(
        problem="1337", action=(1, 1), moves_remaining=0
    )
    with pytest.raises(ValueError):
        env.finalize_state(env_state)

    env_state = MathyEnvState(problem="4x + 2x").get_out_state(
        problem="4x + 2", action=(1, 1), moves_remaining=0
    )
    with pytest.raises(ValueError):
        env.finalize_state(env_state)

    env_state = MathyEnvState(problem="4x + 2x").get_out_state(
        problem="4x + 2y", action=(1, 1), moves_remaining=0
    )
    with pytest.raises(ValueError):
        env.finalize_state(env_state)


def test_mathy_env_can_timestep_loop():
    env = MathyEnv()
    assert env is not None
    problem = "5y * 9x + 8z + 8x + 3z * 10y * 11x + 10y"
    env_state = MathyEnvState(problem=problem, max_moves=35)
    for i in range(3):
        rule_actions = env.get_valid_moves(env_state)
        rule_indices = [i for i, value in enumerate(rule_actions) if 1 in value]
        random.shuffle(rule_indices)
        rule_nodes = rule_actions[rule_indices[0]]
        node_indices = [i for i, value in enumerate(rule_nodes) if value == 1]
        env_state, value, changed = env.get_next_state(
            env_state, (rule_indices[0], node_indices[0])
        )
    assert env_state.to_observation([[], []]) is not None


def test_mathy_env_invalid_action_behaviors():
    env = MathyEnv()
    assert env is not None
    problem = "5y * 9x + 8z + 8x + 3z * 10y * 11x + 10y"
    env_state = MathyEnvState(problem=problem, max_moves=35)
    for i in range(3):
        rule_actions = env.get_valid_moves(env_state)
        rule_indices = [i for i, value in enumerate(rule_actions) if 1 in value]
        random.shuffle(rule_indices)
        rule_nodes = rule_actions[rule_indices[0]]
        node_indices = [i for i, value in enumerate(rule_nodes) if value == 1]
        env_state, value, changed = env.get_next_state(
            env_state, (rule_indices[0], node_indices[0])
        )
    assert env_state.to_observation([[], []]) is not None


def test_mathy_env_preferred_term_commute():
    rule_idx = 1
    problem = "5y"
    env_state = MathyEnvState(problem=problem, max_moves=1)

    env = MathyEnv(preferred_term_commute=False)
    assert isinstance(env.rules[rule_idx], CommutativeSwapRule), "update rule_idx"
    commute_nodes = env.get_valid_moves(env_state)[rule_idx]
    assert 1 not in commute_nodes, "shouldn't be able to commute preferred order terms"

    env = MathyEnv(preferred_term_commute=True)
    commute_nodes = env.get_valid_moves(env_state)[rule_idx]
    assert 1 in commute_nodes, "should be able to commute preferred order terms"


def test_mathy_env_previous_state_penalty():
    """When previous_state_penalty=True, a negative reward is given when
    revisiting already seen problem states. If an agent revisits the
    state too many times, the game ends."""

    # We define the input problem with 3 nodes for simplicity
    # "x * y" == ["x","*","y"]
    # Because the tree is small and balanced, we can commute the
    # same node over and over to flip back-and-forth between x * y
    # and y * x.
    problem = "x * y"
    env = MathyEnv(previous_state_penalty=True)
    rule_idx = 1
    node_idx = 1
    assert isinstance(env.rules[rule_idx], CommutativeSwapRule), "update rule_idx"
    action = (rule_idx, node_idx)
    env_state = MathyEnvState(problem=problem, max_moves=10)
    # Commute the first time so we are revisit the initial state
    # as we apply the same action again.
    env_state, _, _ = env.get_next_state(env_state, action)

    # After three visits to the same state, the game ends.
    last_penalty = 0.0
    found_terminal = False
    for i in range(3):
        env_state, transition, changed = env.get_next_state(env_state, action)
        assert transition.reward < 0.0
        # The penalty scales up based on the number of visits to the state
        assert transition.reward < last_penalty
        last_penalty = transition.reward

        if i < 2:
            # Visit the opposite state and ignore it (we only care about revisiting
            # the initial state)
            env_state, _, _ = env.get_next_state(env_state, action)
        else:
            # After the third time, we should receive a terminal transition
            assert is_terminal_transition(transition) is True
            found_terminal = True

    assert found_terminal is True, "did not receive expected terminal transition"


def test_mathy_env_win_conditions():
    expectations = [
        ("70656 * (x^2 * z^6)", True),
        ("b * (44b^2)", False),
        ("z * (1274z^2)", False),
        ("4x^2", True),
        ("100y * x + 2", True),
        ("10y * 10x + 2", False),
        ("10y + 1000y * (y * z)", False),
        ("4 * (5y + 2)", False),
        ("2", True),
        ("4x * 2", False),
        ("4x * 2x", False),
        ("4x + 2x", False),
        ("4 + 2", False),
        ("3x + 2y + 7", True),
        ("3x^2 + 2x + 7", True),
        ("3x^2 + 2x^2 + 7", False),
    ]

    # Valid solutions but out of scope so they aren't counted as wins.
    #
    # This works because the problem sets exclude this type of > 2 term
    # polynomial expressions
    out_of_scope_valid = []

    env = PolySimplify()
    for text, is_win in expectations + out_of_scope_valid:
        env_state = MathyEnvState(problem=text)
        reward = env.get_state_transition(env_state)
        assert text == text and is_terminal_transition(reward) == bool(is_win)
