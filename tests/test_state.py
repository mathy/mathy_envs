from typing import List, Tuple

from mathy_envs.envs.poly_simplify import PolySimplify
from mathy_envs.state import MathyEnvState


def test_state_to_observation():
    """to_observation has defaults to allow calling with no arguments"""
    env_state = MathyEnvState(problem="4x+2")
    assert env_state.to_observation() is not None


def test_state_encodes_hierarchy():
    """Verify that the observation generated encodes hierarchy properly
    so the model can determine the precise nodes to act on"""

    diff_pairs: List[Tuple[str, str]] = [
        ("4x + (3u + 7x + 3u) + 4u", "4x + 3u + 7x + 3u + 4u"),
        ("7c * 5", "7 * (c * 5)"),
        ("5v + 20b + (10v + 7b)", "5v + 20b + 10v + 7b"),
        ("5s + 60 + 12s + s^2", "5s + 60 + (12s + s^2)"),
    ]
    env = PolySimplify()

    for one, two in diff_pairs:
        state_one = MathyEnvState(problem=one)
        obs_one = state_one.to_observation(env.get_valid_moves(state_one))

        state_two = MathyEnvState(problem=two)
        obs_two = state_two.to_observation(env.get_valid_moves(state_two))

        assert obs_one.nodes != obs_two.nodes


def test_state_sanity():
    state = MathyEnvState(problem="4+4")
    assert state is not None


def test_state_encode_player():
    env_state = MathyEnvState(problem="4x+2")
    env_state = env_state.get_out_state(
        problem="2+4x", moves_remaining=10, action=(0, 0)
    )
    agent = env_state.agent
    assert agent.problem == "2+4x"
    assert agent.moves_remaining == 10
    assert agent.action == (0, 0)


def test_state_serialize_string():
    env_state = MathyEnvState(problem="4x+2")
    for i in range(10):
        env_state = env_state.get_out_state(
            problem="2+4x", moves_remaining=10 - i, action=(i, i)
        )

    state_str = env_state.to_string()
    compare = MathyEnvState.from_string(state_str)
    assert env_state.agent.problem == compare.agent.problem
    assert env_state.agent.moves_remaining == compare.agent.moves_remaining
    for one, two in zip(env_state.agent.history, compare.agent.history):
        assert one.raw == two.raw
        assert one.action == two.action


def test_state_serialize_numpy():
    env_state = MathyEnvState(problem="4x+2")
    for i in range(10):
        env_state = env_state.get_out_state(
            problem="2+4x", moves_remaining=10 - i, action=(i, i)
        )

    state_np = env_state.to_np()
    compare = MathyEnvState.from_np(state_np)
    assert env_state.agent.problem == compare.agent.problem
    assert env_state.agent.moves_remaining == compare.agent.moves_remaining
    for one, two in zip(env_state.agent.history, compare.agent.history):
        assert one.raw == two.raw
        assert one.action == two.action
