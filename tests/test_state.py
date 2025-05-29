from typing import List, Tuple

import numpy as np
import pytest

from mathy_envs.envs.poly_simplify import PolySimplify
from mathy_envs.state import (
    MathyEnvState,
    MathyGraphObservation,
    MathyHierarchicalObservation,
    MathyMessagePassingObservation,
    MathyObservation,
    ObservationType,
)


def test_state_to_observation():
    """to_observation has defaults to allow calling with no arguments"""
    env_state = MathyEnvState(problem="4x+2")
    assert env_state.to_observation() is not None


def test_state_unified_observation_interface():
    """Test the unified to_observation method with different observation types"""
    env_state = MathyEnvState(problem="4x+2*3")

    # Test flat observation (default)
    flat_obs = env_state.to_observation()
    assert isinstance(flat_obs, MathyObservation)

    # Test explicit flat observation
    flat_obs_explicit = env_state.to_observation(obs_type=ObservationType.FLAT)
    assert isinstance(flat_obs_explicit, MathyObservation)
    assert flat_obs == flat_obs_explicit

    # Test graph observation
    graph_obs = env_state.to_observation(obs_type=ObservationType.GRAPH)
    assert isinstance(graph_obs, MathyGraphObservation)

    # Test hierarchical observation
    hier_obs = env_state.to_observation(obs_type=ObservationType.HIERARCHICAL)
    assert isinstance(hier_obs, MathyHierarchicalObservation)

    # Test message passing observation
    mp_obs = env_state.to_observation(obs_type=ObservationType.MESSAGE_PASSING)
    assert isinstance(mp_obs, MathyMessagePassingObservation)


def test_state_graph_observation_structure():
    """Test graph observation properties and structure"""
    env_state = MathyEnvState(problem="4x+2*3")
    obs = env_state.to_observation(obs_type=ObservationType.GRAPH)

    # Check basic properties
    assert hasattr(obs, "node_features")
    assert hasattr(obs, "adjacency")
    assert hasattr(obs, "mask")
    assert hasattr(obs, "type")
    assert hasattr(obs, "time")

    # Check shapes make sense
    n_nodes = obs.node_features.shape[0]
    assert obs.adjacency.shape == (n_nodes, n_nodes)
    assert len(obs.mask) == n_nodes

    # Check feature dimensions
    assert obs.node_features.shape[1] == 2  # [type_id, value]

    # Check adjacency matrix properties
    assert obs.adjacency.dtype == np.float32
    assert np.all((obs.adjacency == 0) | (obs.adjacency == 1))  # Binary matrix

    # For "4x+2*3", we should have a tree structure
    # Root (+) should connect to left (4x) and right (2*3)
    assert np.sum(obs.adjacency) > 0  # Should have some connections


def test_state_hierarchical_observation_structure():
    """Test hierarchical observation groups nodes by depth correctly"""
    env_state = MathyEnvState(problem="(4+2)*3")  # 3 levels: *, +, leaves
    obs = env_state.to_observation(obs_type=ObservationType.HIERARCHICAL)

    # Check basic structure
    assert hasattr(obs, "levels")
    assert hasattr(obs, "max_depth")
    assert hasattr(obs, "root_level")

    # Check depth properties
    assert obs.root_level == 0
    assert obs.max_depth >= 0
    assert len(obs.levels) == obs.max_depth + 1

    # Check each level has proper structure
    for depth, level_data in obs.levels.items():
        assert "features" in level_data
        assert "parent_connections" in level_data
        assert isinstance(level_data["features"], np.ndarray)
        assert level_data["features"].shape[1] == 2  # [type_id, value]

        # Parent connections should reference valid levels/indices
        for conn in level_data["parent_connections"]:
            parent_level, parent_idx, child_level, child_idx = conn
            assert parent_level in obs.levels
            assert child_level == depth
            assert parent_idx < len(obs.levels[parent_level]["features"])


def test_state_message_passing_observation_structure():
    """Test message passing observation edge format"""
    env_state = MathyEnvState(problem="4x+2")
    obs = env_state.to_observation(obs_type=ObservationType.MESSAGE_PASSING)

    # Check basic properties
    assert hasattr(obs, "node_features")
    assert hasattr(obs, "edge_index")
    assert hasattr(obs, "edge_types")
    assert hasattr(obs, "num_nodes")

    # Check dimensions match
    assert obs.node_features.shape[0] == obs.num_nodes
    assert obs.node_features.shape[1] == 3  # [type_id, value, is_leaf]

    # Check edge format (PyTorch Geometric style)
    if obs.edge_index.size > 0:
        assert obs.edge_index.shape[0] == 2  # [source, target]
        assert obs.edge_index.shape[1] == len(obs.edge_types)

        # All edge indices should be valid node indices
        assert np.all(obs.edge_index >= 0)
        assert np.all(obs.edge_index < obs.num_nodes)

        # Edge types should be 0 (left) or 1 (right)
        assert np.all((obs.edge_types == 0) | (obs.edge_types == 1))


def test_state_observation_consistency():
    """Test that different observation types represent the same underlying problem"""
    env_state = MathyEnvState(problem="2*x+3")

    flat_obs = env_state.to_observation(obs_type=ObservationType.FLAT)
    graph_obs = env_state.to_observation(obs_type=ObservationType.GRAPH)
    hier_obs = env_state.to_observation(obs_type=ObservationType.HIERARCHICAL)
    mp_obs = env_state.to_observation(obs_type=ObservationType.MESSAGE_PASSING)

    # All should represent the same number of nodes
    n_nodes_flat = len(flat_obs.nodes)
    n_nodes_graph = graph_obs.node_features.shape[0]
    n_nodes_hier = sum(len(level["features"]) for level in hier_obs.levels.values())
    n_nodes_mp = mp_obs.num_nodes

    assert n_nodes_flat == n_nodes_graph == n_nodes_hier == n_nodes_mp

    # Time information should be consistent
    assert flat_obs.time == graph_obs.time
    # (Hierarchical and MP don't include time in current implementation)


def test_state_complex_expression_observations():
    """Test observations work correctly for complex mathematical expressions"""
    env_state = MathyEnvState(problem="(a+b)*(c-d)/e")

    # Test each observation type handles complexity
    flat_obs = env_state.to_observation(obs_type=ObservationType.FLAT)
    graph_obs = env_state.to_observation(obs_type=ObservationType.GRAPH)
    hier_obs = env_state.to_observation(obs_type=ObservationType.HIERARCHICAL)
    mp_obs = env_state.to_observation(obs_type=ObservationType.MESSAGE_PASSING)

    # Should have multiple nodes for this complex expression
    assert len(flat_obs.nodes) > 5
    assert graph_obs.node_features.shape[0] > 5
    assert sum(len(level["features"]) for level in hier_obs.levels.values()) > 5
    assert mp_obs.num_nodes > 5

    # Hierarchical should have multiple levels
    assert hier_obs.max_depth > 2

    # Graph should have connections
    assert np.sum(graph_obs.adjacency) > 0

    # Message passing should have edges
    assert mp_obs.edge_index.size > 0


def test_state_observation_normalization():
    """Test normalization works across observation types"""
    env_state = MathyEnvState(problem="100+200")

    # Test with normalization (default)
    norm_flat = env_state.to_observation(normalize=True)
    norm_graph = env_state.to_observation(
        obs_type=ObservationType.GRAPH, normalize=True
    )

    # Values should be normalized to [0, 1] range
    assert np.max(norm_flat.values) <= 1.0
    assert np.min(norm_flat.values) >= 0.0

    # Graph node features should be normalized
    assert np.max(norm_graph.node_features) <= 1.0
    assert np.min(norm_graph.node_features) >= 0.0

    # Test without normalization
    unnorm_flat = env_state.to_observation(normalize=False)

    # Should have larger raw values
    assert np.max(unnorm_flat.values) > 1.0


def test_state_edge_case_observations():
    """Test observations work for edge cases"""
    # Single node
    single_state = MathyEnvState(problem="5")
    single_obs = single_state.to_observation(obs_type=ObservationType.GRAPH)
    assert single_obs.node_features.shape[0] == 1
    assert np.sum(single_obs.adjacency) == 0  # No edges for single node

    # Variable only
    var_state = MathyEnvState(problem="x")
    var_obs = var_state.to_observation(obs_type=ObservationType.MESSAGE_PASSING)
    assert var_obs.num_nodes == 1
    assert var_obs.edge_index.size == 0  # No edges

    # Test hierarchical with single level
    hier_obs = single_state.to_observation(obs_type=ObservationType.HIERARCHICAL)
    assert hier_obs.max_depth == 0
    assert len(hier_obs.levels) == 1


def test_state_invalid_observation_type():
    """Test that invalid observation types raise appropriate errors"""
    env_state = MathyEnvState(problem="4x+2")

    with pytest.raises(ValueError, match="Unknown observation type"):
        # This should raise an error since we're bypassing the enum
        env_state.to_observation(obs_type="invalid_type")


def test_state_to_observation_normalization():
    """normalize argument converts all values to range 0.0-1.0"""
    env_state = MathyEnvState(problem="4+2")
    obs: MathyObservation = env_state.to_observation(normalize=False)
    assert np.max(obs.values) == 4.0

    norm: MathyObservation = env_state.to_observation(normalize=True)
    assert np.max(norm.values) == 1.0


def test_state_to_observation_normalized_problem_type():
    """normalize argument converts all values and type hash to range 0.0-1.0"""
    env_state = MathyEnvState(problem="4+2")
    obs: MathyObservation = env_state.to_observation()
    print(obs.type)
    assert np.max(obs.time) <= 1.0
    assert np.min(obs.time) >= 0.0

    assert np.max(obs.values) <= 1.0
    assert np.min(obs.values) >= 0.0

    assert np.max(obs.type) <= 1.0
    assert np.min(obs.type) >= 0.0


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
