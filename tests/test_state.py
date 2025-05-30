from typing import List, Tuple

import numpy as np
import pytest

from mathy_envs.envs.poly_simplify import PolySimplify
from mathy_envs.state import (
    MathyEnvState,
    MathyGraphObservation,
    MathyHierarchicalObservation,
    MathyMessagePassingObservation,
    ObservationType,
)


def test_state_to_observation():
    """to_observation has defaults to allow calling with no arguments"""
    env_state = MathyEnvState(problem="4x+2")
    assert env_state.to_observation() is not None


def test_state_unified_observation_interface():
    """Test the unified to_observation method with different observation types"""
    env_state = MathyEnvState(problem="4x+2*3")

    # Test flat observation (default) - now returns np.ndarray
    flat_obs = env_state.to_observation()
    assert isinstance(flat_obs, np.ndarray)

    # Test explicit flat observation
    flat_obs_explicit = env_state.to_observation(obs_type=ObservationType.FLAT)
    assert isinstance(flat_obs_explicit, np.ndarray)
    assert np.array_equal(flat_obs, flat_obs_explicit)

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
    env_state = MathyEnvState(
        problem="4x+2*3", num_rules=7
    )  # Need num_rules for action mask
    obs = env_state.to_observation(obs_type=ObservationType.GRAPH, max_seq_len=10)

    # Narrow type with assertion
    assert isinstance(obs, MathyGraphObservation)

    # Check basic properties
    assert hasattr(obs, "node_features")
    assert hasattr(obs, "adjacency")
    assert hasattr(obs, "action_mask")
    assert hasattr(obs, "num_nodes")

    # Check shapes make sense - should be padded to max_seq_len
    assert obs.node_features.shape == (
        10,
        4,
    )  # padded to max_seq_len, 4 features: [type_id, value, time, is_leaf]
    assert obs.adjacency.shape == (10, 10)  # padded adjacency matrix
    assert len(obs.action_mask) == 7 * 10  # num_rules * max_seq_len

    # Check actual number of nodes
    assert obs.num_nodes > 0
    assert obs.num_nodes <= 10  # should be within max_seq_len

    # Check feature dimensions - now 4D: [type_id, value, time, is_leaf]
    assert obs.node_features.shape[1] == 4

    # Check adjacency matrix properties
    assert obs.adjacency.dtype == np.float32
    assert np.all((obs.adjacency == 0) | (obs.adjacency == 1))  # Binary matrix

    # For "4x+2*3", we should have a tree structure
    # Root (+) should connect to left (4x) and right (2*3)
    actual_adjacency = obs.adjacency[: obs.num_nodes, : obs.num_nodes]
    assert np.sum(actual_adjacency) > 0  # Should have some connections

    # Test feature content - check that time and is_leaf features are present
    actual_features = obs.node_features[: obs.num_nodes]
    if obs.num_nodes > 0:
        # Time feature should be consistent across all nodes (same timestep)
        time_values = actual_features[:, 2]  # time is 3rd feature (index 2)
        assert np.all(time_values == time_values[0])  # all nodes should have same time

        # is_leaf feature should be binary
        is_leaf_values = actual_features[:, 3]  # is_leaf is 4th feature (index 3)
        assert np.all((is_leaf_values == 0.0) | (is_leaf_values == 1.0))

    # Test to_dict method
    obs_dict = obs.to_dict()
    assert "node_features" in obs_dict
    assert "adjacency" in obs_dict
    assert "action_mask" in obs_dict
    assert "num_nodes" in obs_dict


def test_state_hierarchical_observation_structure():
    """Test hierarchical observation groups nodes by depth correctly"""
    env_state = MathyEnvState(problem="(4+2)*3", num_rules=7)  # 3 levels: *, +, leaves
    obs = env_state.to_observation(
        obs_type=ObservationType.HIERARCHICAL, max_seq_len=10
    )

    # Check basic structure
    assert hasattr(obs, "node_features")
    assert hasattr(obs, "level_indices")
    assert hasattr(obs, "action_mask")
    assert hasattr(obs, "max_depth")
    assert hasattr(obs, "num_nodes")

    # Narrow type with assertion
    assert isinstance(obs, MathyHierarchicalObservation)

    # Check shapes are padded - now 4D features: [type_id, value, time, is_leaf]
    assert obs.node_features.shape == (10, 4)  # padded to max_seq_len, 4 features
    assert obs.level_indices.shape == (10,)  # padded to max_seq_len
    assert len(obs.action_mask) == 7 * 10  # num_rules * max_seq_len

    # Check depth properties
    assert obs.max_depth >= 0
    assert obs.num_nodes > 0

    # Non-zero nodes should have valid level indices
    if obs.num_nodes > 0:
        actual_levels = obs.level_indices[: obs.num_nodes]
        assert np.all(actual_levels >= 0)
        assert np.all(actual_levels <= obs.max_depth)

    # Test feature content - check that all 4 features are present
    actual_features = obs.node_features[: obs.num_nodes]
    if obs.num_nodes > 0:
        # Time feature should be consistent across all nodes
        time_values = actual_features[:, 2]  # time is 3rd feature (index 2)
        assert np.all(time_values == time_values[0])  # all nodes should have same time

        # is_leaf feature should be binary
        is_leaf_values = actual_features[:, 3]  # is_leaf is 4th feature (index 3)
        assert np.all((is_leaf_values == 0.0) | (is_leaf_values == 1.0))

    # Test to_dict method
    obs_dict = obs.to_dict()
    assert "node_features" in obs_dict
    assert "level_indices" in obs_dict
    assert "action_mask" in obs_dict
    assert "max_depth" in obs_dict
    assert "num_nodes" in obs_dict


def test_state_message_passing_observation_structure():
    """Test message passing observation edge format"""
    env_state = MathyEnvState(problem="4x+2", num_rules=7)
    obs = env_state.to_observation(
        obs_type=ObservationType.MESSAGE_PASSING, max_seq_len=10
    )

    # Narrow type with assertion
    assert isinstance(obs, MathyMessagePassingObservation)

    # Check basic properties
    assert hasattr(obs, "node_features")
    assert hasattr(obs, "edge_index")
    assert hasattr(obs, "edge_types")
    assert hasattr(obs, "action_mask")
    assert hasattr(obs, "num_nodes")
    assert hasattr(obs, "num_edges")

    # Check dimensions are padded - now 4D features: [type_id, value, time, is_leaf]
    assert obs.node_features.shape[0] == 10  # padded to max_seq_len
    assert (
        obs.node_features.shape[1] == 4
    )  # 4 features: [type_id, value, time, is_leaf]
    assert obs.edge_index.shape == (2, 20)  # (2, max_seq_len * 2)
    assert obs.edge_types.shape == (20,)  # max_seq_len * 2
    assert len(obs.action_mask) == 7 * 10  # num_rules * max_seq_len

    # Check actual counts
    assert obs.num_nodes > 0
    assert obs.num_nodes <= 10
    assert obs.num_edges >= 0

    # Check edge format (PyTorch Geometric style)
    if obs.num_edges > 0:
        actual_edges = obs.edge_index[:, : obs.num_edges]
        actual_edge_types = obs.edge_types[: obs.num_edges]

        # All edge indices should be valid node indices
        assert np.all(actual_edges >= 0)
        assert np.all(actual_edges < obs.num_nodes)

        # Edge types should be 0 (left) or 1 (right)
        assert np.all((actual_edge_types == 0) | (actual_edge_types == 1))

    # Test feature content
    actual_features = obs.node_features[: obs.num_nodes]
    if obs.num_nodes > 0:
        # Time feature should be consistent across all nodes
        time_values = actual_features[:, 2]  # time is 3rd feature (index 2)
        assert np.all(time_values == time_values[0])  # all nodes should have same time

        # is_leaf feature should be binary
        is_leaf_values = actual_features[:, 3]  # is_leaf is 4th feature (index 3)
        assert np.all((is_leaf_values == 0.0) | (is_leaf_values == 1.0))

    # Test to_dict method
    obs_dict = obs.to_dict()
    assert "node_features" in obs_dict
    assert "edge_index" in obs_dict
    assert "edge_types" in obs_dict
    assert "action_mask" in obs_dict
    assert "num_nodes" in obs_dict
    assert "num_edges" in obs_dict


def test_consistent_feature_dimensions():
    """Test that all observation types now use consistent 4D features"""
    env_state = MathyEnvState(problem="2*x+3", num_rules=7)
    max_seq_len = 10

    # Test all observation types have 4D features
    graph_obs = env_state.to_observation(
        obs_type=ObservationType.GRAPH, max_seq_len=max_seq_len
    )
    hier_obs = env_state.to_observation(
        obs_type=ObservationType.HIERARCHICAL, max_seq_len=max_seq_len
    )
    mp_obs = env_state.to_observation(
        obs_type=ObservationType.MESSAGE_PASSING, max_seq_len=max_seq_len
    )

    # Narrow type with assertions
    assert isinstance(graph_obs, MathyGraphObservation)
    assert isinstance(hier_obs, MathyHierarchicalObservation)
    assert isinstance(mp_obs, MathyMessagePassingObservation)

    # All should have 4D features: [type_id, value, time, is_leaf]
    assert graph_obs.node_features.shape[1] == 4
    assert hier_obs.node_features.shape[1] == 4
    assert mp_obs.node_features.shape[1] == 4

    # Check that features are consistent across observation types
    if graph_obs.num_nodes > 0 and hier_obs.num_nodes > 0 and mp_obs.num_nodes > 0:
        # All should have same number of nodes (same underlying expression)
        assert graph_obs.num_nodes == hier_obs.num_nodes == mp_obs.num_nodes

        # Time feature should be the same across all observation types
        graph_time = graph_obs.node_features[0, 2]  # time from first node
        hier_time = hier_obs.node_features[0, 2]
        mp_time = mp_obs.node_features[0, 2]

        assert graph_time == hier_time == mp_time


def test_feature_content_validation():
    """Test that the 4D features contain expected content"""
    env_state = MathyEnvState(problem="5+x", num_rules=7)
    max_seq_len = 10

    obs = env_state.to_observation(
        obs_type=ObservationType.GRAPH, max_seq_len=max_seq_len, normalize=True
    )

    assert isinstance(obs, MathyGraphObservation)

    if obs.num_nodes > 0:
        actual_features = obs.node_features[: obs.num_nodes]

        # Feature 0: type_id - should be normalized (0-1 range)
        type_ids = actual_features[:, 0]
        assert np.all(type_ids >= 0.0) and np.all(type_ids <= 1.0)

        # Feature 1: value - should be normalized (0-1 range)
        values = actual_features[:, 1]
        assert np.all(values >= 0.0) and np.all(values <= 1.0)

        # Feature 2: time - should be 0-1 range (episode progress)
        time_vals = actual_features[:, 2]
        assert np.all(time_vals >= 0.0) and np.all(time_vals <= 1.0)
        # Time should be the same for all nodes in a single observation
        assert np.all(time_vals == time_vals[0])

        # Feature 3: is_leaf - should be binary (0.0 or 1.0)
        is_leaf_vals = actual_features[:, 3]
        assert np.all((is_leaf_vals == 0.0) | (is_leaf_vals == 1.0))
        # For "5+x", we should have both leaf nodes (5, x) and non-leaf (+)
        assert np.any(is_leaf_vals == 1.0)  # Should have at least one leaf


def test_state_observation_consistency():
    """Test that different observation types represent the same underlying problem"""
    env_state = MathyEnvState(problem="2*x+3", num_rules=7)
    max_seq_len = 10

    env_state.to_observation(obs_type=ObservationType.FLAT, max_seq_len=max_seq_len)
    graph_obs = env_state.to_observation(
        obs_type=ObservationType.GRAPH, max_seq_len=max_seq_len
    )
    hier_obs = env_state.to_observation(
        obs_type=ObservationType.HIERARCHICAL, max_seq_len=max_seq_len
    )
    mp_obs = env_state.to_observation(
        obs_type=ObservationType.MESSAGE_PASSING, max_seq_len=max_seq_len
    )

    # Narrow type with assertion
    assert isinstance(graph_obs, MathyGraphObservation)
    assert isinstance(hier_obs, MathyHierarchicalObservation)
    assert isinstance(mp_obs, MathyMessagePassingObservation)

    # All should represent the same number of actual nodes
    # For flat observation, we need to determine actual nodes from the array structure
    # The array structure is: [type, time, nodes..., values..., action_mask...]
    # We'll use the structured observations for comparison
    n_nodes_graph = graph_obs.num_nodes
    n_nodes_hier = hier_obs.num_nodes
    n_nodes_mp = mp_obs.num_nodes

    assert n_nodes_graph == n_nodes_hier == n_nodes_mp

    # All should have same action mask size
    graph_len = len(graph_obs.action_mask)
    hier_len = len(hier_obs.action_mask)
    mp_len = len(mp_obs.action_mask)
    assert graph_len == hier_len == mp_len == 7 * max_seq_len


def test_state_complex_expression_observations():
    """Test observations work correctly for complex mathematical expressions"""
    env_state = MathyEnvState(problem="(a+b)*(c-d)/e", num_rules=7)
    max_seq_len = 20

    # Test each observation type handles complexity
    flat_obs = env_state.to_observation(
        obs_type=ObservationType.FLAT, max_seq_len=max_seq_len
    )
    graph_obs = env_state.to_observation(
        obs_type=ObservationType.GRAPH, max_seq_len=max_seq_len
    )
    hier_obs = env_state.to_observation(
        obs_type=ObservationType.HIERARCHICAL, max_seq_len=max_seq_len
    )
    mp_obs = env_state.to_observation(
        obs_type=ObservationType.MESSAGE_PASSING, max_seq_len=max_seq_len
    )

    # Narrow type with assertion
    assert isinstance(flat_obs, np.ndarray)
    assert isinstance(graph_obs, MathyGraphObservation)
    assert isinstance(hier_obs, MathyHierarchicalObservation)
    assert isinstance(mp_obs, MathyMessagePassingObservation)

    # Should have multiple nodes for this complex expression
    assert isinstance(flat_obs, np.ndarray)
    assert graph_obs.num_nodes > 5
    assert hier_obs.num_nodes > 5
    assert mp_obs.num_nodes > 5

    # Hierarchical should have multiple levels
    assert hier_obs.max_depth > 2

    # Graph should have connections
    actual_adjacency = graph_obs.adjacency[: graph_obs.num_nodes, : graph_obs.num_nodes]
    assert np.sum(actual_adjacency) > 0

    # Message passing should have edges
    assert mp_obs.num_edges > 0


def test_state_observation_normalization():
    """Test normalization works across observation types"""
    env_state = MathyEnvState(problem="100+200", num_rules=7)
    max_seq_len = 10

    # Test with normalization (default)
    norm_graph = env_state.to_observation(
        obs_type=ObservationType.GRAPH, normalize=True, max_seq_len=max_seq_len
    )

    # Narrow type with assertion
    assert isinstance(norm_graph, MathyGraphObservation)

    # Graph node features should be normalized
    actual_features = norm_graph.node_features[: norm_graph.num_nodes]
    if norm_graph.num_nodes > 0:
        assert np.max(actual_features) <= 1.0
        assert np.min(actual_features) >= 0.0

    # Test without normalization
    unnorm_graph = env_state.to_observation(
        obs_type=ObservationType.GRAPH, normalize=False, max_seq_len=max_seq_len
    )
    assert isinstance(unnorm_graph, MathyGraphObservation)

    # Should have larger raw values for constants
    actual_unnorm_features = unnorm_graph.node_features[: unnorm_graph.num_nodes]
    if unnorm_graph.num_nodes > 0:
        # At least one feature should be > 1.0 for the large constants
        assert np.max(actual_unnorm_features) > 1.0


def test_state_edge_case_observations():
    """Test observations work for edge cases"""
    max_seq_len = 10

    # Single node
    single_state = MathyEnvState(problem="5", num_rules=7)
    single_obs = single_state.to_observation(
        obs_type=ObservationType.GRAPH, max_seq_len=max_seq_len
    )
    # Narrow type with assertion
    assert isinstance(single_obs, MathyGraphObservation)
    assert single_obs.num_nodes == 1
    # No edges for single node
    actual_adjacency = single_obs.adjacency[
        : single_obs.num_nodes, : single_obs.num_nodes
    ]
    assert np.sum(actual_adjacency) == 0

    # Variable only
    var_state = MathyEnvState(problem="x", num_rules=7)
    var_obs = var_state.to_observation(
        obs_type=ObservationType.MESSAGE_PASSING, max_seq_len=max_seq_len
    )
    # Narrow type with assertion
    assert isinstance(var_obs, MathyMessagePassingObservation)
    assert var_obs.num_nodes == 1
    assert var_obs.num_edges == 0  # No edges

    # Test hierarchical with single level
    hier_obs = single_state.to_observation(
        obs_type=ObservationType.HIERARCHICAL, max_seq_len=max_seq_len
    )
    # Narrow type with assertion
    assert isinstance(hier_obs, MathyHierarchicalObservation)
    assert hier_obs.max_depth == 0
    assert hier_obs.num_nodes == 1


def test_state_invalid_observation_type():
    """Test that invalid observation types raise appropriate errors"""
    env_state = MathyEnvState(problem="4x+2")

    with pytest.raises(ValueError, match="Unknown observation type"):
        # This should raise an error since we're bypassing the enum
        env_state.to_observation(obs_type="invalid_type")  # type:ignore[call-arg]


def test_state_to_observation_normalization():
    """normalize argument converts all values to range 0.0-1.0"""
    env_state = MathyEnvState(problem="4+2", num_rules=7)
    max_seq_len = 10

    # Test with graph observation since flat is now different
    obs = env_state.to_observation(
        obs_type=ObservationType.GRAPH, normalize=False, max_seq_len=max_seq_len
    )
    # Narrow type with assertion
    assert isinstance(obs, MathyGraphObservation)
    actual_features = obs.node_features[: obs.num_nodes]
    if obs.num_nodes > 0:
        assert (
            np.max(actual_features[:, 1]) == 4.0
        )  # value column should have max of 4.0

    norm = env_state.to_observation(
        obs_type=ObservationType.GRAPH, normalize=True, max_seq_len=max_seq_len
    )
    # Narrow type with assertion
    assert isinstance(norm, MathyGraphObservation)
    actual_norm_features = norm.node_features[: norm.num_nodes]
    if norm.num_nodes > 0:
        assert np.max(actual_norm_features[:, 1]) == 1.0  # normalized value column


def test_state_to_observation_normalized_problem_type():
    """Test that flat observations contain normalized components"""
    env_state = MathyEnvState(problem="4+2", num_rules=7)
    obs = env_state.to_observation(max_seq_len=10)

    # Flat observation is now a numpy array
    assert isinstance(obs, np.ndarray)

    # Array structure: [type_hash, time, nodes..., values..., action_mask...]
    # First element should be normalized type hash (0-1 range)
    # Second element should be normalized time (0-1 range)
    assert 0.0 <= obs[0] <= 1.0  # type hash
    assert 0.0 <= obs[1] <= 1.0  # time

    # The rest should also be in normalized ranges (nodes and values are normalized)
    # Only action mask at the end might be 0/1 binary


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
        state_one = MathyEnvState(problem=one, num_rules=len(env.rules))
        obs_one = state_one.to_observation(
            move_mask=env.get_valid_moves(state_one), max_seq_len=env.max_seq_len
        )
        # Narrow type with assertion
        assert isinstance(obs_one, np.ndarray)

        state_two = MathyEnvState(problem=two, num_rules=len(env.rules))
        obs_two = state_two.to_observation(
            move_mask=env.get_valid_moves(state_two), max_seq_len=env.max_seq_len
        )
        # Narrow type with assertion
        assert isinstance(obs_two, np.ndarray)

        # Flat observations are now numpy arrays, so they should be different
        assert not np.array_equal(obs_one, obs_two)


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


def test_action_mask_processing():
    """Test that action masks are properly processed and padded"""
    env_state = MathyEnvState(problem="x+2", num_rules=7)

    # Create a mock raw mask (typical format from get_valid_moves)
    raw_mask = [
        [1, 0, 1],  # rule 0: valid at nodes 0 and 2
        [0, 1, 0],  # rule 1: valid at node 1
        [1, 1, 1],  # rule 2: valid at all nodes
        [0, 0, 0],  # rules 3-6: no valid moves
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    max_seq_len = 10
    obs = env_state.to_observation(
        move_mask=raw_mask, obs_type=ObservationType.GRAPH, max_seq_len=max_seq_len
    )
    # Narrow type with assertion
    assert isinstance(obs, MathyGraphObservation)

    # Action mask should be flattened and padded
    expected_size = 7 * 10  # num_rules * max_seq_len
    assert len(obs.action_mask) == expected_size

    # First few elements should match the raw mask pattern
    # Rule 0: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0] (padded to max_seq_len)
    # Rule 1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # etc.
    assert obs.action_mask[0] == 1.0  # rule 0, node 0
    assert obs.action_mask[1] == 0.0  # rule 0, node 1
    assert obs.action_mask[2] == 1.0  # rule 0, node 2
    assert obs.action_mask[10] == 0.0  # rule 1, node 0
    assert obs.action_mask[11] == 1.0  # rule 1, node 1
