from mathy_envs import MathyEnv, MathyEnvState, envs
from mathy_envs.state import MathyGraphObservation, ObservationType

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]

# Create graph observation with consistent features
observation = state.to_observation(
    obs_type=ObservationType.GRAPH, max_seq_len=100, normalize=True
)

# Narrow down to the specific observation type
assert isinstance(observation, MathyGraphObservation)

# Check the consistent feature format: [type_id, value, time, is_leaf]
print(f"Node features shape: {observation.node_features.shape}")  # (100, 4)
print(f"Feature dimensions: type_id, value, time, is_leaf")

# Check actual number of nodes
assert observation.num_nodes > 0
assert observation.num_nodes <= 100  # should be within max_seq_len

# Check the observation structure
assert observation.adjacency is not None, "Adjacency matrix should not be None"
assert observation.node_features is not None, "Node features should not be None"
assert observation.action_mask is not None, "Action mask should not be None"

# Show actual node features for debugging
actual_features = observation.node_features[: observation.num_nodes]
print(
    f"First node features: {actual_features[0] if len(actual_features) > 0 else 'No nodes'}"
)
