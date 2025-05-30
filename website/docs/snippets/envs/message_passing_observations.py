from mathy_envs import MathyEnv, MathyEnvState, envs
from mathy_envs.state import MathyMessagePassingObservation, ObservationType

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]

mp_obs = state.to_observation(obs_type=ObservationType.MESSAGE_PASSING, max_seq_len=100)
assert isinstance(mp_obs, MathyMessagePassingObservation)

print(f"Node features shape: {mp_obs.node_features.shape}")  # (100, 4)
print(f"Edge index shape: {mp_obs.edge_index.shape}")  # (2, 200)
print(f"Edge types shape: {mp_obs.edge_types.shape}")  # (200,)
print(f"Actual nodes: {mp_obs.num_nodes}")
print(f"Actual edges: {mp_obs.num_edges}")

# Show edge type distribution
actual_edge_types = mp_obs.edge_types[: mp_obs.num_edges]
left_edges = sum(1 for t in actual_edge_types if t == 0)
right_edges = sum(1 for t in actual_edge_types if t == 1)
print(f"Left child edges: {left_edges}, Right child edges: {right_edges}")
