from mathy_envs import MathyEnv, MathyEnvState, envs
from mathy_envs.state import MathyHierarchicalObservation, ObservationType

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]

hier_obs = state.to_observation(obs_type=ObservationType.HIERARCHICAL, max_seq_len=100)
assert isinstance(hier_obs, MathyHierarchicalObservation)

print(f"Node features shape: {hier_obs.node_features.shape}")  # (100, 4)
print(f"Level indices shape: {hier_obs.level_indices.shape}")  # (100,)
print(f"Max depth: {hier_obs.max_depth}")
print(f"Actual nodes: {hier_obs.num_nodes}")

# Show node distribution by level
actual_levels = hier_obs.level_indices[: hier_obs.num_nodes]
for level in range(hier_obs.max_depth + 1):
    count = sum(1 for l in actual_levels if l == level)
    print(f"Level {level}: {count} nodes")
