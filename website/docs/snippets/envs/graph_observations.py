# graph_observations.py
from mathy_envs import MathyEnv, MathyEnvState, envs
from mathy_envs.state import MathyGraphObservation, ObservationType

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]

graph_obs = state.to_observation(obs_type=ObservationType.GRAPH, max_seq_len=100)
assert isinstance(graph_obs, MathyGraphObservation)

print(f"Node features shape: {graph_obs.node_features.shape}")  # (100, 4)
print(f"Adjacency shape: {graph_obs.adjacency.shape}")         # (100, 100)
print(f"Action mask length: {len(graph_obs.action_mask)}")     # num_rules * 100
print(f"Actual nodes: {graph_obs.num_nodes}")

# Check adjacency connections
actual_adj = graph_obs.adjacency[:graph_obs.num_nodes, :graph_obs.num_nodes]
print(f"Graph has {actual_adj.sum()} connections")
