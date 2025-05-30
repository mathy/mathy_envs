from mathy_envs import MathyEnv, MathyEnvState, envs
from mathy_envs.state import MathyGraphObservation, ObservationType

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]
observation = env.state_to_observation(
    state, obs_type=ObservationType.GRAPH, max_seq_len=100
)
# Narrow down to the specific observation type
assert isinstance(observation, MathyGraphObservation)

# Check actual number of nodes
assert observation.num_nodes > 0
assert observation.num_nodes <= 100  # should be within max_seq_len

# Check the observation structure
assert observation.adjacency is not None, "Adjacency matrix should not be None"
assert observation.node_features is not None, "Node features should not be None"
assert observation.action_mask is not None, "Action mask should not be None"
