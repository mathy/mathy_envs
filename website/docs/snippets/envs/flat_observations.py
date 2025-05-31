from mathy_envs import MathyEnv, MathyEnvState, envs
from mathy_envs.state import ObservationType

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]

# Flat observation (default)
flat_obs = state.to_observation(obs_type=ObservationType.FLAT, max_seq_len=100)
print(f"Flat observation shape: {flat_obs.shape}")
# Contains: [type_hash, time, nodes..., values..., action_mask...]
