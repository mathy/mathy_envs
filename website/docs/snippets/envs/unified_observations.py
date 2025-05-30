from mathy_envs import MathyEnv, MathyEnvState, envs
from mathy_envs.state import ObservationType

env: MathyEnv = envs.PolySimplify()
state: MathyEnvState = env.get_initial_state()[0]

# All observation types through unified interface
obs_types = [
    ObservationType.FLAT,
    ObservationType.GRAPH,
    ObservationType.HIERARCHICAL,
    ObservationType.MESSAGE_PASSING,
]

for obs_type in obs_types:
    obs = state.to_observation(obs_type=obs_type, max_seq_len=100, normalize=True)
    print(f"{obs_type.value}: {type(obs).__name__}")
