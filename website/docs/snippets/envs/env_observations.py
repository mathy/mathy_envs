from mathy_envs import MathyEnv, envs
from mathy_envs.state import ObservationType

# Initialize environment
env: MathyEnv = envs.PolySimplify()

# Get initial state and valid moves
state, problem = env.get_initial_state()
valid_moves = env.get_valid_moves(state)

# Create observation with action mask
obs = state.to_observation(
    move_mask=valid_moves,
    obs_type=ObservationType.GRAPH,
    max_seq_len=env.max_seq_len,
    normalize=True,
)

print(f"Problem: {problem}")
print(f"Observation type: {type(obs).__name__}")
print(f"Valid actions available: {obs.action_mask.sum()}")
