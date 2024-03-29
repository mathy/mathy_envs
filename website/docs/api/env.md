```python

import mathy_envs.env
```

## MathyEnv
```python
MathyEnv(
    self, 
    rules: Optional[List[mathy_core.rule.BaseRule]] = None, 
    max_moves: int = 20, 
    verbose: bool = False, 
    invalid_action_response: Literal['raise', 'penalize', 'terminal'] = 'raise', 
    reward_discount: float = 0.99, 
    max_seq_len: int = 128, 
    previous_state_penalty: bool = True, 
    preferred_term_commute: bool = False, 
)
```
Implement a math solving game where a player wins by executing the
right sequence of actions to reduce a math expression to an agreeable
basic representation in as few moves as possible.
### action_size
Return the number of available actions
### core_rules
```python
MathyEnv.core_rules(
    preferred_term_commute: bool = False, 
) -> List[mathy_core.rule.BaseRule]
```
Return the mathy core agent actions
### finalize_state
```python
MathyEnv.finalize_state(self, state: mathy_envs.state.MathyEnvState) -> None
```
Perform final checks on a problem state, to ensure the episode yielded
results that were uncorrupted by transformation errors.
### get_actions_for_node
```python
MathyEnv.get_actions_for_node(
    self, 
    expression: mathy_core.expressions.MathExpression, 
    rule_list: Optional[List[Type[mathy_core.rule.BaseRule]]] = None, 
) -> List[List[int]]
```
Return a valid actions mask for the given expression and rule list.

Action masks are 2d lists of length (num_rules, max_seq_len) where a 0 indicates
the action is not valid in the current state, and a 1 indicates that it is
a valid action to take.
### get_agent_actions_count
```python
MathyEnv.get_agent_actions_count(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
) -> int
```
Return number of all possible actions
### get_env_namespace
```python
MathyEnv.get_env_namespace(self) -> str
```
Return a unique dot namespaced string representing the current
environment. e.g. mycompany.envs.differentiate
### get_initial_state
```python
MathyEnv.get_initial_state(
    self, 
    params: Optional[mathy_envs.types.MathyEnvProblemArgs] = None, 
    print_problem: bool = True, 
) -> Tuple[mathy_envs.state.MathyEnvState, mathy_envs.types.MathyEnvProblem]
```
Generate an initial MathyEnvState for an episode
### get_lose_signal
```python
MathyEnv.get_lose_signal(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
) -> float
```
Calculate the reward value for failing to complete the episode. This is done
so that the reward signal can be problem-type dependent.
### get_next_state
```python
MathyEnv.get_next_state(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    action: Union[int, numpy.int64, Tuple[int, int]], 
) -> Tuple[mathy_envs.state.MathyEnvState, mathy_envs.time_step.TimeStep, mathy_core.rule.ExpressionChangeRule]
```

__Parameters__

- __env_state__: current env_state
- __action__:    a tuple of two integers representing the rule and node to act on

__Returns__

`next_state`: env_state after applying action

`transition`: the timestep that represents the state transition

`change`: the change descriptor describing the change that happened

### get_penalizing_actions
```python
MathyEnv.get_penalizing_actions(
    self, 
    state: mathy_envs.state.MathyEnvState, 
) -> List[Type[mathy_core.rule.BaseRule]]
```
Get the list of penalizing action types. When these actions
are selected, the agent gets a negative reward.
### get_rewarding_actions
```python
MathyEnv.get_rewarding_actions(
    self, 
    state: mathy_envs.state.MathyEnvState, 
) -> List[Type[mathy_core.rule.BaseRule]]
```
Get the list of rewarding action types. When these actions
are selected, the agent gets a positive reward.
### get_state_transition
```python
MathyEnv.get_state_transition(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
) -> mathy_envs.time_step.TimeStep
```
Given an input state calculate the transition value of the timestep.

__Parameters__

- __env_state__: current env_state

__Returns__

`transition`: the current state value transition

### get_token_at_index
```python
MathyEnv.get_token_at_index(
    self, 
    expression: mathy_core.expressions.MathExpression, 
    index: int, 
) -> Optional[mathy_core.expressions.MathExpression]
```
Get the token that is `index` from the left of the expression
### get_valid_moves
```python
MathyEnv.get_valid_moves(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
) -> List[List[int]]
```
Get a 2d list describing the valid moves for the current state.

The first dimension contains the list of known rules in the order that
they're registered, and the second dimension contains a list of the max
sequence length size that is 1/0 representing that the node at that index
for the given rule is valid.

### get_valid_rules
```python
MathyEnv.get_valid_rules(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
) -> List[int]
```
Get a vector the length of the number of valid rules that is
filled with 0/1 based on whether the rule has any nodes in the
expression that it can be applied to.

!!! note

    If you want to get a list of which nodes each rule can be
    applied to, prefer to use the `get_valid_moves` method.

### get_win_signal
```python
MathyEnv.get_win_signal(self, env_state: mathy_envs.state.MathyEnvState) -> float
```
Calculate the reward value for completing the episode. This is done
so that the reward signal can be scaled based on the time it took to
complete the episode.
### is_terminal_state
```python
MathyEnv.is_terminal_state(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
) -> bool
```
Determine if a given state is terminal or not.

__Arguments__

- __env_state (MathyEnvState)__: The state to inspect

__Returns__

`(bool)`: A boolean indicating if the state is terminal or not.

### max_moves_fn
```python
MathyEnv.max_moves_fn(
    self, 
    problem: mathy_envs.types.MathyEnvProblem, 
    config: mathy_envs.types.MathyEnvProblemArgs, 
) -> int
```
Return the environment specific maximum move count for a given prolem.
### print_history
```python
MathyEnv.print_history(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    pretty: bool = True, 
) -> None
```
Render the history of an episode from a given state.

__Arguments__

- __env_state (MathyEnvState)__: The state to render the history of.

### print_state
```python
MathyEnv.print_state(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    action_name: str, 
    token_index: int = -1, 
    change: Optional[mathy_core.rule.ExpressionChangeRule] = None, 
    change_reward: float = 0.0, 
    pretty: bool = False, 
) -> None
```
Render the given state to stdout for visualization
### problem_fn
```python
MathyEnv.problem_fn(
    self, 
    params: mathy_envs.types.MathyEnvProblemArgs, 
) -> mathy_envs.types.MathyEnvProblem
```
Return a problem for the environment given a set of parameters
to control problem generation.

This is implemented per environment so each environment can
generate its own dataset with no required configuration.
### random_action
```python
MathyEnv.random_action(
    self, 
    expression: mathy_core.expressions.MathExpression, 
    rule: Optional[Type[mathy_core.rule.BaseRule]] = None, 
) -> Tuple[int, int]
```
Get a random action index that represents a particular rule
### render_state
```python
MathyEnv.render_state(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    action_name: str, 
    token_index: int = -1, 
    change: Optional[mathy_core.rule.ExpressionChangeRule] = None, 
    change_reward: float = 0.0, 
    pretty: bool = False, 
) -> str
```
Render the given state to a string suitable for printing to a log
### state_to_observation
```python
MathyEnv.state_to_observation(
    self, 
    state: mathy_envs.state.MathyEnvState, 
    max_seq_len: Optional[int] = None, 
) -> mathy_envs.state.MathyObservation
```
Convert an environment state into an observation that can be used
by a training agent.
### to_action
```python
MathyEnv.to_action(
    self, 
    action: Union[int, numpy.int64, Tuple[int, int]], 
) -> Tuple[int, int]
```
Resolve a given action input to a tuple of (rule_index, node_index).

When given an int, it is treated as an index into the flattened 2d action
space. When given a tuple, it is assumed to be (rule, node)
### to_hash_key
```python
MathyEnv.to_hash_key(self, env_state: mathy_envs.state.MathyEnvState) -> str
```
Convert env_state to a string for MCTS cache
### transition_fn
```python
MathyEnv.transition_fn(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    expression: mathy_core.expressions.MathExpression, 
    features: mathy_envs.state.MathyObservation, 
) -> Optional[mathy_envs.time_step.TimeStep]
```
Provide environment-specific transitions per timestep.