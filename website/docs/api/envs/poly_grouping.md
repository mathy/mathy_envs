```python

import mathy_envs.envs.poly_grouping
```

## PolyGroupLikeTerms
```python
PolyGroupLikeTerms(
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
A Mathy environment for grouping polynomial terms that are like.

The goal is to commute all the like terms so they become siblings as quickly as
possible.

### transition_fn
```python
PolyGroupLikeTerms.transition_fn(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    expression: mathy_core.expressions.MathExpression, 
    features: mathy_envs.state.MathyObservation, 
) -> Optional[mathy_envs.time_step.TimeStep]
```
If all like terms are siblings.