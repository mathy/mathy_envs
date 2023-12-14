# mathy_envs.envs.poly_haystack_like_terms

## PolyHaystackLikeTerms
```python
PolyHaystackLikeTerms(self, kwargs: Any)
```
Act on any node in the expression that has another term like it
somewhere else. For example in the problem:

2x + 8 + 13.2y + z^2 + 5x
^^---------------------^^

Applying any rule to one of those nodes is a win. The idea here is that
in order to succeed at this task, the model must build a representation
that can identify like terms in a large expression tree.

### transition_fn
```python
PolyHaystackLikeTerms.transition_fn(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    expression: mathy_core.expressions.MathExpression, 
    features: mathy_envs.state.MathyObservation, 
) -> Optional[mathy_envs.time_step.TimeStep]
```
If all like terms are siblings.
