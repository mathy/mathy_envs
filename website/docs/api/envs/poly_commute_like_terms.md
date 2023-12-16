```python

import mathy_envs.envs.poly_commute_like_terms
```

## PolyCommuteLikeTerms
```python
PolyCommuteLikeTerms(self, ops: Optional[List[str]] = None, kwargs: Any)
```
A Mathy environment for moving like terms near each other to enable
further simplification.

This task is intended to test the model's ability to identify like terms
in a large string of unlike terms and its ability to use the commutative
swap rule to reorder the expression bringing the like terms close together.

### max_moves_fn
```python
PolyCommuteLikeTerms.max_moves_fn(
    self, 
    problem: mathy_envs.types.MathyEnvProblem, 
    config: mathy_envs.types.MathyEnvProblemArgs, 
) -> int
```
This task is to move two terms near each other, which requires
as many actions as there are blocker nodes. The problem complexity
is a direct measure of this value.
### rule
Distributive Property
`ab + ac = a(b + c)`

 The distributive property can be used to expand out expressions
 to allow for simplification, as well as to factor out common properties
 of terms.

 **Factor out a common term**

 This handles the `ab + ac` conversion of the distributive property, which
 factors out a common term from the given two addition operands.

           +               *
          / \             / \
         /   \           /   \
        /     \    ->   /     \
       *       *       a       +
      / \     / \             / \
     a   b   a   c           b   c

### transition_fn
```python
PolyCommuteLikeTerms.transition_fn(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    expression: mathy_core.expressions.MathExpression, 
    features: mathy_envs.state.MathyObservation, 
) -> Optional[mathy_envs.time_step.TimeStep]
```
If the expression has any nodes that the DistributiveFactorOut rule
can be applied to, the problem is solved.