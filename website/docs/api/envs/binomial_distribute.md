A set of binomials is multiplied and must be simplified to satisfy the win conditions.

## Challenge

In binomial multiplication, the agent must learn to distribute the binomial multiplications quickly and factor out the common terms to leave a simplified representation.

Examples

- `(4 + g^2)(9 + e^3)` must be simplified to `36 + (4e^3 + (9g^2 + g^2 * e^3))`
- `(a + a) * a` must be simplified to `2a^2`
- `(c + 5) * c` must be simplified to `c^2 + 5c`
- `(i^3 + 2)(i^3 + 9)` must be simplified to `i^6 + (11i^3 + 18)`
- `(3 + 12o)(10 + 8o)` must be simplified to `30 + (144o + 96o^2)`

## Win Conditions

A problem is considered solved when no remaining complex terms exist in the expression.

### No Complex Terms

Terms are considered complex when there's a more concise way to express them.

Examples

- `2 * 4x` is **complex** because it has **multiple coefficients** which could be simplified to `8x`
- `4x * y * j^2` is **not complex** despite being verbose because there is only a **single coefficient** and **no matching variables**

## Example Episode

A trained agent learns to distribute and simplify binomial and monomial multiplications.

### Input

`(k^4 + 7)(4 + h^2)`

`mathy:(k^4 + 7)(4 + h^2)`

### Steps

| Step                  | Text                                |
| --------------------- | ----------------------------------- |
| initial               | (k^4 + 7)(4 + h^2)                  |
| distributive multiply | (4 + h^2) \* k^4 + (4 + h^2) \* 7   |
| distributive multiply | 4k^4 + k^4 \* h^2 + (4 + h^2) \* 7  |
| commutative swap      | 4k^4 + k^4 \* h^2 + 7 \* (4 + h^2)  |
| distributive multiply | 4k^4 + k^4 \* h^2 + (7 \* 4 + 7h^2) |
| constant arithmetic   | 4k^4 + k^4 \* h^2 + (28 + 7h^2)     |
| solution              | **4k^4 + k^4 \* h^2 + 28 + 7h^2**   |

### Solution

`4k^4 + k^4 * h^2 + 28 + 7h^2`

`mathy:4k^4 + k^4 * h^2 + 28 + 7h^2`

# API

```python

import mathy_envs.envs.binomial_distribute
```


## BinomialDistribute
```python
BinomialDistribute(
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
A Mathy environment for distributing pairs of binomials.

The FOIL method is sometimes used to solve these types of problems, where
FOIL is just the distributive property applied to two binomials connected
with a multiplication.
### problem_fn
```python
BinomialDistribute.problem_fn(
    self, 
    params: mathy_envs.types.MathyEnvProblemArgs, 
) -> mathy_envs.types.MathyEnvProblem
```
Given a set of parameters to control term generation, produce
2 binomials expressions connected by a multiplication.
### transition_fn
```python
BinomialDistribute.transition_fn(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    expression: mathy_core.expressions.MathExpression, 
    features: mathy_envs.state.MathyObservation, 
) -> Optional[mathy_envs.time_step.TimeStep]
```
If there are no like terms.
