Core to working with algebra problems is the ability to combine like terms in polynomials. Mathy provides an environment that generates problems that require simplification to satisfy the win conditions.

## Challenge

In Poly Simplify, the agent must learn to quickly combine and simplify all the like terms in the generated input expression.

Examples

- `4x + 2y + 2x` must be simplified to `6x + 2y`
- `23j + 7 + 12x + 2j` must be simplified to `25j + 7 + 12x`
- `1.3j + 2j - 7` must be simplified to `3.3j - 7`

## Win Conditions

Solve problems by combining all like terms in the provided expression.

### No Like Terms

Terms are like when connected by an addition or subtraction, and both terms share a variable and exponent.

Examples

- `4x + 2y` there are no like terms because `x` and `y` are **different variables**
- `2x^2 + 4x` there are no like terms because `x^2` and `x` have **different exponents**
- `82x + 14x` the terms are like because `x` and `x` are the same
- `12x + 12y` there are no like terms because `x` and `y` **different variables**

### No Complex Terms

Complex terms are those that can be restated more simply.

Examples

- `2 * 4x` is **complex** because it has **multiple coefficients** which could be simplified to `8x`
- `4x * y * j^2` is **not complex** despite being verbose because there is only a **single coefficient** and **no matching variables**

## Example Episode

A trained agent learns to combine multiple low-level actions into higher-level ones that combine like terms.

### Input

`1k + 210r + 7z + 11k + 10z`

`mathy:1k + 210r + 7z + 11k + 10z`

### Steps

| Step                   | Text                                     |
| ---------------------- | ---------------------------------------- |
| input                  | 1k + 210r + 7z + 11k + 10z               |
| commutative swap       | 11k + **(1k + 210r + 7z)** + 10z         |
| distributive factoring | 11k + (1k + 210r) + **(7 + 10) \* z**    |
| distributive factoring | **(11 + 1) \* k** + 210r + (7 + 10) \* z |
| constant arithmetic    | (11 + 1) \* k + 210r + **17z**           |
| constant arithmetic    | **12k** + 210r + 17z                     |
| solution               | **12k + 210r + 17z**                     |

### Solution

`12k + 210r + 17z`

`mathy:12k + 210r + 17z`

# API

```python

import mathy_envs.envs.poly_simplify
```


## PolySimplify
```python
PolySimplify(self, ops: Optional[List[str]] = None, kwargs: Any)
```
A Mathy environment for simplifying polynomial expressions.

NOTE: This environment only generates polynomial problems with
 addition operations. Subtraction, Multiplication and Division
 operators are excluded. This is a good area for improvement.

### problem_fn
```python
PolySimplify.problem_fn(
    self, 
    params: mathy_envs.types.MathyEnvProblemArgs, 
) -> mathy_envs.types.MathyEnvProblem
```
Given a set of parameters to control term generation, produce
a polynomial problem with (n) total terms divided among (m) groups
of like terms. A few examples of the form: `f(n, m) = p`
- (3, 1) = "4x + 2x + 6x"
- (6, 4) = "4x + v^3 + y + 5z + 12v^3 + x"
- (4, 2) = "3x^3 + 2z + 12x^3 + 7z"

### transition_fn
```python
PolySimplify.transition_fn(
    self, 
    env_state: mathy_envs.state.MathyEnvState, 
    expression: mathy_core.expressions.MathExpression, 
    features: mathy_envs.state.MathyObservation, 
) -> Optional[mathy_envs.time_step.TimeStep]
```
If there are no like terms.
