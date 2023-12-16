```python

import mathy_envs.envs.complex_simplify
```
Core to working with algebra problems is the ability to simplify complex terms. Mathy provides an environment that generates problems with complex terms that require simplification to satisfy the win conditions.

## Challenge

In Complex Simplify, the agent must learn to quickly combine coefficients and like variables inside a single complex term.

Examples

- `1x * 2x^1` must be simplified to `2x^2`
- `7j * y^1 * y^2` must be simplified to `7j * y^3`

## Win Conditions

A problem is considered solved when no remaining complex terms exist in the expression.

### No Complex Terms

Terms are considered complex when there's a more concise way to express them.

Examples

- `2 * 4x` is **complex** because it has **multiple coefficients** which could be simplified to `8x`
- `4x * y * j^2` is **not complex** despite being verbose because there is only a **single coefficient** and **no matching variables**

## Example Episode

A trained agent learns to combine multiple low-level actions into higher-level ones that `simplify complex terms`

### Input

`4a^4 * 5a^4 * 2b^4`

`mathy:4a^4 * 5a^4 * 2b^4`

### Steps

| Step                    | Text                        |
| ----------------------- | --------------------------- |
| initial                 | 4a^4 \* 5a^4 \* 2b^4        |
| constant arithmetic     | **20a^4** \* a^4 \* 2b^4    |
| variable multiplication | **20 \* a^(4 + 4)** \* 2b^4 |
| constant arithmetic     | 20 \* **a^8** \* 2b^4       |
| commutative swap        | **(a^8 \* 2b^4)** \* 20     |
| commutative swap        | (**2b^4 \* a^8**) \* 20     |
| commutative swap        | 20 \* **2b^4 \* a^8**       |
| constant arithmetic     | **40b^4** \* a^8            |
| solution                | **40b^4 \* a^8**            |

### Solution

`40b^4 * a^8`

`mathy:40b^4 * a^8`


## API


## ComplexSimplify
```python
ComplexSimplify(self, ops: Optional[List[str]] = None, kwargs: Any)
```
A Mathy environment for simplifying complex terms (e.g. 4x^3 * 7y) inside of
expressions. The goal is to simplify the complex term within the allowed number
of environment steps.

### problem_fn
```python
ComplexSimplify.problem_fn(
    self, 
    params: mathy_envs.types.MathyEnvProblemArgs, 
) -> mathy_envs.types.MathyEnvProblem
```
Given a set of parameters to control term generation, produce
a complex term that has a simple representation that must be found.
- "4x * 2y^2 * 7q"
- "7j * 2z^6"
- "x * 2y^7 * 8z * 2x"

