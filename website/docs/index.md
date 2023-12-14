# mathy_envs

<p align="center">
    <em>Develop agents that can complete Mathy's challenging algebra environments.</em>
</p>
<p align="center">
<a href="https://github.com/mathy/mathy_envs/actions">
    <img src="https://github.com/mathy/mathy_envs/workflows/Build/badge.svg" />
</a>
<a href="https://codecov.io/gh/mathy/mathy_envs">
    <img src="https://codecov.io/gh/mathy/mathy_envs/branch/master/graph/badge.svg?token=CqPEOdEMJX" />
</a>
<a href="https://pypi.org/project/mathy_envs" target="_blank">
    <img src="https://badge.fury.io/py/mathy_envs.svg" alt="Package version">
</a>
</p>


Mathy includes a framework for building reinforcement learning environments that transform math expressions using a set of user-defined actions.

Built-in environments aim to simplify algebra problems and expose generous customization points for user-created ones.


- **Large Action Spaces**: Mathy environments have 2d action spaces, where you can apply any known rule to any node in the tree. Without masking this makes mathy environments very difficult to explore.
- **Masked Action Support** To enable better curriculum learning and toy problem creation, mathy agents are given access to a mask of valid actions given the state of the system. When used to select actions, mathy environments become much easier to explore.
- **Rich Reward Signals**: The environments are constructed such that agents receive reward feedback at every action. Custom environments can implement their own reward schemes.
- **Curriculum Learning**: Mathy envs cover related but distinct math problem types, and scale their complexity based on multiple controllable inputs. They also include built-in easy/normal/hard variants of each environment.

## Requirements

- Python 3.6+

## Installation

```bash
$ pip install mathy_envs
```

## Episodes

Mathy agents interact with environments through sequences of interactions called episodes, which follow a standard RL episode lifecycle:

!!! info "Episode Pseudocode."

    1.  set **state** to an **initial state** from the **environment**
    2.  **while** **state** is not **terminal**
        - take an **action** and update **state**
    3.  **done**

## Extensions

Because algebra problems represent only a tiny sliver of the uses for math expression trees, Mathy has customization points to alter or create entirely new environments with little effort.

### New Problems

Generating a new problem type while subclassing a base environment is the simplest way to create a custom challenge for the agent.

You can inherit from a base environment like [Poly Simplify](/envs/poly_simplify), which has win-conditions that require all the like-terms to be gone from an expression and all complex terms to be simplified. From there, you can provide any valid input expression:

```Python
{!./snippets/envs/custom_problem_text.py!}
```

### New Actions

Build your tree transformation actions and use them with the built-in agents:

```Python
{!./snippets/envs/custom_actions.py!}
```

### Custom Win Conditions

Environments can implement custom logic for win conditions or inherit them from a base class:

```Python
{!./snippets/envs/custom_win_conditions.py!}
```

### Custom Timestep Rewards

Specify which actions to give the agent positive and negative rewards:

```Python
{!./snippets/envs/custom_timestep_rewards.py!}
```

### Custom Episode Rewards

Specify (or calculate) custom floating-point episode rewards:

```Python
{!./snippets/envs/custom_episode_rewards.py!}
```

## Other Libraries

Mathy supports alternative reinforcement learning libraries.

### Gymnasium

Mathy has support [Gymnasium](https://gymnasium.farama.org/){target=\_blank} via a small wrapper.

You can import the `mathy_envs.gym` module separately to register the environments:

```python
{!./snippets/envs/openai_gym.py!}
```

## Contributors

Mathy wouldn't be possible without the contributions of the following people:

<div class="contributors-wrapper">
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a target="_blank" href="https://www.justindujardin.com/"><img src="https://avatars0.githubusercontent.com/u/101493?v=4" width="100px;" alt=""/><br /><sub><b>Justin DuJardin</b></sub></a></td>
    <td align="center"><a target="_blank" href="https://twitter.com/Miau_DB"><img src="https://avatars3.githubusercontent.com/u/7149899?v=4" width="100px;" alt=""/><br /><sub><b>Guillem Duran Ballester</b></sub></a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
</div>

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!
