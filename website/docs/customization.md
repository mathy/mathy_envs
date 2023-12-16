
Because algebra problems represent only a tiny sliver of the uses for math expression trees, Mathy Envs has customization points to alter or create entirely new environments with little effort.

Let's consider a few examples:

### New Problems

Generating a new problem type while subclassing a base environment is likely the simplest way to create a custom challenge for the agent.

You can inherit from a base environment like [Poly Simplify](api/envs/poly_simplify.md), which has win-conditions that require all the like-terms to be gone from an expression and all complex terms to be simplified. From there, you can provide any valid input expression:

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
