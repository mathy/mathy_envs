Mathy Env observations contain a rich set of features, and can optionally be normalized to the range 0-1.

Let's quickly review how we go from AsciiMath text inputs to a set of feature for neural networks to consume:

## Text -> Observation

[Mathy Core](https://core.mathy.ai) processes an input problem by parsing its text into a tree, then mathy envs convert that into a sequence of nodes/values, and finally those features are concatenated with the current environment time, type, and valid action mask.

### Text to Tree

A problem text is [encoded into tokens](https://core.mathy.ai/api/tokenizer), then [parsed into a tree](https://core.mathy.ai/api/parser) that preserves the order of operations while removing parentheses and whitespace.
Consider the tokens and tree that result from the input: `-3 * (4 + 7)`

**Tokens**

`tokens:-3 * (4 + 7)`

**Tree**

`mathy:-3 * (4 + 7)`

Please observe that the tree representation is more concise than the tokens array because it doesn't have nodes for hierarchical features like parentheses.

Converting text to trees is accomplished with the [expression parser](https://core.mathy.ai/api/parser):

```python
{!./snippets/envs/text_to_tree.py!}
```

### Tree to List

Rather than expose [tree structures](https://core.mathy.ai/api/expressions/#mathexpression) to environments, we [traverse them](https://core.mathy.ai/api/expressions/#to_list) to produce node/value lists.

!!! info "tree list ordering"

    You might have noticed that the previous tree features are not expressed in the natural order we might read. As observed by [Lample and Charton](https://arxiv.org/pdf/1912.01412.pdf){target=\_blank} trees must be visited in an order that preserves the order-of-operations so that the model can pick up on the hierarchical features of the input.

    For this reason, we visit trees in `pre` order for serialization.

Converting math expression trees to lists is done with a helper:

```python
{!./snippets/envs/tree_to_list.py!}
```

### Lists to Observations

Mathy turns a list of math expression nodes into a feature list that captures the input characteristics. Specifically, mathy converts a node list into two lists, one with **node types** and another with **node values**:

`features:-3 * (4 + 7)`

- The first row contains input token characters stripped of whitespace and parentheses.
- The second row is the sequence of floating-point **node values** for the tree, with each non-constant node represented by a mask value.
- The third row is the **node type** integer representing the node's class in the tree.

While feature lists may be directly passable to an ML model, they don't include any information about the problem's state over time. To work with information over time, mathy agents draw extra information from the environment when building observations. This additional information includes:

- **Environment Problem Type**: environments all specify an [environment namespace](/api/env/#get_env_namespace) that is converted into a pair of [hashed string values](/api/state/#get_problem_hash) using different random seeds.
- **Episode Relative Time**: each observation can see a 0-1 floating-point value that indicates how close the agent is to running out of moves.
- **Valid Action Mask**: mathy gives weighted estimates for each action at every node. If there are five possible actions and ten nodes in the tree, there are **up to** 50 possible actions. A same-sized (e.g., 50) mask of 0/1 values is provided so the model can mask out nodes with no valid actions when returning probability distributions.

Mathy has utilities for making the conversion:

```python
{!./snippets/envs/lists_to_observations.py!}
```
