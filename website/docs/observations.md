Mathy Env observations contain a rich set of features that can be represented in multiple formats to suit different neural network architectures. All observation types can optionally be normalized to the range 0-1.

Let's quickly review how we go from AsciiMath text inputs to different observation formats for neural networks to consume:

## Text -> Tree -> Observations

[Mathy Core](https://core.mathy.ai) processes an input problem by parsing its text into a tree, then mathy envs convert that tree into different observation formats depending on your model architecture needs.

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

## Observation Types

Mathy supports four different observation formats to accommodate different neural network architectures:

### 1. Flat Observations (Default)

The original observation format that represents the expression as a flat sequence of features. This format first converts the tree to ordered lists, then concatenates all information into a single 1D array.

**Use cases**: Traditional MLPs, RNNs, transformers expecting sequential input

#### Tree to List Conversion

Rather than expose [tree structures](https://core.mathy.ai/api/expressions/#mathexpression) directly, flat observations [traverse them](https://core.mathy.ai/api/expressions/#to_list) to produce node/value lists.

!!! info "tree list ordering"

    You might have noticed that the tree features are not expressed in the natural order we might read. As observed by [Lample and Charton](https://arxiv.org/pdf/1912.01412.pdf){target=\_blank} trees must be visited in an order that preserves the order-of-operations so that the model can pick up on the hierarchical features of the input.

    For this reason, we visit trees in `pre` order for serialization.

Converting math expression trees to lists is done with a helper:

```python
{!./snippets/envs/tree_to_list.py!}
```

#### Flat Observation Structure

**Features included**:

- Problem type hash (2 values)
- Relative episode time (1 value)
- Node types (padded sequence)
- Node values (padded sequence)
- Action mask (flattened)

```python
{!./snippets/envs/flat_observations.py!}
```

### 2. Graph Observations

Represents the mathematical expression as an adjacency matrix with node features, suitable for Graph Convolutional Networks (GCNs) and similar architectures. This format works directly with the tree structure.

**Use cases**: Graph Convolutional Networks, Graph Attention Networks, predictive coding models

**Structure**:

- `node_features`: `[type_id, value, time, is_leaf]` for each node
- `adjacency`: Binary matrix encoding parent-child relationships
- `action_mask`: Valid actions at each node
- `num_nodes`: Actual number of nodes (before padding)

**Edge semantics**: Parent nodes connect to their children, preserving mathematical precedence

```python
{!./snippets/envs/graph_observations.py!}
```

### 3. Hierarchical Observations

Groups nodes by their depth in the expression tree, enabling models to process expressions level by level. This format preserves the tree structure while organizing nodes by hierarchy.

**Use cases**: Hierarchical processing models, predictive coding architectures, models that benefit from depth-aware processing

**Structure**:

- `node_features`: `[type_id, value, time, is_leaf]` for each node
- `level_indices`: Tree depth for each node
- `action_mask`: Valid actions at each node
- `max_depth`: Maximum tree depth
- `num_nodes`: Actual number of nodes

**Organization**: Nodes are ordered by tree level, allowing models to process expressions hierarchically

```python
{!./snippets/envs/hierarchical_observations.py!}
```

### 4. Message Passing Observations

Formats expressions for PyTorch Geometric-style Graph Neural Networks with explicit edge lists and edge types. This format also works directly with the tree structure.

**Use cases**: PyTorch Geometric models, Graph Neural Networks with edge features, message passing architectures

**Structure**:

- `node_features`: `[type_id, value, time, is_leaf]` for each node
- `edge_index`: PyG-format edge list `(2, num_edges)`
- `edge_types`: Edge type for each edge (0=left child, 1=right child)
- `action_mask`: Valid actions at each node
- `num_nodes`: Actual number of nodes
- `num_edges`: Actual number of edges

**Edge types**:

- `0`: Parent → left child relationship
- `1`: Parent → right child relationship

```python
{!./snippets/envs/message_passing_observations.py!}
```

## Unified Observation Interface

All observation types are accessible through a single interface:

```python
{!./snippets/envs/unified_observations.py!}
```

## Common Features Across All Types

All observation formats share these characteristics:

### Node Features

Every observation type uses consistent node features: `[type_id, value, time, is_leaf]`

- **type_id**: Integer representing the node's mathematical operation or value type
- **value**: Floating-point value for constants, 0.0 for operators
- **time**: Normalized episode progress (0.0 = start, 1.0 = end)
- **is_leaf**: Binary indicator (1.0 for leaf nodes, 0.0 for operators)

### Action Mask

The action mask format is identical across all observation types:

- Flattened array of size `num_rules × max_seq_len`
- Binary values: 1.0 = valid action, 0.0 = invalid action
- Represents which transformation rules can be applied to which nodes

### Normalization

When `normalize=True` (default), all features are scaled to the range [0, 1]:

- Node types and values are min-max normalized
- Time features are already normalized (episode progress)
- Action masks remain binary (0/1)

### Padding

All observations are padded to `max_seq_len` to enable batching:

- Node features padded with zeros
- Adjacency matrices padded to square matrices
- Edge lists padded with dummy edges
- Action masks padded to consistent size

## Choosing an Observation Type

| Architecture                | Recommended Type      | Reason                                      |
| --------------------------- | --------------------- | ------------------------------------------- |
| MLP, RNN, Transformer       | Flat                  | Sequential processing of flattened features |
| Graph Convolutional Network | Graph                 | Native adjacency matrix representation      |
| Graph Attention Network     | Graph                 | Node features + adjacency for attention     |
| PyTorch Geometric GNN       | Message Passing       | Optimized edge list format                  |
| Hierarchical/Tree LSTM      | Hierarchical          | Depth-aware processing                      |
| Predictive Coding Models    | Hierarchical or Graph | Level-based or parent-child predictions     |

## Example Usage

### Basic Observation Creation

```python
{!./snippets/envs/lists_to_observations.py!}
```

### Environment Integration

```python
{!./snippets/envs/env_observations.py!}
```

The observation system provides flexibility to match your model architecture while maintaining consistent feature representations across all formats.
