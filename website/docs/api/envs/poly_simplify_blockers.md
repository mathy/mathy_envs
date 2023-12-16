```python

import mathy_envs.envs.poly_simplify_blockers
```

## PolySimplifyBlockers
```python
PolySimplifyBlockers(self, ops: Optional[List[str]] = None, kwargs: Any)
```
A Mathy environment for polynomial problems that have a variable
string of mismatched terms separating two like terms.

The goal is to:
  1. Commute the like terms so they become siblings
  2. Combine the sibling like terms
