from typing import List, Tuple, Union

import numpy as np
from gym import spaces


class MaskedDiscrete(spaces.Discrete):
    r"""A masked discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    Example::
        >>> MaskedDiscrete(3, mask=(1,1,0))
    """
    mask: np.array

    def __init__(self, n: int, mask: Union[List[int], Tuple[int, ...]]):
        assert isinstance(mask, (tuple, list))
        assert len(mask) == n
        self.mask = np.array(mask)
        super(MaskedDiscrete, self).__init__(n)

    def sample(self) -> int:
        probability = self.mask / np.sum(self.mask)
        return self.np_random.choice(self.n, p=probability)
