from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.space import MaskNDArray


class MaskedDiscrete(spaces.Discrete):
    r"""A masked discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    Example::
        >>> MaskedDiscrete(3, mask=(1,1,0))
    """

    def update_mask(self, mask: MaskNDArray) -> None:
        assert isinstance(mask, (tuple, list, np.ndarray))
        assert (
            len(mask) == self.n
        ), f"Mask length {len(mask)} does not match space size {self.n}."
        self.mask = np.array(mask)

    def __init__(self, n: int, mask: MaskNDArray):
        super(MaskedDiscrete, self).__init__(n)  # type:ignore
        self.update_mask(mask)

    def sample(self, mask: Optional[MaskNDArray] = None) -> np.int64:  # type:ignore
        """Generates a single random sample from this space, respecting the mask."""
        mask = self.mask if mask is None else mask
        probability = mask / np.sum(mask)
        return self.np_random.choice(self.n, p=probability)
