from typing import Optional

import gym.spaces
import numpy as np


class OneHotEncoding(gym.spaces.MultiBinary):
    def sample(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        one_hot_vector = np.zeros(self.n, dtype=np.int8)
        one_hot_vector[self.np_random.integers(self.n)] = 1
        return one_hot_vector

    def contains(self, x) -> bool:
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = np.count_nonzero(x == 0)
            number_of_ones = np.count_nonzero(x == 1)
            return (number_of_zeros == (self.n - 1)) and (number_of_ones == 1)
        else:
            return False

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"OneHotEncoding({self.n})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return isinstance(other, OneHotEncoding) and self.n == other.n
