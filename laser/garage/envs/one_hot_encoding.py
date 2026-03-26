from typing import Optional

import numpy as np
import gymnasium as gym


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

    def flatten(self, x):
        """Return a flattened observation x.

        Args:
            x (:obj:`Iterable`): The object to flatten.

        Returns:
            np.ndarray: An array of x collapsed into one dimension.

        """

    def unflatten(self, x):
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            np.ndarray: An array of x in the shape of self.shape.

        """

    def flatten_n(self, xs):
        """Return flattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element.

        """

    def unflatten_n(self, xs):
        """Return unflattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element and self.shape.

        """

    def concat(self, other):
        """Concatenate with another space of the same type.

        Args:
            other (Space): A space to be concatenated with this space.

        Returns:
            Space: A concatenated space.

        """

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""

    def to_tf_placeholder(self, name, batch_dims):
        """Create a tensor placeholder from the Space object.

        Args:
            name (str): name of the variable
            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of the object.

        Returns:
            tf.Tensor: Tensor object with the same properties as
                the Dict where the shape is modified by batch_dims.

        """

    def to_theano_tensor(self, name, batch_dims):
        """Create a theano tensor from the Space object.

        Args:
            name (str): name of the variable
            batch_dims (:obj:`list`): batch dimensions to add to the
                shape of the object.

        Returns:
            theano.tensor.TensorVariable: Tensor object with the
                same properties as the Dict where the shape is
                modified by batch_dims.

        """

    @property
    def padding_token(self):
        return np.zeros(self.n, dtype=np.int8)

    def is_padding(self, x):
        return np.all(x == self.padding_token)
