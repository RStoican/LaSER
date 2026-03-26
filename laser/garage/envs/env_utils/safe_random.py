"""
A random number generator that is thread safe. To be used in environments. Uses the Python native random functions
instead of np.random.

Recommended by: https://github.com/lmzintgraf/varibad?tab=readme-ov-file
"""
import random


class SafeRandom:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def random(self):
        return random.random()

    def integers(self, low, high=None):
        if high is None:
            high = low
            low = 0
        return random.randrange(low, high)

    def randint(self, low, high=None):
        return self.integers(low, high)

    def choice(self, a, size=None, replace=True):
        if replace:
            raise NotImplementedError
        if size is None:
            size = 1

        samples = random.sample(a, size)
        return samples if size != 1 else samples[0]

    def uniform(self, a, b):
        return a + (b-a) * self.random()

    def normal(self, loc, scale, thread_safe=True):
        if thread_safe:
            # Slower, but thread safe
            return random.normalvariate(mu=loc, sigma=scale)
        # Faster, but might cause issues in multithreading runs
        return random.gauss(mu=loc, sigma=scale)
