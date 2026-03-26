"""
Taken from https://github.com/openai/baselines
"""

import numpy as np
import torch

from laser.garage.torch._functions import global_device


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self._create_rms(shape)
        self.count = epsilon

    def _create_rms(self, shape):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + self._square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def _square(self, delta):
        return np.square(delta)


class TorchRunningMeanStd(RunningMeanStd):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.device = global_device() if device is None else device
        super().__init__(epsilon, shape)

    def _create_rms(self, shape):
        self.mean = torch.zeros(shape).float().to(self.device)
        self.var = torch.ones(shape).float().to(self.device)

    def update(self, x):
        x = x.permute(0, 1, 3, 2)  # (p, q, m*H, d)
        x = x.reshape((-1, x.shape[-1]))  # (p*q*m*H, d)
        batch_mean = x.mean(dim=0)  # (d)
        batch_var = x.var(dim=0) if x.shape[0] > 1 else 0  # (d)
        batch_count = x.shape[0]  # p*q*m*H
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def masked_update(self, x, lens, h):
        """

        Args:
            x: (p, q, d, m*H)
            lens: (p, q, m)
            h: Horizon

        Returns:

        """
        d = x.shape[2]
        x = x.permute(0, 1, 3, 2)  # (p, q, m*H, d)
        x = x.reshape(-1, h, d)  # (p*q*m, H, d)
        batch = x.shape[0]

        mask = torch.arange(h).expand(batch, h).to(global_device())  # (p*q*m, H)
        mask = mask <= lens.reshape(-1).unsqueeze(1)  # (p*q*m, H)
        mask = mask.unsqueeze(-1).repeat_interleave(repeats=d, dim=-1)  # (p*q*m, H, d)

        x = x.reshape(-1, d)  # (p*q*m*H, d)
        mask = mask.reshape(-1, d)  # (p*q*m*H, d)

        masked_x = mask * x
        sums = masked_x.sum(dim=0)  # (d)
        batch_count = mask.sum(dim=0)  # (d)
        batch_mean = sums / batch_count  # (d)

        squared_diffs = ((x - batch_mean) ** 2) * mask
        batch_var = squared_diffs.sum(dim=0) / batch_count  # (d)

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def _square(self, delta):
        return torch.pow(delta, 2)
