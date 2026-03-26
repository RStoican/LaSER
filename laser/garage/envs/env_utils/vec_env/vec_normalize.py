"""
Taken from https://github.com/openai/baselines
"""
import numpy as np
import torch

from laser.garage.torch._functions import global_device
from laser.garage.utils import helpers as utl
from . import VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, clipobs=10., cliprew=10., gamma=0.99, epsilon=1e-8,
                 normalise_rew=False):
        VecEnvWrapper.__init__(self, venv)

        self.normalise_rew = normalise_rew

        # clip params
        self.clipobs = clipobs
        self.cliprew = cliprew

        # discounted return for each environment
        self.ret = torch.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def step_wait(self):
        # execute action
        obs, rews, news, infos = self.venv.step_wait()

        # normalise
        rews = self._rewfilt(rews, news)
        return obs, rews, news, infos

    def _rewfilt(self, rews, news):
        if self.normalise_rew:
            ret_rms = utl.rew_rms()
            # update discounted return
            self.ret = self.ret * self.gamma + rews
            self.ret[news] = 0.
            # update rolling mean / std
            if self.training:
                utl.update_rew_rms(self.ret)
            # normalise
            rews_norm = np.clip(rews / np.sqrt(ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return [rews, rews_norm]
        else:
            return [rews, rews]

    def reset_mdp(self, index=None):
        if index is None:
            obs = self.venv.reset_mdp()
        else:
            self.venv.remotes[index].send(('reset_mdp', None))
            obs = self.venv.remotes[index].recv()
        return obs

    def reset_meta_traj(self):
        obs = self.venv.reset_meta_traj()
        return obs

    def reset(self, index=None, task=None):
        self.ret = torch.zeros(self.num_envs)
        if index is None:
            obs = self.venv.reset(task=task)
        else:
            try:
                self.venv.remotes[index].send(('reset', task))
                obs = self.venv.remotes[index].recv()
            except AttributeError:
                obs = self.venv.envs[index].reset(task=task)
        return obs

    def compute_eval_metrics(self, data):
        self.venv.remotes[0].send(('compute_eval_metrics', data))
        return self.venv.remotes[0].recv()

    @property
    def train_tasks(self):
        return self.venv.train_tasks

    @property
    def test_tasks(self):
        return self.venv.test_tasks

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
        """
        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

