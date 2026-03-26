"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import random

import gymnasium as gym
import torch

from laser.garage.envs.env_utils.vec_env import VecEnvWrapper
from laser.garage.envs.env_utils.vec_env.dummy_vec_env import DummyVecEnv
from laser.garage.envs.env_utils.vec_env.subproc_vec_env import SubprocVecEnv, SubprocVecTensorEnv
from laser.garage.envs.env_utils.vec_env.vec_normalize import VecNormalize
from laser.garage.envs.wrappers import TimeLimitMask, BAMDPWrapper


def make_env(env_id, seed, rank, episodes_per_task, tasks, h, add_done_info, i=None, **kwargs):
    def _thunk():
        if i is not None:
            kwargs['i'] = i
        env = gym.make(env_id, seed=seed+rank, h=h, **kwargs)
        if tasks is not None:
            env.unwrapped.reset_task = lambda x: env.unwrapped.set_task(random.choice(tasks))
        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        env = BAMDPWrapper(env=env, episodes_per_task=episodes_per_task, add_done_info=add_done_info)
        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma,
                  device, episodes_per_task,
                  normalise_rew, tasks,
                  h,
                  env_index=None,
                  rank_offset=0,
                  add_done_info=None,
                  eval=False,
                  **kwargs):
    """
    :param ret_rms: running return and std for rewards
    """
    if 'Torch' in env_name:
        kwargs['device'] = device
        subproc = SubprocVecTensorEnv
        vec = VecTensor
    else:
        subproc = SubprocVecEnv
        vec = VecPyTorch

    envs = [make_env(env_id=env_name, seed=seed, rank=rank_offset + i,
                     episodes_per_task=episodes_per_task,
                     tasks=tasks,
                     h=h,
                     add_done_info=add_done_info,
                     i=env_index,
                     **kwargs)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = subproc(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, normalise_rew=normalise_rew)
        else:
            envs = VecNormalize(envs, normalise_rew=normalise_rew, gamma=gamma)
        if eval:
            envs.eval()

    envs = vec(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset_mdp(self, index=None):
        obs = self.venv.reset_mdp(index=index)
        if isinstance(obs, list):
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def reset(self, index=None, task=None):
        if task is not None:
            assert isinstance(task, list)
        state = self.venv.reset(index=index, task=task)
        if isinstance(state, list):
            state = [torch.from_numpy(s).float().to(self.device) for s in state]
        elif isinstance(state, torch.Tensor):
            pass
        else:
            state = torch.from_numpy(state).float().to(self.device)
        return state

    def reset_meta_traj(self):
        state = self.venv.reset_meta_traj()
        if isinstance(state, list):
            state = [torch.from_numpy(s).float().to(self.device) for s in state]
        elif isinstance(state, torch.Tensor):
            pass
        else:
            state = torch.from_numpy(state).float().to(self.device)
        return state

    def step_async(self, actions):
        # actions = actions.squeeze(1).cpu().numpy()
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        state, reward, done, info = self.venv.step_wait()
        if isinstance(state, list):  # raw + normalised
            state = [torch.from_numpy(s).float().to(self.device) for s in state]
        elif isinstance(state, torch.Tensor):
            pass
        else:
            state = torch.from_numpy(state).float().to(self.device)
        if isinstance(reward, list):  # (raw, normalised)
            for i in range(len(reward)):
                if isinstance(reward[i], torch.Tensor):
                    reward[i] = reward[i].unsqueeze(dim=1)
                else:
                    reward[i] = torch.from_numpy(reward[i]).unsqueeze(dim=1).float().to(self.device)
        elif isinstance(reward, torch.Tensor):
            reward = reward.unsqueeze(dim=1)
        else:
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        return state, reward, done, info

    def reset_task(self, tasks):
        assert isinstance(tasks, list)
        self.venv.reset_task(tasks)

    def compute_eval_metrics(self, infos, total_tasks):
        return self.venv.compute_eval_metrics((infos, total_tasks))

    @property
    def train_tasks(self):
        return self.venv.train_tasks

    @property
    def test_tasks(self):
        return self.venv.test_tasks

    def __getattr__(self, attr):
        """ If env does not have the attribute then call the attribute in the wrapped_env """

        if attr in ['_max_episode_steps', 'task_dim', 'belief_dim', 'num_states']:
            return self.unwrapped.get_env_attr(attr)

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


class VecTensor(VecPyTorch):
    def step_async(self, actions):
        actions = actions
        self.venv.step_async(actions)
