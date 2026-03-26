"""
Taken from https://github.com/openai/baselines
"""
from multiprocessing import Process, Pipe

import numpy as np
import torch

from . import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'reset_mdp':
                ob = env.reset_mdp()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'get_task':
                remote.send(env.get_task())
            elif cmd == 'reset_task':
                env.unwrapped.reset_task(data)
            elif cmd == 'reset_meta_traj':
                ob = env.reset_meta_traj()
                remote.send(ob)
            elif cmd == 'compute_eval_metrics':
                metrics = env.compute_eval_metrics(data)
                remote.send(metrics)
            else:
                # try to get the attribute directly
                remote.send(getattr(env.unwrapped, cmd))
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple envs in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create envs to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        self.backend = np
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return self.backend.stack(obs), self.backend.stack(rews), self.backend.stack(dones), infos

    def reset(self, task=None):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', task))
        return self.backend.stack([remote.recv() for remote in self.remotes])

    def reset_task(self, tasks):
        self._assert_not_closed()
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))

    def reset_mdp(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset_mdp', None))
        return self.backend.stack([remote.recv() for remote in self.remotes])

    def reset_meta_traj(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset_meta_traj', None))
        return self.backend.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def get_env_attr(self, attr):
        self.remotes[0].send((attr, None))
        return self.remotes[0].recv()

    def get_task(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_task', None))
        return self.backend.stack([remote.recv() for remote in self.remotes])

    # FIXME Check if we need
    def get_belief(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_belief', None))
        return self.backend.stack([remote.recv() for remote in self.remotes])

    @property
    def train_tasks(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('train_tasks', None))
        return [remote.recv() for remote in self.remotes]

    @property
    def test_tasks(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('test_tasks', None))
        return [remote.recv() for remote in self.remotes]

    @property
    def tester_type(self):
        self._assert_not_closed()
        self.remotes[0].send(('tester_type', None))
        return self.remotes[0].recv()


class SubprocVecTensorEnv(SubprocVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.backend = torch

    def reset_meta_traj(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset_meta_traj', None))
        return torch.cat([remote.recv().unsqueeze(0) for remote in self.remotes], dim=0)
    # def step_wait(self):
    #     self._assert_not_closed()
    #     results = [remote.recv() for remote in self.remotes]
    #     self.waiting = False
    #     obs, rews, dones, infos = zip(*results)
    #     return torch.stack(obs), torch.stack(rews), torch.stack(dones), infos
    #     return self.backend.stack(obs), self.backend.stack(rews), self.backend.stack(dones), infos
    #
    # def reset(self, task=None):
    #     self._assert_not_closed()
    #     for remote in self.remotes:
    #         remote.send(('reset', task))
    #     return torch.stack([remote.recv() for remote in self.remotes])
    #     return self.backend.stack([remote.recv() for remote in self.remotes])
    #
    # def reset_task(self, tasks):
    #     self._assert_not_closed()
    #     for remote, task in zip(self.remotes, tasks):
    #         remote.send(('reset_task', task))
    #
    # def reset_mdp(self):
    #     self._assert_not_closed()
    #     for remote in self.remotes:
    #         remote.send(('reset_mdp', None))
    #     return self.backend.stack([remote.recv() for remote in self.remotes])
    #
    # def reset_meta_traj(self):
    #     self._assert_not_closed()
    #     for remote in self.remotes:
    #         remote.send(('reset_meta_traj', None))
    #     return torch.cat([remote.recv().unsqueeze(0) for remote in self.remotes], dim=0)
    #     return self.backend.stack([remote.recv() for remote in self.remotes])
    #
    # def get_task(self):
    #     self._assert_not_closed()
    #     for remote in self.remotes:
    #         remote.send(('get_task', None))
    #     return self.backend.stack([remote.recv() for remote in self.remotes])
    #
    # # FIXME Check if we need
    # def get_belief(self):
    #     self._assert_not_closed()
    #     for remote in self.remotes:
    #         remote.send(('get_belief', None))
    #     return self.backend.stack([remote.recv() for remote in self.remotes])