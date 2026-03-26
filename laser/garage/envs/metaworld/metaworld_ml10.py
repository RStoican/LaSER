from abc import ABC
from typing import Iterable

import gymnasium as gym
import metaworld
import numpy as np
from laser.garage.envs._utils import from_gym


class MetaWorldML10(gym.Env, ABC):
    def __init__(self, seed, task_type, mode, given_tasks, h=200):
        # Construct the benchmark, sampling tasks
        ml10 = metaworld.ML10(seed=seed)

        # Create an environment with given task type
        self.train_env = ml10.train_classes
        self.test_env = ml10.test_classes
        if mode == 'train':
            env_collection = self.train_env
        elif mode == 'test':
            env_collection = self.test_env
        else:
            raise ValueError(f'Expected environment mode to be train or test, got {mode}')

        if given_tasks is not None:
            self.train_tasks, self.test_tasks = given_tasks
        else:
            self.train_tasks = ml10.train_tasks
            self.test_tasks = ml10.test_tasks
        tasks = self.train_tasks if mode == 'train' else self.test_tasks

        self._env_collection = []
        for name, env_cls in env_collection.items():
            env = env_cls()
            env.max_path_length = h

            env_tasks = [task for task in tasks if task.env_name == name]
            task_id = env.np_random.integers(len(env_tasks))

            env.set_task(env_tasks[task_id])
            self._env_collection.append((env, task_id))

        self._env = None

        self._task_id = None
        self.reset_task()

        self._max_episode_steps = h
        self.max_episode_steps = h
        self._env.max_path_length = self._max_episode_steps

        self._step = 0

        # self.train_env.action_space = from_gym(self.train_env.action_space)
        # self.train_env.observation_space = from_gym(self.train_env.observation_space)
        # self.test_env.action_space = from_gym(self.test_env.action_space)
        # self.test_env.observation_space = from_gym(self.test_env.observation_space)

    def step(self, action):
        if self.is_padding_action(action):
            # print(action)
            # print('===== ADDING PADDING =====')
            obs = np.zeros(self.observation_space.shape)
            info = {'success': 0,
                    'task_id': self._task_id, 'reward_raw': 0, 'padding': True}
            return obs, 0, False, False, info

        obs, reward, done, truncate, info = self._env.step(action)
        # FIXME Remove info['success'] = self.env.np_random.choice([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if done:
            raise NotImplementedError
        info.update({'task_id': self._task_id, 'reward_raw': reward, 'padding': False})
        # The task is done either when the agent succeeds, or when it reaches the max number of steps
        self._step += 1
        done = truncate or self._step >= self.max_episode_steps
        return obs, reward, done, truncate, info

    def reset(self, seed=None, return_info=False, options=None):
        self._step = 0
        obs, info = self._env.reset()
        return obs

    def reset_task(self, task_id=None):
        self._step = 0

        env_id = task_id
        if isinstance(env_id, Iterable):
            assert len(env_id) == 2
            env_id = env_id[0]

        env_id = self._env_collection[0][0].np_random.integers(len(self._env_collection)) if env_id is None else env_id
        env, task = self._env_collection[env_id]

        self._env = env
        self._task_id = (env_id, task)

        return self._task_id

    def set_task(self, task):
        raise NotImplementedError
        self._task = task
        self._env.set_task(self._task)

    def get_task(self):
        return self._task_id

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def is_padding_action(self, action):
        # FIXME
        return np.all(action < -1000)
        # return np.all(action == np.full(self.action_space.shape[0], -1234.5678, dtype=float))
