from abc import ABC
from typing import Iterable

import gymnasium as gym
import metaworld
import numpy as np
from laser.garage.envs._utils import from_gym


class MetaWorldML1(gym.Env, ABC):
    def __init__(self, seed, task_type, mode, given_tasks, h=200):
        self.mode = mode

        # Construct the benchmark, sampling tasks
        ml1 = metaworld.ML1(task_type, seed=seed)

        # Create an environment with given task type
        self.train_env = ml1.train_classes[task_type]()
        self.test_env = ml1.test_classes[task_type]()
        if self.mode == 'train':
            self._env = self.train_env
        elif self.mode == 'test':
            self._env = self.test_env
        else:
            raise ValueError(self._invalid_mode())

        if given_tasks is not None:
            self.train_tasks, self.test_tasks = given_tasks
        else:
            self.train_tasks = ml1.train_tasks
            self.test_tasks = ml1.test_tasks

        self._task_id, self._task = None, None
        self.reset_task()

        self._max_episode_steps = h
        self.max_episode_steps = h
        self._env.max_path_length = self._max_episode_steps

        self._step = 0

        # self.train_env.action_space = from_gym(self.train_env.action_space)
        # self.train_env.observation_space = from_gym(self.train_env.observation_space)
        # self.test_env.action_space = from_gym(self.test_env.action_space)
        # self.test_env.observation_space = from_gym(self.test_env.observation_space)

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == 'train':
            self._env = self.train_env
        elif self.mode == 'test':
            self._env = self.test_env
        else:
            raise ValueError(self._invalid_mode())
        self._env.max_path_length = self._max_episode_steps

    def step(self, action):
        if self.is_padding_action(action):
            # print(action)
            # print('===== ADDING PADDING =====')
            obs = np.zeros(self.observation_space.shape)
            info = {'success': 0,
                    'task': self._task, 'task_id': self._task_id, 'reward_raw': 0, 'padding': True}
            return obs, 0, False, False, info

        obs, reward, done, truncate, info = self._env.step(action)
        # FIXME Remove info['success'] = self.env.np_random.choice([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if done:
            raise NotImplementedError
        info.update({'task': self._task, 'task_id': self._task_id, 'reward_raw': reward, 'padding': False})
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
        if self.mode == 'train':
            tasks = self.train_tasks
        elif self.mode == 'test':
            tasks = self.test_tasks
        else:
            raise ValueError(self._invalid_mode())

        if isinstance(task_id, Iterable):
            assert len(task_id) == 1
            task_id = task_id[0]
        task_id = self._env.np_random.integers(len(tasks)) if task_id is None else task_id
        self._task_id = task_id
        self._task = tasks[task_id]

        self._env.set_task(self._task)
        return self._task_id

    def set_task(self, task):
        self._task = task
        self._env.set_task(self._task)

    def get_task(self):
        return self._task_id

    @property
    def tasks_for_mode(self):
        if self.mode == 'train':
            return self.train_tasks
        elif self.mode == 'test':
            return self.test_tasks
        else:
            raise ValueError(self._invalid_mode())

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

    def _invalid_mode(self):
        return f'Expected environment mode to be train or test, got {self.mode}'

    def compute_eval_metrics(self, data):
        infos, total_tasks = data
        success = []
        for task_idx in range(total_tasks):
            # Task success will be either 0 or 1
            task_success = 0
            for step_idx in range(task_idx, len(infos), total_tasks):
                task_info = infos[step_idx]
                task_success += int(task_info['success'])
                if task_success == 1:
                    break
            success.append(task_success)
        # infos = data
        # for info in infos:
        #     print(info['success'])
        # print(f'Success: {success}')
        return {
            'success_mean': np.array(success).mean(),
        }
