from typing import Iterable

import gymnasium as gym
import numpy as np


class Hopper(gym.Env):
    def __init__(self, mode, h, seed):
        self._env = env = gym.make("Hopper-v5", max_episode_steps=h, terminate_when_unhealthy=True)

        self._env = gym.wrappers.ClipAction(self._env)

        self.original_mass = np.copy(self._env.unwrapped.model.body_mass)
        self.original_inertia = np.copy(self._env.unwrapped.model.body_inertia)
        self.original_friction = np.copy(self._env.unwrapped.model.geom_friction)
        self.original_damping = np.copy(self._env.unwrapped.model.dof_damping)

        self.mode = mode
        if self.mode == 'train':
            self.log_scale_limit = (-1, 1)  # [1.5^-1, 1.5^1]
            self.neg = False
            self.train_tasks = [self._create_task() for _ in range(50000)]
            self.test_tasks = None
        elif self.mode == 'test':
            self.log_scale_limit = (1, 1.5)  # [1.5^-1.5, 1.5^-1] U [1.5^1, 1.5^1.5]
            self.neg = True
            self.train_tasks = None
            self.test_tasks = [self._create_task() for _ in range(50000)]
        else:
            raise ValueError(self._invalid_mode())

        self.scale = {}

        self.reset_task()

        self._max_episode_steps = env.spec.max_episode_steps
        self.max_episode_steps = env.spec.max_episode_steps

    def step(self, action):
        if self.is_padding_action(action):
            obs = np.zeros(self.observation_space.shape)
            info = {'task': self._task, 'task_id': self._task_id, 'reward_raw': 0, 'padding': True}
            return obs, 0, False, False, info

        obs, reward, done, truncate, info = self._env.step(action)

        info.update({'task': self._task, 'task_id': self._task_id, 'reward_raw': reward, 'padding': False})
        done = done or truncate

        return obs, reward, done, truncate, info

    def reset(self, seed=None, return_info=False, options=None):
        obs, info = self._env.reset()
        return obs

    def reset_task(self, task_id=None):
        mass = np.copy(self.original_mass)
        inertia = np.copy(self.original_inertia)
        friction = np.copy(self.original_friction)
        damping = np.copy(self.original_damping)

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

        mass *= self._task['mass_multiplier']
        inertia *= self._task['inertia_multiplier']
        friction *= self._task['friction_multiplier']
        damping *= self._task['damping_multiplier']

        self._env.unwrapped.model.body_mass = mass
        self._env.unwrapped.model.body_inertia = inertia
        self._env.unwrapped.model.geom_friction = friction
        self._env.unwrapped.model.dof_damping = damping

        return self._task_id

    def _create_task(self):
        mass_multiplier = self._compute_multiplier(self.original_mass.shape)
        inertia_multiplier = self._compute_multiplier(self.original_inertia.shape)
        friction_multiplier = self._compute_multiplier(self.original_friction.shape)
        damping_multiplier = self._compute_multiplier(self.original_damping.shape)

        return {
            'mass_multiplier': mass_multiplier,
            'inertia_multiplier': inertia_multiplier,
            'friction_multiplier': friction_multiplier,
            'damping_multiplier': damping_multiplier,
        }

    def _compute_multiplier(self, shape):
        exponent = np.random.uniform(self.log_scale_limit[0], self.log_scale_limit[1], size=shape)
        exponent_sign = np.random.choice([-1, 1], size=shape) if self.neg else 1
        exponent = exponent_sign * exponent
        return np.array(1.5) ** exponent

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def _invalid_mode(self):
        return f'Expected environment mode to be train or test, got {self.mode}'

    def is_padding_action(self, action):
        # FIXME
        return np.all(action < -1000)
        # return np.all(action == np.full(self.action_space.shape[0], -1234.5678, dtype=float))
