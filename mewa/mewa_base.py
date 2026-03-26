import abc
import os.path
from abc import ABC
from datetime import datetime

import gym
import numpy as np
from gym.utils import seeding

from mewa.mewa_utils.one_hot_encoding import OneHotEncoding
from mewa.mewa_utils.utils import create_logger


def set_up_logger(log_path, seed):
    if log_path is None:
        return None

    now = datetime.now()
    log_path_no_extension, extension = os.path.splitext(log_path)
    unique_log_path = f'{log_path_no_extension}_{seed}_{now.strftime("%Y_%m_%d_%H_%M_%S_%f")}{extension}'
    log_filename = os.path.splitext(os.path.basename(unique_log_path))[0]
    return create_logger(f'mewa.{log_filename}', unique_log_path)


class MEWA(gym.Env, ABC):
    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,
                 seed,
                 split_dict,
                 tasks,
                 verbose,
                 log_path,
                 action_shape,
                 observation_space):
        """
        The abstract base class of MEWA. This should be extended with concrete implementations of MEWA

        :param task_path: For narrow distributions: the path to a task file.
        For wide distributions: the path to a directory containing multiple task files
        :param wide_tasks: The number of (wide) tasks
        :param narrow_tasks: The number of task variations for each (wide) task (i.e. narrow tasks per wide task)
        :param complex_worker: Whether the worker can make mistakes or not
        :param seed: The random seed of the environment
        :param split_dict: Workers can be sampled from different regions of the distribution during meta-training and
        meta-testing. If None, use the entire distribution
        :param tasks: A set of tasks is sampled when this object is created. Use this parameter instead, if given
        :param verbose: The amount of information to print
        :param log_path: An optional path to a logging file
        :param observation_space: The types of observations used by this environment
        """

        super(MEWA, self).__init__()
        self.complex_worker = complex_worker
        self.split_dict = split_dict

        seed = seed if seed is not None and seed >= 0 else np.random.randint(0, 65536)

        # Set up the logger
        self.verbose = verbose
        self.logger = set_up_logger(log_path, seed)

        # Set up the RNG
        self._print(f'Creating environment with seed {seed}', log=True)
        self._np_random = None
        MEWA.reset(self, seed=seed)

        # FIXME The obs and act spaces should depend on the task description
        #  (especially important for wide distributions)
        # Create the state and action spaces
        self.action_space = OneHotEncoding(action_shape, seed=int(self._np_random.integers(65536)))
        self.observation_space = observation_space
        self.observation_space.seed(int(self._np_random.integers(65536)))

        # For each task, the reward function will be given in the task description
        self._reward_function = None

        # Keep track of the sub-goals completed
        self._progress = 0

        # Used for normalising rewards
        self._reward_normaliser = None

        # Randomly generate the required number of tasks. If a list of tasks has been given, use that instead
        self.tasks = self.sample_tasks(task_path, wide_tasks, narrow_tasks) if tasks is None else tasks

    @abc.abstractmethod
    def sample_tasks(self, task_path, wide_count, narrow_count):
        pass

    @abc.abstractmethod
    def reset_task(self, task_id):
        pass

    # Change the tasks to a list of tasks with uniform workers (starting from the strongest to the weakest worker)
    @abc.abstractmethod
    def create_uniform_workers(self, task_path, worker_count):
        pass

    def get_all_task_idx(self):
        return list(range(len(self.tasks)))

    def set_tasks(self, tasks):
        self.tasks = tasks
        for task in self.tasks:
            self._update_reward_normaliser(task)

    # def set_tasks(self, tasks):
    #     self.tasks = tasks
    #     if len(self.tasks) > 1:
    #         for task in self.tasks:
    #             self._update_reward_normaliser(task)

    # Reset the current task to the initial state
    def reset(self, seed=None, return_info=False, options=None):
        # # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        if seed is None and self._np_random is None:
            raise EnvironmentError("A seed (int) should be passed to the environment the first time reset() is called")

        _ = None if not return_info else (None, None)
        return _

    def get_progress(self):
        return self._progress

    def _get_reward(self, done):
        if done:
            return self._reward_function[-1]
        return self._reward_function[self._progress]

    def _update_reward_normaliser(self, task):
        min_return, max_return = self._get_min_max_return_normaliser(task)
        self._reward_normaliser[task['description']] = {
            'min_return': min_return,
            'max_return': max_return
        }

    def _get_min_max_return_normaliser(self, task):
        if self._reward_normaliser is None:
            self._reward_normaliser = {}

        worker_task = task['task']['worker_task']
        rewards = worker_task['rewards']

        # Compute the min and max returns of this task
        min_return = self.max_episode_steps * rewards[0]
        max_return = []
        for index in range(len(rewards)):
            min_steps = worker_task['blocks_per_struct']
            if index == 0:
                min_steps -= 1
            max_return.append(min_steps * rewards[index])
        max_return = np.sum(max_return)
        return min_return, max_return

    # Decode a potentially one-hot-encoded action into a single integer
    def _decode_action(self, action):
        POTENTIAL_ERROR = f'MEWA expects actions to be one-hot-encoded vectors ' \
                          f'or integers representing indexes. Got: {action}'

        from collections.abc import Iterable
        if isinstance(action, Iterable):
            # Decode the action
            zeros = np.count_nonzero(action == 0)
            ones = np.count_nonzero(action == 1)
            if zeros != len(action) - 1 or ones != 1:
                raise TypeError(POTENTIAL_ERROR)
            action_index = np.argmax(action)
        else:
            try:
                action_index = int(action)
            except TypeError as e:
                raise TypeError(POTENTIAL_ERROR) from e
        return action_index

    def _print(self, message, log=False, verbose=1):
        pass
        # if self.logger is not None and log:
        #     self.logger.info(message)
        # if self.verbose >= verbose:
        #     print(message)
