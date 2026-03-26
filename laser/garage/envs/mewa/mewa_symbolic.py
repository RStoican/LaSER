import numpy as np
from laser.garage.envs.env_utils.safe_random import SafeRandom
from laser.garage.envs.one_hot_encoding import OneHotEncoding
from gymnasium.spaces import Box

from mewa.mewa_symbolic import MEWASymbolic as MEWASymbolicOriginal


class MEWASymbolic(MEWASymbolicOriginal):
    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,

                 seed=None,
                 split_dict=None,
                 tasks=None,
                 h=50,
                 uniform_human=False,

                 verbose=0,
                 log_path=None):
        self._uniform_human = uniform_human
        super().__init__(task_path, wide_tasks, narrow_tasks, complex_worker, seed, split_dict, tasks,
                         h, verbose, log_path)
        self._task_id = self.reset_task()
        self._max_episode_steps = self.max_episode_steps

        # Update to gymnasium
        self.observation_space = Box(low=self.observation_space.low, high=self.observation_space.high,
                                     shape=self.observation_space.shape, dtype=np.double)
        self.action_space = OneHotEncoding(4)

    @property
    def tester_type(self):
        from laser.garage.envs.mewa.mewa_curated import MEWACuratedTester
        return MEWACuratedTester

    def step(self, action):
        if self.is_padding_action(action):
            obs = np.zeros(self.observation_space.shape)
            info = {'task': self._task['worker_personality'], 'task_id': self._task_id, 'padding': True,
                    'reward_raw': 0}
            return obs, 0, 0, info

        obs, reward, done, info = super().step(action)
        info.update({'task': self._task['worker_personality'], 'task_id': self._task_id, 'padding': False})
        return obs, reward, done, info

    def reset_task(self, task_id=None):
        """
        :param task_id: The ID of the task to reset the environment to. If None or not given, set the environment to a
        randomly chosen task
        :return: task_id

        The original MEWA reset_task() method also resets the environment (i.e. to the start of the task). However,
        LaSER expects reset_task to only reset the task
        """
        task_id = self._np_random.randint(len(self.tasks)) if task_id is None else task_id
        self._task_id = task_id
        self._print(f'Resetting task to {task_id}\n'
                    f'     Description: {self.tasks[task_id]["description"]}\n'
                    f'     Human: {self.tasks[task_id]["worker_personality"]}', log=True)
        self._task = self.tasks[task_id]
        self._reward_function = self._task['task']['worker_task']['rewards']
        return self._task_id

    def set_task(self, task):
        self._task = task
        self._reward_function = self._task['task']['worker_task']['rewards']

    def sample_tasks(self, task_path, wide_count, narrow_count):
        # Reset the random number generator to a thread-safe one
        self._np_random = SafeRandom()
        if not self._uniform_human:
            return super().sample_tasks(task_path, wide_count, narrow_count)
        return self._sample_tasks_uniform(task_path, wide_count, narrow_count)

    def _sample_tasks_uniform(self, task_path, wide_count, narrow_count):
        import os.path
        from os import listdir
        from os.path import isfile, join

        self._split_index = 0

        # Sample the task descriptions
        if os.path.isdir(task_path):
            all_descriptions = [task_file for task_file in listdir(task_path) if isfile(join(task_path, task_file))]
            if wide_count == -1:
                descriptions = all_descriptions
            else:
                # wide_count = min(wide_count, len(all_descriptions))
                descriptions = self._np_random.choice(all_descriptions, size=wide_count, replace=False)
        else:
            if wide_count > 1:
                raise ValueError(f'Trying to build a wide distribution of {wide_count} tasks, but only a single yaml '
                                 f'file was given. Give a directory instead')
            descriptions = ['']

        # Create narrow_count tasks for each task description
        tasks = []
        for description in descriptions:
            # Build the task object. This will be used to generate the worker personality
            description_path = os.path.join(task_path, description) if description != '' else task_path
            task = self._load_task(description_path)

            if self.complex_worker is not None:
                task['worker_task']['complex_worker'] = self.complex_worker

            for _ in range(narrow_count):
                # If we don't split the distribution into multiple regions,
                # then just sample a human from the entire distribution
                if self.split_dict is None or len(self.split_dict) == 0:
                    worker_personality = self._sample_human_uniform(task['worker_task'])

                tasks.append({"description": description_path, "worker_personality": worker_personality, "task": task})

            # For each task (NOT each worker variation), update the values used to normalise rewards.
            self._update_reward_normaliser(tasks[-1])
        return tasks

    # Create a random personality, to be used by a worker. For each possible mistake, generate the mean and std for a
    # specific worker. Mean and std are generated using the Gaussian distributions in the task description
    def _sample_human_uniform(self, worker_task):
        if not self.complex_worker:
            return None

        # For each possible mistake, generate the mean for this specific worker
        prev_max = 1.0
        mistake_gaussians = worker_task['mistake_gaussians']
        personality = []
        for index in range(len(mistake_gaussians)):
            mean = self._np_random.uniform(0.0, prev_max)
            prev_max = mean
            personality.append((mean, 0))
        return personality

    def get_task(self):
        return self._task_id

    def is_padding_action(self, action):
        return self.action_space.is_padding(action)

    def compute_eval_metrics(self, data):
        return None
