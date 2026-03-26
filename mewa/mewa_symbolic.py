import copy
import os.path
from abc import ABC
from os import listdir
from os.path import isfile, join

import numpy as np
from gym import spaces

from mewa.mewa_base import MEWA
from mewa.mewa_utils.utils import color_to_one_hot, load_task


class MEWASymbolic(MEWA, ABC):
    TASK_DIR = '/workspace/tasks/'

    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,

                 seed=None,
                 split_dict=None,
                 tasks=None,
                 max_episode_steps=100,

                 verbose=0,
                 log_path=None):
        """
        A symoblic version of MEWA, with a discrete, low-dimensional state space. For more details, see mewa_base

        :param max_episode_steps: The max number of steps before the tasks fails
        """

        self.complex_worker = complex_worker
        self.max_episode_steps = max_episode_steps

        if wide_tasks is None or 0 <= wide_tasks <= 1:
            task = self._load_task(task_path)
            total_actions = len(task['blocks'])

            # Agent obs + worker obs + supervisor obs
            shape = total_actions + 2*total_actions + total_actions + 2
            observation_space = spaces.Box(low=0, high=20, shape=(shape,), dtype=np.double)
        else:
            raise NotImplementedError

        super(MEWASymbolic, self).__init__(
            task_path=task_path,
            wide_tasks=wide_tasks,
            narrow_tasks=narrow_tasks,
            complex_worker=complex_worker,
            seed=seed,
            split_dict=split_dict,
            tasks=tasks,
            action_shape=total_actions,
            # FIXME The obs space should depend on the task description (especially important for wide distributions)
            observation_space=observation_space,
            verbose=verbose,
            log_path=log_path
        )

        # The current state of the env
        self._obs = None

        # The current task
        self._task = None

        self._split_index = 0
        self._step_count = 0

    def reset_task(self, task_id=None):
        task_id = self._np_random.randint(len(self.tasks)) if task_id is None else task_id
        self._print(f'Resetting task to {task_id}\n'
                    f'     Description: {self.tasks[task_id]["description"]}\n'
                    f'     Human: {self.tasks[task_id]["worker_personality"]}', log=True)
        self._task = self.tasks[task_id]
        self._reward_function = self._task['task']['worker_task']['rewards']
        self._progress = 0
        self._step_count = 0

    def reset(self, seed=None, return_info=False, options=None):
        self._print('reset()', log=True, verbose=3)
        super(MEWASymbolic, self).reset(seed, return_info, options)
        self._obs = self._create_initial_obs()
        self._progress = 0
        self._step_count = 0
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self._take_step(action)
        info['reward_raw'] = reward

        self._step_count += 1
        if self._step_count >= self.max_episode_steps:
            done = True

        # For wide task distributions, normalise the reward
        if self._reward_normaliser is not None:
            task = self._task['description']
            try:
                normaliser = self._reward_normaliser[task]['max_return'] - self._reward_normaliser[task]['min_return']
            except KeyError:
                min_return, max_return = self._get_min_max_return_normaliser(self._task)
                normaliser = max_return - min_return
            reward /= normaliser
            if done:
                try:
                    reward -= self._reward_normaliser[task]['min_return'] / normaliser
                except KeyError:
                    min_return, max_return = self._get_min_max_return_normaliser(self._task)
                    reward -= min_return / normaliser

        return obs, reward, done, info

    def get_task(self):
        return self._task

    def sample_tasks(self, task_path, wide_count, narrow_count):
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
                    worker_personality = self._sample_human_simple(task['worker_task'])
                else:
                    # Else, randomly choose one of the sampling methods provided
                    if self._np_random.random() < 0.5:
                        worker_personality = self._sample_human_default(task['worker_task'])
                    else:
                        worker_personality = self._sample_human_balanced(task['worker_task'], use_inc_dist=True)

                tasks.append({"description": description_path, "worker_personality": worker_personality, "task": task})

            # For each task (NOT each worker variation), update the values used to normalise rewards.
            self._update_reward_normaliser(tasks[-1])
            # if len(descriptions) > 1:
            #     self._update_reward_normaliser(tasks[-1])
        return tasks

    # Change the tasks to a list of tasks with uniform workers (starting from the strongest to the weakest worker)
    def create_uniform_workers(self, task_path, worker_count):
        pass

    def _take_step(self, action):
        # The action is the index of the chosen block, either one-hot encoded or not. In the former case, decode it
        color_index = self._decode_action(action)
        action_label = self._task['task']['block_colors'][color_index]

        next_obs, done, info = self._do_state_transition(color_index)

        self._print(f'   Turn {self._step_count}: {action_label}' + info, log=True, verbose=3)
        return next_obs, self._get_reward(done), done, {}

    def _get_obs(self):
        return copy.deepcopy(self._obs)

    def _do_state_transition(self, color_index):
        if self._obs[color_index] == 0:
            # Not a valid action
            return self._get_obs(), False, ''

        # There are 3 parts of the environment state "acted" upon: the agent's, the structures' and the goals' states
        self._do_agent_action(color_index)
        complete_structure_mistake, info = self._do_struct_action(color_index)
        self._do_goal_action(color_index, complete_structure_mistake)

        # Return the observation (state) built based on the 3 parts above
        return self._get_obs(), self._is_game_done(), info

    def _do_agent_action(self, color_index):
        if self._obs[color_index] != 0:
            self._obs[color_index] -= 1

    def _do_struct_action(self, color_index):
        struct_index = len(self._task['task']['blocks']) + 2 * color_index
        self._obs[struct_index] += 1

        action_info = self._task['task']['block_colors'][color_index]

        # Check if a mistake happened
        is_mistake, info = self._is_mistake(struct_index)
        if is_mistake:
            info += f' | Making mistake when working on the {action_info} structure'
            self._obs[struct_index + 1] = 1
        else:
            if self._obs[struct_index] == 1:
                info += f' | Starting {action_info} structure'

        # If the current action completes the structure, update the state
        complete_structure_mistake = None
        if self._obs[struct_index] == self._task['task']['worker_task']['blocks_per_struct']:
            # Keep track of whether the complete structure has a mistake in it or not
            complete_structure_mistake = (self._obs[struct_index + 1] == 1)

            # Remove the mistake blocks from the play area and return them to the RL agent
            self._obs[struct_index] = 0
            self._obs[struct_index + 1] = 0
            if complete_structure_mistake:
                info += f' | Resetting {action_info} structure'
                self._obs[color_index] += self._task['task']['worker_task']['blocks_per_struct']
            else:
                info += f' | Finishing {action_info} structure'
        return complete_structure_mistake, info

    def _do_goal_action(self, color_index, complete_structure_mistake):
        if (complete_structure_mistake is not None) and (not complete_structure_mistake):
            # Add the complete structure to the goal and build the next stage of the goal, if available
            total_actions = len(self._task['task']['blocks'])
            padding = total_actions + 2*total_actions
            self._obs[padding + color_index] = 1
            self._build_goal()

    def _is_mistake(self, struct_index):
        # A mistake can happen when a complex worker adds the second block to a structure
        if self.complex_worker and self._obs[struct_index] == 2:
            # The mistake type depends on how many types of colours (i.e. structures) the worker has
            total_actions = len(self._task['task']['blocks'])
            padding = total_actions + 2 * total_actions
            mistake_type = (self._obs[total_actions:padding:2] > 0).sum() - 1

            # FIXME: Add the option for Gaussian sampling
            # Get the mistake probability and make sure 0 <= mistake_probability <= 1 holds
            mistake_probability = self._task['worker_personality'][mistake_type][0]
            mistake_probability = min(max(mistake_probability, 0), 1)

            info = f' | Probability mistake type {mistake_type}: {round(100 * mistake_probability, 2)}%'
            return self._np_random.random() < mistake_probability, info
        return False, ''

    def _is_game_done(self):
        total_actions = len(self._task['task']['blocks'])
        padding = total_actions + 2 * total_actions
        structures_done = self._obs[padding:padding+total_actions].sum() == self._task['task']['supervisor_task']['struct_count']
        goal_done = self._progress == self._task['task']['supervisor_task']['subgoals_count']
        return structures_done and goal_done

    def _build_goal(self):
        task = self._task['task']
        if self._progress >= task['supervisor_task']['subgoals_count']:
            return

        total_actions = len(self._task['task']['blocks'])
        padding = total_actions + 2 * total_actions

        # The requirements are given in the task and depend on the supervisor's current progress
        requirement = task['supervisor_task']['subgoals_requirements'][self._progress][0]
        requirement = color_to_one_hot(task['block_colors'], requirement)

        req_color_index = np.where(requirement == 1)[0][0]
        req_achieved = self._obs[padding + req_color_index] == 1
        if req_achieved:
            # Use some of the blocks to build the next stage of the goal
            subgoal = task['supervisor_task']['subgoals'][self._progress]
            self._obs[-2] -= len(subgoal)
            self._obs[-1] += len(subgoal)
            self._progress += 1

            # The supervisor has made some progress, so check whether the next requirement hasn't already been achieved
            self._build_goal()

    def _create_initial_obs(self):
        task = self._task['task']

        # The initial state of the RL agent's blocks
        #    obs = [#orange, #blue, #green, #red]
        agent_obs = copy.deepcopy(task['blocks'])  # orange, blue, green, red

        # The initial obs of incomplete structures is an array of 0s, with one element for each colour and one for each
        # mistake
        #    obs = #actions * [#blocks/color, color_mistake]
        total_actions = len(task['blocks'])
        struct_obs = 2 * total_actions * [0]

        # The initial goal obs will contain a 0 for each possible sub-structure, the initial number of blocks required
        # to complete each sub-goal and a 0 for how many of those blocks have been used
        #    obs = #actions * [complete_substr/color] + [#supervisor blocks, #used supervisor blocks]
        goal_obs = total_actions * [0] + [task['supervisor_blocks'], 0]

        return np.concatenate([agent_obs, struct_obs, goal_obs]).astype(np.double)

    def _load_task(self, description_path):
        self._print(f"Loading task {description_path}", log=True, verbose=1)
        return load_task(description_path)

    """
    The default sampling method. Sample the human "personality" from different regions of the distribution, depending on
    whether this task is used for training or testing
    """
    def _sample_human_default(self, worker_task):
        VERBOSE_LEVEL = 2

        # Sample the interval that will be used for this human, depending on whether this task will be used for
        # training or testing
        if self._split_index < self.split_dict["train_count"]:
            mean_interval_index, std_interval_index = self._create_valid_distribution(self.split_dict["split_train"])
        else:
            mean_interval_index, std_interval_index = self._create_valid_distribution(self.split_dict["split_eval"])
        self._split_index += 1

        info = 'Creating worker personality:'
        personality = []
        for index in range(len(worker_task['mistake_gaussians'])):
            mistake_gauss = worker_task['mistake_gaussians'][index]

            # Split the distribution for this mistake into several equal parts
            mean_intervals = self._split_interval(mistake_gauss[0], self.split_dict["split"][0])
            std_intervals = self._split_interval(mistake_gauss[1], self.split_dict["split"][1])

            # Sample the mean and std
            mean = self._np_random.uniform(mean_intervals[mean_interval_index][0],
                                           mean_intervals[mean_interval_index][1])
            std = self._np_random.uniform(std_intervals[std_interval_index][0],
                                          std_intervals[std_interval_index][1])

            personality.append((mean, std))
            if index > 0:
                info += ','
            info += f' {(mean, std)}'
        self._print(info, log=True, verbose=VERBOSE_LEVEL)
        return personality

    # Choose the region of the distribution from which to sample the current human behaviour
    def _create_valid_distribution(self, available_intervals):
        mean_index = self._np_random.integers(0, len(available_intervals[0]))
        mean_interval_index = available_intervals[0][mean_index]

        std_index = self._np_random.integers(0, len(available_intervals[1]))
        std_interval_index = available_intervals[1][std_index]

        return mean_interval_index, std_interval_index

    """
    When the testing tasks are sampled from different regions of the distribution than training tasks, it is very 
    likely that the testing distribution is not split into 2 equal distributions (i.e. because the training region 
    is not exactly in the centre of the whole distribution). To ensure we have equal samples from all the regions of 
    the distribution, first sample one of the two regions with equal probability, the sample the task from it
    
    Also, if use_inc_dist, then ensure that adding a new block ALWAYS reduces the probability of making a mistake. 
    If not use_inc_dist, then we can have a case where e.g. the mistake probability of type 0 is sampled from the left 
    side of the distribution and the one for type 1 is sampled from the right side, i.e. adding an additional block 
    greatly INCREASES the mistake probability
    
    The disadvantage of this sampling method is that good humans will generally be overall good, while bad humans will 
    be generally overall bad. We don't recommend using only this method for sampling tasks. See the sample_tasks() 
    method for a combination of multiple sampling approaches
    """
    def _sample_human_balanced(self, worker_task, use_inc_dist):
        VERBOSE_LEVEL = 2
        info = 'Creating worker personality:'

        personality = []
        previous_mistake_prob = np.inf
        for index in range(len(worker_task['mistake_gaussians'])):
            # Get the mean and std of the average worker
            mistake_gauss = worker_task['mistake_gaussians'][index]
            avg_worker_mean, avg_worker_std = mistake_gauss[0]
            std_gaussian = mistake_gauss[1]

            # Create the regions of the distribution used for sampling training or testing tasks
            if self._split_index < self.split_dict["train_count"]:
                interval_limits = (
                    (avg_worker_mean - 2 * avg_worker_std, avg_worker_mean - 0.5 * avg_worker_std),
                    (avg_worker_mean + 0.5 * avg_worker_std, avg_worker_mean + 2 * avg_worker_std)
                )
            else:
                interval_limits = (
                    (0, avg_worker_mean - 2 * avg_worker_std),
                    (avg_worker_mean + 2 * avg_worker_std, 1)
                )

            # Sample an interval, then sample the worker personality mean for this mistake type from the interval
            interval = self._create_valid_interval(interval_limits, previous_mistake_prob, use_inc_dist)
            mean = self._np_random.uniform(interval[0], interval[1])
            std = self._np_random.normal(loc=std_gaussian[0], scale=std_gaussian[1])

            personality.append((mean, std))
            previous_mistake_prob = mean

            if index > 0:
                info += ','
            info += f' {(mean, std)}'

        self._split_index += 1
        self._print(info, log=True, verbose=VERBOSE_LEVEL)
        return personality

    # Create a valid interval from which to sample the current human behaviour
    def _create_valid_interval(self, interval_limits, previous_mistake_prob, use_inc_dist):
        # Depending on the previous mistake probability, create 1 or 2 intervals
        if use_inc_dist:
            # FIXME Use with increased_split
            if previous_mistake_prob > interval_limits[1][0]:
                intervals = (
                    interval_limits[0],
                    (interval_limits[1][0], min(interval_limits[1][1], previous_mistake_prob))
                )
            else:
                intervals = (
                    (interval_limits[0][0], min(interval_limits[0][1], previous_mistake_prob)),
                    None
                )
        else:
            # FIXME Use with no_order_constraints
            intervals = interval_limits

        sampled_interval = intervals[self._np_random.integers(0, 2)]
        if sampled_interval is None:
            sampled_interval = intervals[0]
        if sampled_interval[1] - sampled_interval[0] < 0:
            sampled_interval = (sampled_interval[1], sampled_interval[0])
        return sampled_interval

    # Create a random personality, to be used by a worker. For each possible mistake, generate the mean and std for a
    # specific worker. Mean and std are generated using the Gaussian distributions in the task description
    def _sample_human_simple(self, worker_task):
        VERBOSE_LEVEL = 2
        info = 'Creating worker personality:'
        if not self.complex_worker:
            self._print("   Not a complex worker; personality not needed", log=True, verbose=2)
            return None

        # For each possible mistake, generate the mean and std for this specific worker. Mean and std are generated
        # using the Gaussian distributions in the task description
        mistake_gaussians = worker_task['mistake_gaussians']
        personality = []
        for index in range(len(mistake_gaussians)):
            mistake_gaussian = mistake_gaussians[index]

            # The gaussian for sampling the worker's mean and std
            mean_gaussian = mistake_gaussian[0]
            std_gaussian = mistake_gaussian[1]

            mean = self._np_random.normal(loc=mean_gaussian[0], scale=mean_gaussian[1])
            std = self._np_random.normal(loc=std_gaussian[0], scale=std_gaussian[1])

            personality.append((mean, std))
            if index > 0:
                info += ','
            info += f' {(mean, std)}'
        self._print(info, log=True, verbose=VERBOSE_LEVEL)
        return personality

    def _split_interval(self, gauss, split):
        # Simulate an interval of samples drawn from a Gaussian
        mean, std = gauss
        split_points = np.array([mean - 3 * std, mean - 2 * std, mean - 0.5 * std,
                                 mean + 0.5 * std, mean + 2 * std, mean + 3 * std])
        if split != 5 and split != 1:
            raise ValueError(f'Split should be 5 for mean and 1 for std. Got: {split}')

        # Split the interval into #split intervals of equal length
        intervals = np.zeros((split, 2))
        intervals[:, 0] = split_points[:split]
        intervals[:, 1] = np.roll(split_points, -1)[:split]
        return intervals
