import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from laser.garage.torch._functions import global_device


def is_padding(state, next_state, rewards):
    # The previous state is either the last state in the trajectory or padding
    valid_prev_state = state[-1] == 1 or not torch.any(state)
    step = torch.cat((next_state, rewards), dim=-1)
    return valid_prev_state and not torch.any(step)


class DatasetStorage:
    def __init__(self,
                 num_processes,
                 dataset_size,  # number of tasks in the dataset (i.e. p)
                 meta_traj_per_task,  # number of meta-trajectories per task (i.e. q)
                 traj_per_meta_traj,  # number of trajectories per meta-trajectory (i.e. m)
                 max_trajectory_len,  # how long a trajectory can be at max (i.e. horizon H)
                 obs_dim, action_dim,
                 store_norm_rewards=False):
        """
        Store everything that is needed for the VAE update
        :param num_processes:
        """

        self.step_dim = obs_dim + action_dim + 1

        self.dataset_size = dataset_size  # number of tasks in the dataset (i.e. p)
        self.meta_traj_per_task = meta_traj_per_task  # number of meta-trajectories per task (i.e. q)
        self.traj_per_meta_traj = traj_per_meta_traj  # number of trajectories per meta-trajectory (i.e. m)
        self.max_traj_len = max_trajectory_len  # how long a trajectory can be at max (i.e. horizon H)
        self.num_processes = num_processes
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.store_norm_rewards = store_norm_rewards

        # TODO Keep track of traj (unpadded) lengths (e.g. for logging)
        self.state, self.actions, self.rewards, \
            self._done_meta_traj, self._task_index, self._meta_traj_index, self._meta_step_index, \
            self._lens, self.rewards_normalised = 9 * (None,)
        self._init_dataset()

    def _init_dataset(self):
        # buffers for collected meta-trajectories (stored on CPU)
        # Create the dataset D as 3 buffers: state, actions, rewards
        self.state = self._create_dataset_buffer(self.obs_dim)  # (p, q, d_obs, m*H)
        self.actions = self._create_dataset_buffer(self.action_dim)  # (p, q, d_act, m*H)
        self.rewards = self._create_dataset_buffer(1)  # (p, q, 1, m*H)

        self._done_meta_traj = torch.zeros(self.num_processes).long()

        self._task_index = None  # We will have p tasks in total. Keep track of the current task
        self._meta_traj_index = None  # Each task will contain q meta-trajectories. Keep track of the current meta-traj
        self._meta_step_index = torch.zeros(self.num_processes).long()  # Each meta-traj will contain m*H meta-steps

        self._lens = torch.zeros((self.dataset_size, self.meta_traj_per_task, self.traj_per_meta_traj))  # (p, q, m)

        if self.store_norm_rewards:
            self.rewards_normalised = self._create_dataset_buffer(1)  # (p, q, 1, m*H)

    def to_device(self, device):
        self.state = self.state.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self._done_meta_traj = self._done_meta_traj.to(device)
        self._meta_step_index = self._meta_step_index.to(device)
        self._lens = self._lens.to(device)
        if self.store_norm_rewards:
            self.rewards_normalised = self.rewards_normalised.to(device)

    def get_dataset(self, split=False):
        split_dataset = self.state, self.actions, self.rewards
        if split:
            return split_dataset
        return torch.cat(split_dataset, dim=-2)  # (p, q, d, m*H)

    # The (unpadded) number of tasks currently available
    def __len__(self):
        return self._task_index + self.num_processes if self._task_index is not None else 0

    # Return all the data we have for batch_size randomly selected tasks
    def get_batch(self, batch_size, split=False, replace=False, lens=False, to_device=None):
        to_device = global_device() if to_device is None else to_device
        actual_batch_size = min(len(self), batch_size)

        # Select batch_size many tasks, from the unpadded data
        task_indices = np.random.choice(range(len(self)), actual_batch_size, replace=replace)

        # If there are not enough tasks in the dataset, add padding to the first dimension to reach batch_size
        padding = 2 * len(self.state.shape) * [0]
        padding[-1] = max(0, batch_size - actual_batch_size)

        # Get the data from the selected tasks, with some potential padding
        state = F.pad(self.state[task_indices].clone(), padding).to(to_device)  # (batch_size, q, d_obs, m*H)
        actions = F.pad(self.actions[task_indices].clone(), padding).to(to_device)  # (batch_size, q, d_act, m*H)
        rewards = F.pad(self.rewards[task_indices].clone(), padding).to(to_device)  # (batch_size, q, 1, m*H)

        # Shuffle the meta-trajectories in each task
        num_meta_traj = state.shape[1]
        meta_traj_indices = np.random.choice(num_meta_traj, num_meta_traj, replace=False)
        state = state[:, meta_traj_indices]
        actions = actions[:, meta_traj_indices]
        rewards = rewards[:, meta_traj_indices]

        batch = state, actions, rewards
        if not split:
            batch = torch.cat(batch, dim=-2)  # (batch_size, q, d, m*H)

        if not lens:
            return batch

        padding = 2 * len(self._lens.shape) * [0]
        padding[-1] = max(0, batch_size - actual_batch_size)
        lens = F.pad(self._lens[task_indices].clone(), padding).to(to_device)  # (batch_size, q, m)
        lens = lens[:, meta_traj_indices]
        return batch, lens

    # Return the most recent meta-trajectory
    def get_working_meta_traj(self, split=False, lens=False, to_device=None):
        to_device = global_device() if to_device is None else to_device
        working_state = self.state[self._task_index:self._task_index+self.num_processes, self._meta_traj_index]\
            .clone().to(to_device)
        working_actions = self.actions[self._task_index:self._task_index+self.num_processes, self._meta_traj_index]\
            .clone().to(to_device)
        working_rewards = self.rewards[self._task_index:self._task_index+self.num_processes, self._meta_traj_index]\
            .clone().to(to_device)

        split_working_meta_traj = working_state, working_actions, working_rewards
        if split:
            working_meta_traj = split_working_meta_traj
        else:
            working_meta_traj = torch.cat(split_working_meta_traj, dim=-2)  # (num_processes, d, m*H)

        if not lens:
            return working_meta_traj
        working_lens = self._lens[self._task_index:self._task_index+self.num_processes, self._meta_traj_index]\
            .clone().to(to_device)
        return working_meta_traj, working_lens

    def get_working_rew_norm(self, to_device=None):
        if self.store_norm_rewards:
            return (
                self.rewards_normalised[self._task_index:self._task_index + self.num_processes, self._meta_traj_index]
                .clone().to(to_device))
        return None

    # We are given num_processes many steps (i.e. prev/next states, action, reward), each step from a different
    # task. If the meta-trajectory for the i-th task (i.e. process) is over, then the i-th step will be padding
    def insert_meta_step(self, state, actions, next_state, rewards, infos, meta_traj_done, rew_norm=None):
        # For all the meta-traj that are done (i.e. have m traj), make sure the given step is just padding
        # FIXME Do this without for loop
        for task in range(self.num_processes):
            assert not self._done_meta_traj[task] \
                   or is_padding(state[task], next_state[task], rewards[task])
        # Update the meta-trajs that are done
        self._done_meta_traj = torch.max(
            self._done_meta_traj,
            meta_traj_done if isinstance(meta_traj_done, torch.Tensor) else torch.from_numpy(meta_traj_done)
        )

        # FIXME Do this without for loop
        for proc in range(self.num_processes):
            if 'padding' not in infos[proc] or ('padding' in infos[proc] and not infos[proc]['padding']):
                self.state[self._task_index + proc, self._meta_traj_index, :, self._meta_step_index[proc]] \
                    = state[proc].clone()
                self.actions[self._task_index + proc, self._meta_traj_index, :, self._meta_step_index[proc]] \
                    = actions[proc].detach().clone()
                self.rewards[self._task_index + proc, self._meta_traj_index, :, self._meta_step_index[proc]] \
                    = rewards[proc].clone()

                traj_index = (self._meta_step_index[proc] - 1) // self.max_traj_len
                self._lens[self._task_index + proc, self._meta_traj_index, traj_index] += 1

                if self.store_norm_rewards and rew_norm is not None:
                    self.rewards_normalised[self._task_index + proc, self._meta_traj_index, :, self._meta_step_index[proc]] \
                        = rew_norm[proc].clone()

                self._meta_step_index[proc] += 1

    def insert_initial_state(self, initial_state, process=None):
        if process is None:
            self.state[self._task_index:self._task_index + self.num_processes, self._meta_traj_index, :, 0] \
                = initial_state.clone()

            self._meta_step_index += 1
            self._lens[self._task_index:self._task_index + self.num_processes, self._meta_traj_index, 0] += 1

        else:
            self.state[self._task_index + process, self._meta_traj_index, :, self._meta_step_index[process]] \
                = initial_state.clone()

            self._meta_step_index[process] += 1
            traj_index = (self._meta_step_index[process] - 1) // self.max_traj_len
            self._lens[self._task_index + process, self._meta_traj_index, traj_index] += 1

    # Start inserting data from a new task
    def new_task(self):
        # The first meta-traj and meta-step in each task
        self._meta_traj_index = None
        self._meta_step_index *= 0
        self._done_meta_traj *= 0

        # Update the task index
        if self._task_index is None:
            # First task
            self._task_index = 0
        elif self._task_index + 2*self.num_processes <= self.dataset_size:
            # Next task
            self._task_index = self._task_index + self.num_processes
        else:
            # FIXME Shifting by more than num_processes at each step might make data collection faster. However,
            #  we'll always have a relatively high number of padding tasks
            # The dataset is full. So, shift to the left, removing the first num_processes tasks and freeing the last
            # num_processes tasks (i.e. replace them with padding)
            self.state = self.state.roll(shifts=-self.num_processes, dims=0)
            self.state[-self.num_processes:] *= 0
            self.actions = self.actions.roll(shifts=-self.num_processes, dims=0)
            self.actions[-self.num_processes:] *= 0
            self.rewards = self.rewards.roll(shifts=-self.num_processes, dims=0)
            self.rewards[-self.num_processes:] *= 0
            self._lens = self._lens.roll(shifts=-self.num_processes, dims=0)
            self._lens[-self.num_processes:] *= 0
            if self.store_norm_rewards:
                self.rewards_normalised = self.rewards_normalised.roll(shifts=-self.num_processes, dims=0)
                self.rewards_normalised[-self.num_processes:] *= 0

            # Reset the task index to the last num_processes slots
            self._task_index = self.dataset_size - self.num_processes

    # Start inserting data from a new meta-trajectory
    def new_meta_traj(self):
        self._meta_traj_index = 0 if self._meta_traj_index is None else self._meta_traj_index + 1
        self._meta_step_index *= 0
        self._done_meta_traj *= 0

    def new_traj(self, indices):
        # FIXME Do this without for loop
        for index in indices:
            # Update the meta-step of this meta-trajectory to the start of the next trajectory
            traj_index = (self._meta_step_index[index] - 1) // self.max_traj_len
            self._meta_step_index[index] = self.max_traj_len * (traj_index + 1)

    def save(self):
        return (
            self.state.clone(),
            self.actions.clone(),
            self.rewards.clone(),
            self._done_meta_traj.clone(),
            self._task_index,
            self._meta_traj_index,
            self._meta_step_index.clone(),
            self._lens.clone(),
            self.rewards_normalised.clone() if self.store_norm_rewards else None,
        )

    def load(self, load):
        self.state, self.actions, self.rewards, self._done_meta_traj, \
            self._task_index, self._meta_traj_index, self._meta_step_index, self._lens, rewards_normalised = load

    def clear(self):
        self.state *= 0
        self.actions *= 0
        self.rewards *= 0
        self._done_meta_traj *= 0
        self._task_index = None
        self._meta_traj_index = None
        self._meta_step_index *= 0
        self._lens *= 0
        if self.store_norm_rewards:
            self.rewards_normalised *= 0

    def is_empty(self):
        return len(self) == 0 \
            and self.state.abs().sum() == 0 \
            and self.actions.abs().sum() == 0 \
            and self.rewards.abs().sum() == 0 \
            and self._done_meta_traj.abs().sum() == 0 \
            and (self._meta_traj_index is None or self._meta_traj_index == 0) \
            and self._meta_step_index.abs().sum() == 0 \
            and self._lens.abs().sum() == 0 \
            and (not self.store_norm_rewards or self.rewards_normalised.abs().sum() == 0)

    def _create_dataset_buffer(self, dim, total_tasks=None):
        total_tasks = self.dataset_size if total_tasks is None else total_tasks
        return torch.zeros((total_tasks,
                            self.meta_traj_per_task,
                            dim,
                            self.traj_per_meta_traj * self.max_traj_len))  # (p, q, dim, m*H)

    @property
    def meta_traj_index(self):
        return self._meta_traj_index

    @property
    def meta_step_index(self):
        return self._meta_step_index

    @property
    def lens(self):
        return self._lens


class DatasetStoragePrecollect(DatasetStorage):
    def __init__(self,
                 num_processes,
                 dataset_size,  # number of tasks in the dataset (i.e. p)
                 meta_traj_per_task,  # number of meta-trajectories per task (i.e. q)
                 traj_per_meta_traj,  # number of trajectories per meta-trajectory (i.e. m)
                 max_trajectory_len,  # how long a trajectory can be at max (i.e. horizon H)
                 obs_dim, action_dim,
                 dataset_path,  # path to starting dataset
                 starting_len,  # how many tasks from the starting dataset to keep
                 ):
        self.dataset_path = dataset_path
        self.starting_len = starting_len
        super(DatasetStoragePrecollect, self).__init__(num_processes,
                                                       dataset_size,
                                                       meta_traj_per_task,
                                                       traj_per_meta_traj,
                                                       max_trajectory_len,
                                                       obs_dim, action_dim,)

    def _init_dataset(self):
        assert self.starting_len == -1 or self.starting_len >= 0
        assert self.starting_len <= self.dataset_size

        self.dataset_path = os.path.join(self.dataset_path)
        data_file_type = os.path.splitext(os.path.basename(self.dataset_path))[-1]

        starting_size = self.starting_len if self.starting_len != -1 else None
        if data_file_type == '.npz':
            with np.load(self.dataset_path, allow_pickle=True, mmap_mode='r') as datafile:
                data = datafile['dataset']  # (p', q, dim, m*H)
                lens = datafile['lens']  # (p', q, m)

                meta_traj_per_task = self._get_meta_traj_per_task(data, starting_size)

                starting_dataset = torch.from_numpy(data[:starting_size, :meta_traj_per_task])  # (p', q, dim, m*H)
                starting_lens = torch.from_numpy(lens[:starting_size, :meta_traj_per_task])  # (p', q, dim, m*H)
                starting_size = starting_dataset.shape[0]  # p'
        elif data_file_type == '.pt':
            starting_dataset = torch.load(self.dataset_path)
            starting_lens = starting_dataset['lens']  # (p', q, m)
            starting_dataset = starting_dataset['dataset']  # (p', q, dim, m*H)

            meta_traj_per_task = self._get_meta_traj_per_task(starting_dataset, starting_size)

            starting_dataset = starting_dataset[:starting_size, :meta_traj_per_task]  # (p', q, dim, m*H)
            starting_lens = starting_lens[:starting_size, :meta_traj_per_task]  # (p', q, dim, m*H)
            starting_size = starting_dataset.shape[0]  # p'
        else:
            raise ValueError(f'Expected the dataset to be a .npz or .pt file. Got {data_file_type} instead')

        assert starting_size <= self.dataset_size, \
            f'The given dataset contains {starting_size} tasks, which is more than the max {self.dataset_size}. ' \
            'Set a lower number of tasks to load'
        assert starting_dataset.shape[0] == starting_lens.shape[0]
        assert starting_dataset.shape[1] == starting_lens.shape[1] == self.meta_traj_per_task
        assert starting_dataset.shape[2] == self.step_dim
        assert starting_dataset.shape[3] == self.traj_per_meta_traj * self.max_traj_len

        # Pad the dataset
        padding_tasks = self.dataset_size - starting_size  # p-p'
        dataset = torch.cat((
            starting_dataset,
            torch.zeros((padding_tasks,) + tuple(starting_dataset.shape[1:]))),  # (p-p', q, dim, m*H)
            dim=0)  # (p, q, dim, m*H)

        self.state = dataset[:, :, :self.obs_dim]  # (p, q, d_obs, m*H)
        self.actions = dataset[:, :, self.obs_dim:self.obs_dim+self.action_dim]  # (p, q, d_act, m*H)
        self.rewards = dataset[:, :, -1:]  # (p, q, 1, m*H)

        self._done_meta_traj = torch.zeros(self.num_processes).long()

        if starting_size == 0:
            self._task_index = None  # Start counting from the first task
        else:
            self._task_index = starting_size - self.num_processes  # Start counting from the latest task
            if self._task_index + 2*self.num_processes > self.dataset_size:
                self._task_index = self.dataset_size - self.num_processes
        self._meta_traj_index = None  # Each task will contain q meta-trajectories. Keep track of the current meta-traj
        self._meta_step_index = torch.zeros(self.num_processes).long()  # Each meta-traj will contain m*H meta-steps

        self._lens = torch.cat((
            starting_lens,
            torch.zeros((padding_tasks,) + tuple(starting_lens.shape[1:]))),  # (p-p', q, m)
            dim=0)  # (p, q, m)

    def _get_meta_traj_per_task(self, data, starting_size):
        assert starting_size is None or starting_size <= data.shape[0]
        assert self.meta_traj_per_task <= data.shape[1]

        if self.meta_traj_per_task < data.shape[1]:
            warnings.warn(f'The given dataset contains {data.shape[1]} trajectories per task, '
                          f'but we are only using the first {self.meta_traj_per_task}')
            return self.meta_traj_per_task
        return None
