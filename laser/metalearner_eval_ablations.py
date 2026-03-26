import copy

import numpy as np
import torch

from laser.garage.torch._functions import global_device
from laser.garage.utils import helpers as utl
from laser.garage.utils import metalearner_helpers as mlutl
from metalearner_eval import EvalMetaLearner


class EvalMetaLearnerAblations(EvalMetaLearner):
    def _create_models(self):
        # The dimension of the task context used by the task policy
        if not self.args.ablation_true_task:
            if self.args.context_type == 'full':
                task_context_dim = self.args.latent_dim * self.args.horizon * self.args.traj_per_meta_traj  # H*d_z*m
            elif self.args.context_type == 'traj_latest':
                raise NotImplementedError  # d_z*m
            elif self.args.context_type == 'latest':
                task_context_dim = self.args.latent_dim  # d_z
            else:
                raise ValueError(f'Unknown context type {self.args.context_type}')
        else:
            # FIXME Replace 4 with an argument
            task_context_dim = 4
            self.args.policy_task_context_embedding_dim = 4
            self.args.policy_context_hidden_layers = None

        metalearner_initialiser = mlutl.MetaLearnerInitialiser(args=self.args,
                                                               envs=self.envs,
                                                               exploration_train_mode=False,
                                                               get_iter_idx=lambda: self.iter_idx,
                                                               logger=self.logger,
                                                               task_context_dim=task_context_dim, )

        if self.args.norm_rew_task:
            # FIXME Load ret_rms
            pass

        # Transformer and policies
        self.transformer = metalearner_initialiser.initialise_autoencoder()
        self.exploration_policy = metalearner_initialiser.initialise_policy(policy_type='exploration') \
            if self.args.ablation_use_context and not self.args.ablation_true_task else None
        if not self.args.ablation_use_history:
            self.args.pass_latent_to_policy = False
        self.task_policy = metalearner_initialiser.initialise_policy(policy_type='task')
        self.shared_latent = metalearner_initialiser.initialise_shared_latent()

        if self.args.ablation_true_task:
            # The exploration policy and encoder are not required
            mlutl.load_models(self.args.save_path, self.args.model_label, None, None,
                              None, self.task_policy,
                              task_policy_name=self.args.task_policy_name,
                              norm_state_exploration=self.args.norm_state_exploration)
        else:
            mlutl.load_models(self.args.save_path, self.args.model_label, self.transformer, self.exploration_policy,
                              self.shared_latent, self.task_policy,
                              task_policy_name=self.args.task_policy_name,
                              norm_state_exploration=self.args.norm_state_exploration)

    def _explore(self, exploration_envs, task_envs):
        assert self.transformer.dataset_storage.is_empty()

        # for each process, we log the returns during the first, second, ... episode (such that we have a minimum of
        #   [num_episodes]);
        # the first traj_per_meta_traj are for exploration trajs
        # the second to last column is for the task solving traj
        # the last column is for any overflow and will be discarded at the end, because we need to wait
        #   until all processes have at least [num_episodes] many episodes)
        exploration_returns_per_episode = torch.zeros((self.args.num_processes, self.args.traj_per_meta_traj + 1)) \
            .to(global_device())  # (num_proc, m+1)
        task_returns_per_episode = torch.zeros((self.args.num_processes, self.args.traj_per_meta_traj + 1)) \
            .to(global_device())  # (num_proc, m+1)
        # FIXME Assumes num_processes == 1. Change to use the index in the transformer storage instead
        meta_traj_index = 0

        # Exploit task without any latent task information
        self.transformer.dataset_storage.new_task()
        self.transformer.dataset_storage.new_meta_traj()
        task_returns_per_episode[range(self.args.num_processes), 0] = self._exploit(task_envs,
                                                                                    None,
                                                                                    None,
                                                                                    idx=0)

        self.transformer.dataset_storage.new_task()
        self.transformer.dataset_storage.new_meta_traj()

        # Reset the environment to new tasks (i.e. the only task currently available in the env)
        utl.reset_env(exploration_envs, self.args)
        exploration_tasks, exploration_task_ids = None, None

        # Start collecting data
        # FIXME When not using the context, it will return 0 for all episodes besides first and last.
        #  Fix this (0 can be seen as a very high return in MEWA)
        if self.args.ablation_use_context and not self.args.ablation_true_task:
            prev_state = utl.reset_meta_traj(exploration_envs)  # (num_processes, d_obs)
            if len(prev_state.shape) == 1:
                prev_state = prev_state.unsqueeze(0)
            self.transformer.dataset_storage.insert_initial_state(prev_state)

            # The latent prior
            traj_latent = self._encode_meta_traj(is_exploration_policy=True,
                                                 return_prior=True, starting_state=prev_state)  # (num_processes, d_z)

            meta_traj_done_flag = np.zeros(self.args.num_processes, dtype=int)

            self._print(f'{50 * "="} NEW EXPLORATION TRAJECTORY {50 * "="}')
            while meta_traj_done_flag.sum() != self.args.num_processes:
                # sample actions from the exploration policy (using the latent traj)
                _, action = self._select_action(self.exploration_policy,
                                                prev_state,
                                                traj_latent,
                                                meta_traj_done_flag,
                                                )  # (num_processes, 1), (num_processes, action_dim)

                # Take step in the environment
                #   meta_episode_done is true only when the last traj of the meta-traj ends (default 10 trajs)
                #   next_state will contain the episode_done flag appended at the end of the next_state features
                # If the current trajectory is finished, env_step will automatically reset the environment
                # (but NOT the task) and return:
                #   the last state of the previous traj in next_state
                #   the first state of the new traj in info['start_state']
                # FIXME Look into normalising obs, acts, rews
                next_state, (rew_raw, rew_normalised), meta_episode_done, infos \
                    = utl.env_step(exploration_envs, action, self.args)

                if len(next_state.shape) == 1:
                    next_state = next_state.unsqueeze(0)
                    rew_raw = rew_raw.unsqueeze(0)
                    rew_normalised = rew_normalised.unsqueeze(0)

                # add rewards
                exploration_returns_per_episode[range(self.args.num_processes), meta_traj_index] += infos[0]['reward_raw']

                if exploration_task_ids is None:
                    exploration_task_ids = [info['task_id'] for info in infos]
                    exploration_tasks = [info['task'] for info in infos]
                assert [info['task_id'] for info in infos] == exploration_task_ids \
                       and [info['task'] for info in infos] == exploration_tasks

                meta_traj_done_flag = np.array(
                    [max(meta_traj_done_flag[i], meta_episode_done[i]) for i in range(self.args.num_processes)], dtype=int)

                # Add the new step to the dataset storage
                self.transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw, infos,
                                                                  meta_traj_done_flag)

                # Update the latent with the new step
                traj_latent = self._encode_meta_traj(is_exploration_policy=True)  # (num_processes, d_z)

                traj_done = next_state[:, -1].cpu()
                traj_done_indices = torch.argwhere(traj_done == 1).cpu()

                prev_state = next_state
                self.frames += self.args.num_processes

                if len(traj_done_indices) > 0:
                    # The dataset should expect a new trajectory for this meta-trajectory and task
                    self.transformer.dataset_storage.new_traj(indices=traj_done_indices)
                    meta_traj_index += 1

                    if self.measure_mid_episodes:
                        # FIXME Assumes num_processes == 1
                        exploration_state = self.transformer.dataset_storage.state.clone()
                        exploration_meta_traj_index = copy.deepcopy(self.transformer.dataset_storage.meta_traj_index)
                        # When a trajectory is done, evaluate the task policy (using current exploration policies as context)
                        saved_storage = self.transformer.dataset_storage.save()
                        task_returns_per_episode[range(self.args.num_processes), meta_traj_index] \
                            = self._exploit(task_envs,
                                            exploration_task_ids,
                                            exploration_tasks,
                                            idx=-1)
                        self.transformer.dataset_storage.load(saved_storage)
                        assert torch.all(torch.eq(exploration_state, self.transformer.dataset_storage.state)) \
                               and exploration_meta_traj_index == self.transformer.dataset_storage.meta_traj_index, \
                            'The dataset storage no longer contains exploration data'

                    traj_latent = self._encode_meta_traj(is_exploration_policy=True)  # (num_processes, d_z)

                    for proc in range(self.args.num_processes):
                        if meta_traj_done_flag[proc] == 0 and traj_done[proc]:
                            prev_state[proc] = torch.from_numpy(infos[proc]['start_state']).to(global_device())
                            self.transformer.dataset_storage.insert_initial_state(prev_state[proc], proc)

        task_returns_per_episode[range(self.args.num_processes), -1] \
            = self._exploit(task_envs,
                            exploration_task_ids,
                            exploration_tasks,
                            idx=-1)

        exploration_returns_per_episode = exploration_returns_per_episode[:, :-1]
        return task_returns_per_episode, exploration_returns_per_episode

    def _exploit(self, task_envs, exploration_task_ids, exploration_tasks, idx=None):
        task_returns_per_episode = torch.zeros(self.args.num_processes).to(global_device())  # (num_proc)

        if self.args.ablation_use_context and not self.args.ablation_true_task:
            # Get all trajectories collected by the exploration policy
            exploration_trajectories = self.transformer.dataset_storage.get_batch(
                batch_size=len(self.transformer.dataset_storage))  # (p, 1, d, m*H)
            # if exploration_trajectories.shape[0] > 1:
            #     exploration_trajectories = exploration_trajectories[:1]

            # Task context is given by the (composed) latent of the exploration meta-trajectories
            task_context, _, _, _ = self.transformer.forward(exploration_trajectories,
                                                             compute_shared_latent=False,
                                                             use_static_shared_latent=True)  # (p, 1, H*d_z, m)
            assert task_context.shape[1] == 1, f'Shape: {task_context.shape[1]}'
            # FIXME Assumes num_processes == 1:
            assert task_context.shape[0] == 1, f'Shape: {task_context.shape[0]}'
            task_context = task_context.squeeze(1)  # (p, H*d_z, m)

            if self.args.context_type != 'full':
                task_context = task_context.reshape(
                    task_context.shape[0], self.args.horizon, -1, task_context.shape[-1])  # (p, H, d_z, m)
                if self.args.context_type == 'traj_latest':
                    task_context = task_context[:, -1, :, :]  # (p, d_z, m)
                elif self.args.context_type == 'latest':
                    task_context = task_context[:, -1, :, -1]  # (p, d_z)
            task_context = task_context.reshape(task_context.shape[0], -1)  # (p, H*d_z*m / d_z*m / d_z)
        else:
            task_context = None

        updated_context = False if self.args.ablation_true_task else None
        context = None

        self._print(f'{50 * "="} EXPLOITING {50 * "="}')
        for _ in range(self.args.repeat_task_traj):
            self.transformer.dataset_storage.clear()
            self.transformer.dataset_storage.new_task()
            self.transformer.dataset_storage.new_meta_traj()

            # Reset the environment to the same (single) task
            utl.reset_env(task_envs, self.args)

            # FIXME Assumes num_processes == 1:
            if not self.args.ablation_true_task:
                context = task_context[0] if task_context is not None else None  # (H*d_z*m)
            else:
                if not updated_context:
                    assert context is None
                    # FIXME Replace 4 with an argument
                    context = torch.zeros((1, 4), device=global_device())

            if (context is not None) and (len(context.shape) == 1):
                context = context.unsqueeze(0)

            # Reset the whole BAMDP environment, but to the same task
            prev_state = utl.reset_meta_traj(task_envs)  # (num_processes, d_obs)
            if len(prev_state.shape) == 1:
                prev_state = prev_state.unsqueeze(0)
            self.transformer.dataset_storage.insert_initial_state(prev_state)

            # The latent prior
            traj_latent = self._encode_meta_traj(is_exploration_policy=False,
                                                 return_prior=True, starting_state=prev_state)  # (num_processes, d_z)

            # Run for a single trajectory per task
            traj_done_flag = np.zeros(self.args.num_processes, dtype=int)
            while traj_done_flag.sum() != self.args.num_processes:
                # sample actions from the task policy (using the latent traj and task context)
                _, action = self._select_action(self.task_policy,
                                                prev_state,
                                                traj_latent,
                                                traj_done_flag,
                                                task_context=context,
                                                )  # (num_processes, 1), (num_processes, action_dim)

                # Take step in the environment
                #   meta_episode_done is true only when the last traj of the meta-traj ends (default 10 trajs)
                #   next_state will contain the episode_done flag appended at the end of the next_state features
                # If the current trajectory is finished, env_step will automatically reset the environment
                # (but NOT the task) and return:
                #   the last state of the previous traj in next_state
                #   the first state of the new traj in info['start_state']
                # FIXME Look into normalising obs, acts, rews
                next_state, (rew_raw, rew_normalised), _, infos = utl.env_step(task_envs, action, self.args)

                if self.args.ablation_true_task and not updated_context:
                    if idx == -1:
                        # FIXME Assumes num_processes == 1:
                        assert len(infos) == 1
                        context = []
                        task = infos[0]['task']
                        # FIXME Replace with a for loop
                        context.append(torch.tensor([task[0][0], task[1][0], task[2][0], task[3][0]]).unsqueeze(0))

                        context = torch.cat(context, dim=0)
                        context = context.to(global_device())

                        if len(context.shape) == 1:
                            context = context.unsqueeze(0)
                    elif idx != 0:
                        raise ValueError

                    updated_context = True

                if len(next_state.shape) == 1:
                    next_state = next_state.unsqueeze(0)
                    rew_raw = rew_raw.unsqueeze(0)
                    rew_normalised = rew_normalised.unsqueeze(0)

                # add rewards
                task_returns_per_episode[range(self.args.num_processes)] += infos[0]['reward_raw']

                # Make sure we are using the same tasks as during exploration
                if exploration_task_ids is not None and exploration_tasks is not None:
                    assert [info['task_id'] for info in infos] == exploration_task_ids \
                           and [info['task'] for info in infos] == exploration_tasks

                # A flag for whether the current traj (not meta-traj) is done
                traj_done = next_state[:, -1].cpu()
                traj_done_flag = np.array(
                    [max(traj_done_flag[i], traj_done[i]) for i in range(self.args.num_processes)], dtype=int)

                # Add the new step to the dataset storage
                self.transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw, infos,
                                                                  traj_done_flag)

                # Update the latent with the new step
                traj_latent = self._encode_meta_traj(is_exploration_policy=False)  # (num_processes, d_z)

                prev_state = next_state
                self.frames += self.args.num_processes

        self.transformer.dataset_storage.clear()
        task_returns_per_episode = task_returns_per_episode / self.args.repeat_task_traj
        task_returns_per_episode = task_returns_per_episode.unsqueeze(-1)
        return task_returns_per_episode
