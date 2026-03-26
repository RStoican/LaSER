import copy
import os
import time
import warnings
from typing import Iterable

import numpy as np
import torch
from tqdm import trange

from laser.garage.envs._utils import gym_to_akro
from laser.garage.metalearners.metalearner import MetaLearner
from laser.garage.torch._functions import global_device
from laser.garage.trainers.task_policy_trainer import TaskPolicyTrainer
from laser.garage.utils import helpers as utl
from laser.garage.utils import metalearner_helpers as mlutl


class TaskMetaLearner(MetaLearner):
    def __init__(self, args, logger):
        # Reset to new seed
        utl.seed(args.seed, args.deterministic_execution)

        args.norm_state_task = args.norm_state_exploration or args.norm_state_task

        super().__init__(args, logger)

    def _create_envs(self):
        # initialise environments
        loaded_tasks, train_tasks, test_tasks = None, None, None
        if 'MetaWorld' in self.args.env_name:
            loaded_tasks = mlutl.load_tasks(self.args, self.logger.full_output_folder)
            if loaded_tasks is not None:
                train_tasks, test_tasks = loaded_tasks
            else:
                import metaworld
                warnings.warn(f'Could not load tasks from {self.logger.full_output_folder}')
                if self.args.env_name == 'CustomMetaWorldML1-v0':
                    metaworld_ml = metaworld.ML1(self.args.task_type, seed=self.args.seed)
                elif self.args.env_name == 'CustomMetaWorldML10-v0':
                    metaworld_ml = metaworld.ML10(seed=self.args.seed)
                else:
                    raise ValueError
                train_tasks = metaworld_ml.train_tasks
                test_tasks = metaworld_ml.test_tasks

        self.envs = mlutl.ml_make_vec_envs(self.args, tasks=None, mode='train', eval=False,
                                           loaded_tasks=(train_tasks, None))
        self.eval_envs = mlutl.ml_make_vec_envs(self.args, tasks=None, mode='test', eval=True,
                                                loaded_tasks=(None, test_tasks))

        if 'MetaWorld' in self.args.env_name:
            self._check_all_saved_tasks(loaded_tasks)

        if self.args.single_task_mode:
            raise NotImplementedError
            self.envs, self.train_tasks = mlutl.create_single_task_envs(args, self.envs, self.logger.full_output_folder)
        else:
            self.train_tasks = None

    def _init_model(self):
        metalearner_initialiser = mlutl.MetaLearnerInitialiser(args=self.args,
                                                               envs=self.envs,
                                                               exploration_train_mode=False,
                                                               get_iter_idx=lambda: self.iter_idx,
                                                               logger=self.logger, )

        # Transformer and policies
        self.transformer = metalearner_initialiser.initialise_autoencoder()
        self.exploration_policy = metalearner_initialiser.initialise_policy(
            policy_type='exploration') if self.args.ablation_use_context else None
        if not self.args.ablation_use_history:
            self.args.pass_latent_to_policy = False
        self.task_policy = metalearner_initialiser.initialise_policy(policy_type='task')
        self.shared_latent = metalearner_initialiser.initialise_shared_latent()

        if self.args.ablation_use_context:
            mlutl.load_models(self.args.save_path, self.args.model_label, self.transformer, self.exploration_policy,
                              self.shared_latent, norm_state_exploration=self.args.norm_state_exploration)

        if self.args.norm_state_task and not self.args.norm_state_exploration:
            utl.create_global_state_rms(self.args, force=True)
        if self.args.norm_rew_task:
            utl.create_global_rew_rms(self.args, force=True)

        # The task transformer is the same as the default (static) transformer if we're not fine-tuning.
        # If we are, then the task transformer is a perfect copy of the default, that will be updated using the RL loss
        if self.args.finetune_to_task:
            self.finetune_transformer = metalearner_initialiser.initialise_autoencoder(lr=self.args.lr_finetune,
                                                                                       finetune_mode=True)
            self.finetune_transformer.uni_encoder.load_state_dict(self.transformer.uni_encoder.state_dict())
            self.finetune_transformer.bi_encoder.load_state_dict(self.transformer.bi_encoder.state_dict())
            shared_latent = torch.load(
                os.path.join(os.path.join(self.args.save_path, 'models'), f'shared_latent{self.args.model_label}.pt'))
            self.finetune_transformer.set_static_shared_latent(shared_latent)
            self.task_transformer = self.finetune_transformer
        else:
            self.task_transformer = self.transformer

        # Storage for the task policy
        self.policy_storage = metalearner_initialiser.initialise_policy_storage(policy_type='task')

        # Task policy trainer
        self.task_trainer = TaskPolicyTrainer(self.args, self.task_policy, self.policy_storage)

        self.evaluator = None

        # FIXME Modify the policy arguments before creating the policy
        if not self.args.ablation_use_context:
            self.task_policy.actor_critic.pass_task_context_to_policy = False

    def _gym_to_akro(self):
        # We use akro-type environments
        self.envs = gym_to_akro(self.envs)
        self.eval_envs = gym_to_akro(self.eval_envs)

    def train(self):
        """ Main Meta-Training loop """

        # reset environments
        _ = utl.reset_env(self.envs, self.args)
        _ = utl.reset_env(self.eval_envs, self.args)

        # log once before training
        self.log(None, None, None)

        for self.iter_idx in trange(self.args.task_train_epochs):
            exploration_tasks = self._collect_exploration_traj()
            env_infos, success = self._collect_task_traj(self.task_trainer, exploration_tasks)
            policy_train_stats, self.timer_task_policy_update = self.task_trainer.train_policy(self.iter_idx)

            # Log and cleanup
            self.log(policy_train_stats, env_infos, success)
            self.task_trainer.policy_storage.after_update()

        if not self.args.ablation_fixed_tasks:
            self._log_test()

    def _collect_task_traj(self, policy_trainer, exploration_tasks,
                           envs=None, batches_task_collect=None, tasks_per_iter=None,
                           success_tensor=None):
        envs = self.envs if envs is None else envs
        batches_task_collect = self.args.batches_task_collect if batches_task_collect is None else batches_task_collect
        tasks_per_iter = self.args.tasks_per_iter if tasks_per_iter is None else tasks_per_iter

        assert policy_trainer.policy_storage.is_empty()
        assert not self.transformer.dataset_storage.is_empty() if self.args.ablation_use_context else True

        if self.args.ablation_use_context:
            # Get all trajectories collected by the exploration policy
            exploration_trajectories = self.transformer.dataset_storage.get_batch(
                batch_size=len(self.transformer.dataset_storage))  # (p, 1, d, m*H)

            # Make sure there are no entire exploration meta-trajectories that are padding
            # FIXME Maybe also check there are no trajectories where all steps are padding
            sum_per_task = exploration_trajectories.abs().sum(dim=list(range(1, len(exploration_trajectories.shape))))
            assert sum_per_task.count_nonzero() == sum_per_task.numel()

            with torch.no_grad():
                # Task context is given by the (composed) latent of the exploration meta-trajectories
                task_context, _, _, _ = self.task_transformer.forward(exploration_trajectories,
                                                                      compute_shared_latent=False,
                                                                      use_static_shared_latent=True)  # (p, 1, H*d_z, m)

                assert task_context.shape[1] == 1
                task_context = task_context.squeeze(1)  # (p, H*d_z, m)

                if self.args.context_type != 'full':
                    task_context = task_context.reshape(
                        task_context.shape[0], self.args.horizon, -1, task_context.shape[-1])  # (p, H, d_z, m)
                    if self.args.context_type == 'traj_latest':
                        task_context = task_context[:, -1, :, :]  # (p, d_z, m)
                    elif self.args.context_type == 'latest':
                        task_context = task_context[:, -1, :, -1]  # (p, d_z)
                task_context = task_context.reshape(task_context.shape[0], -1)  # (p, H*d_z*m / d_z*m / d_z)

            if self.args.finetune_to_task:
                self.task_trainer.setup_finetune(exploration_trajectories, self.task_transformer)
        else:
            task_context = None

        # Keep track of the environment info of all tasks
        all_infos = []

        success = torch.zeros(self.args.num_processes).to(global_device())

        start_time = time.time()
        for iter_idx in range(batches_task_collect):  # i \in [p]
            self.previous_task_ids, self.previous_tasks = None, None

            for task_idx in range(tasks_per_iter):
                # We no longer need the data collected during exploration or during the previous task collection
                self.transformer.dataset_storage.clear()
                self.task_transformer.dataset_storage.clear()

                # The dataset should expect meta-trajectories for a new task
                self.task_transformer.dataset_storage.new_task()

                # Keep track of the current tasks being run. These shouldn't change for this iteration
                self.initial_task_ids, self.initial_tasks = None, None

                # Reset the environment to the exploration tasks
                utl.reset_env(envs, self.args)
                envs.reset_task(
                    exploration_tasks['task_ids'][task_idx] if exploration_tasks is not None else [None])

                # Get the (exploration) context of the current tasks
                process_task_idx = task_idx * self.args.num_processes
                context = task_context[process_task_idx:process_task_idx + self.args.num_processes] \
                    if task_context is not None else None

                # The index of the current task in the policy storage
                task_storage_idx = self.args.num_processes * (iter_idx * self.args.tasks_per_iter + task_idx)

                # The dataset should expect a new meta-trajectory for the current task
                self.task_transformer.dataset_storage.new_meta_traj()
                meta_steps = self.task_transformer.dataset_storage.meta_step_index.clone()

                # Reset the whole BAMDP environment, but to the same tasks for the current task_idx
                prev_state = utl.reset_meta_traj(envs)  # (num_processes, d_obs)
                prev_state = prev_state.unsqueeze(0) if self.args.num_processes == 1 and len(prev_state.shape) < 2 \
                    else prev_state
                self.task_transformer.dataset_storage.insert_initial_state(prev_state)

                # The latent prior
                traj_latent = self._encode_task_meta_traj(
                    transformer=self.task_transformer,
                    return_prior=True,
                    starting_state=prev_state,
                )  # (num_processes, d_z)

                infos = None
                traj_done_flag = np.zeros(self.args.num_processes, dtype=int)

                # Run for a single trajectory per task
                while traj_done_flag.sum() != self.args.num_processes:
                    # sample actions from policy
                    value, action = self._select_action(
                        policy_trainer.policy,
                        prev_state,
                        traj_done_flag,
                        traj_latent,
                        task_context=context,
                    )  # (num_processes, 1), (num_processes, action_dim)

                    # Insert the (predicted) value of the current state to the policy storage
                    policy_trainer.policy_storage.insert_value(value,
                                                               task_idx=task_storage_idx,
                                                               meta_traj_idx=0,
                                                               meta_step=meta_steps,
                                                               infos=infos)

                    # The current meta-steps for each meta-trajectory
                    meta_steps = self.task_transformer.dataset_storage.meta_step_index.clone()

                    # Take step in the environment
                    #   meta_episode_done is true only when the last traj of the meta-traj ends (default 10 trajs)
                    #   next_state will contain the episode_done flag appended at the end of the next_state features
                    # If the current trajectory is finished, env_step will automatically reset the environment
                    # (but NOT the task) and return:
                    #   the last state of the previous traj in next_state
                    #   the first state of the new traj in info['start_state']
                    # FIXME Look into normalising obs, acts, rews
                    next_state, (rew_raw, rew_normalised), _, infos \
                        = utl.env_step(envs, action, self.args)
                    all_infos += infos

                    # FIXME
                    if 'MetaWorld' in self.args.env_name:
                        if iter_idx == batches_task_collect - 1 and task_idx == tasks_per_iter - 1:
                            for i, info in enumerate(infos):
                                if info['success'] > 0:
                                    success[i] = 1

                    # Check that the tasks we are currently using are valid
                    self._check_tasks(infos)
                    # Make sure we are using the same tasks as during exploration
                    self._check_exploration_tasks(infos, exploration_tasks, task_idx)

                    # A flag for whether the current traj (not meta-traj) is done
                    traj_done = next_state[:, -1].cpu()
                    traj_done_indices = torch.argwhere(traj_done == 1).cpu()
                    traj_done_flag = np.array(
                        [max(traj_done_flag[i], traj_done[i]) for i in range(self.args.num_processes)], dtype=int)

                    # Add the new step to the dataset storage
                    self.task_transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw,
                                                                           infos, traj_done_flag,
                                                                           rew_norm=rew_normalised)

                    # Update the latent with the new step
                    traj_latent = self._encode_task_meta_traj(
                        transformer=self.task_transformer
                    )  # (num_processes, d_z)

                    prev_state = next_state
                    self.frames += self.args.num_processes - traj_done_flag.sum()

                    # Add the value of the last state for any finished trajectories
                    if len(traj_done_indices) > 0:
                        # Get the value of the last state
                        value, action = self._select_action(
                            policy_trainer.policy,
                            prev_state,
                            traj_done_flag,
                            traj_latent,
                            task_context=context,
                        )  # (num_processes, 1)

                        for proc in range(self.args.num_processes):
                            if traj_done[proc]:
                                # Insert the value of the last state of this episode
                                policy_trainer.policy_storage.insert_value(value,
                                                                           task_idx=task_storage_idx,
                                                                           meta_traj_idx=0,
                                                                           meta_step=meta_steps,
                                                                           process=proc)

                # Insert the current meta-trajectories and latents for num_processes into the policy storage
                self._insert_task_policy_data(policy_trainer,
                                              task_storage_idx,
                                              context)

                self.previous_task_ids = self.initial_task_ids
                self.previous_tasks = self.initial_tasks

        # Make sure we don't keep any of the exploitation data in the transformer storage
        self.task_transformer.dataset_storage.clear()

        self.timer_task_collect = time.time() - start_time
        return all_infos, success.mean()

    def _collect_exploration_traj(self, envs=None):
        if not self.args.ablation_use_context:
            return None

        envs = self.envs if envs is None else envs

        # The reward normaliser should not be updated during exploration
        envs.venv.eval()

        # print(f'Collecting exploration data from {self.args.tasks_per_iter * self.args.num_processes} tasks')
        assert self.transformer.dataset_storage.is_empty()

        # Keep track of the tasks we are exploring, to use for training the task policy
        exploration_tasks = {'task_ids': [], 'tasks': []}

        max_storage_len = 0
        self.previous_task_ids, self.previous_tasks = None, None
        start_time = time.time()

        for task_idx in range(self.args.tasks_per_iter):  # i \in [p]
            # The dataset should expect meta-trajectories for a new task
            self.transformer.dataset_storage.new_task()

            # The dataset storage should exactly fit all exploration trajectories. So, make sure there was no shifting
            assert len(self.transformer.dataset_storage) == 0 \
                   or len(self.transformer.dataset_storage) > max_storage_len, \
                f'Storage size decreased from {max_storage_len} to {len(self.transformer.dataset_storage)}. ' \
                f'The storage is expected to fit {self.args.tasks_per_iter * self.args.num_processes} tasks'
            max_storage_len = max(max_storage_len, len(self.transformer.dataset_storage))

            # Keep track of the current tasks being run. These shouldn't change for this iteration of the outermost loop
            self.initial_task_ids, self.initial_tasks = None, None

            # Reset the environment to new tasks
            utl.reset_env(envs, self.args)

            # The dataset should expect a new meta-trajectory for the current task
            self.transformer.dataset_storage.new_meta_traj()

            # Reset the whole BAMDP environment, but to the same tasks for the current task_idx
            prev_state = utl.reset_meta_traj(envs)  # (num_processes, d_obs)
            prev_state = prev_state.unsqueeze(0) if self.args.num_processes == 1 and len(prev_state.shape) < 2 \
                else prev_state
            self.transformer.dataset_storage.insert_initial_state(prev_state)

            # The latent prior
            traj_latent = self._encode_meta_traj(
                transformer=self.transformer,
                return_prior=True,
                starting_state=prev_state,
            )  # (num_processes, d_z)

            meta_traj_done_flag = np.zeros(self.args.num_processes, dtype=int)
            while meta_traj_done_flag.sum() != self.args.num_processes:
                # Sample actions from policy
                # FIXME Think about testing with a deterministic exploration policy
                _, action = self._select_action(
                    self.exploration_policy,
                    prev_state,
                    meta_traj_done_flag,
                    traj_latent,
                )  # _, (num_processes, action_dim)

                # Take step in the environment
                #   meta_episode_done is true only when the last traj of the meta-traj ends (default 10 trajs)
                #   next_state will contain the episode_done flag appended at the end of the next_state features
                # If the current trajectory is finished, env_step will automatically reset the environment
                # (but NOT the task) and return:
                #   the last state of the previous traj in next_state
                #   the first state of the new traj in info['start_state']
                next_state, (rew_raw, _), meta_episode_done, infos \
                    = utl.env_step(envs, action, self.args)

                # Check that the tasks we are currently using are valid
                self._check_tasks(infos)

                meta_traj_done_flag = np.array(
                    [max(meta_traj_done_flag[i], meta_episode_done[i]) for i in range(self.args.num_processes)],
                    dtype=int)

                # Add the new step to the dataset storage
                self.transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw, infos,
                                                                  meta_traj_done_flag)

                # Update the latent with the new step
                traj_latent = self._encode_meta_traj(
                    transformer=self.transformer,
                )  # (num_processes, d_z)

                # A flag for whether the current traj (not meta-traj) is done
                traj_done = next_state[:, -1].cpu()
                traj_done_indices = torch.argwhere(traj_done == 1).cpu()

                prev_state = next_state

                if len(traj_done_indices) > 0:
                    # The dataset should expect a new trajectory for this meta-trajectory and task
                    self.transformer.dataset_storage.new_traj(indices=traj_done_indices)

                    for proc in range(self.args.num_processes):
                        if traj_done[proc]:
                            # Insert the first state of the new episode
                            if meta_traj_done_flag[proc] == 0:
                                prev_state[proc] = torch.from_numpy(infos[proc]['start_state']).to(global_device())
                                self.transformer.dataset_storage.insert_initial_state(prev_state[proc], proc)
                                # FIXME: traj_latent = self._encode_meta_traj()  # (num_processes, d_z) ???
                            else:
                                infos[proc]['padding'] = True

            self.previous_task_ids = self.initial_task_ids
            self.previous_tasks = self.initial_tasks

            exploration_tasks['task_ids'].append(self.initial_task_ids)
            exploration_tasks['tasks'].append(self.initial_tasks)

        envs.venv.train()

        self.timer_exploration_collect = (time.time() - start_time) / self.args.tasks_per_iter
        return exploration_tasks

    def _insert_task_policy_data(self, policy_trainer, task_storage_idx, context):
        (states, actions, rewards), lens = \
            self.task_transformer.dataset_storage.get_working_meta_traj(split=True, lens=True)
        rewards_normalised = self.task_transformer.dataset_storage.get_working_rew_norm()

        # Keep only the first trajectory. The rest are just padding
        states, actions, rewards = \
            states[:, :, :self.args.horizon], actions[:, :, :self.args.horizon], rewards[:, :, :self.args.horizon]
        lens = lens[:, 0].unsqueeze(-1)
        if rewards_normalised is not None:
            rewards_normalised = rewards_normalised[:, :, :self.args.horizon]

        policy_trainer.policy_storage.insert_meta_traj(meta_traj=(states, actions, rewards),
                                                       task_idx=task_storage_idx,
                                                       meta_traj_idx=0,
                                                       lens=lens,
                                                       norm_rewards=rewards_normalised)
        if self.args.ablation_use_context:
            policy_trainer.policy_storage.insert_task_context(task_context=context,
                                                              task_idx=task_storage_idx,
                                                              meta_traj_idx=0)

        if self.args.ablation_use_history:
            with torch.no_grad():
                # For unidirectional attention, these latents should be the same as the ones computed online
                traj_latent = self._encode_task_meta_traj(
                    transformer=self.task_transformer,
                    return_latest_latent=False,
                )  # (num_processes, d_z, m*H)
                traj_latent = traj_latent[:, :, :self.args.horizon]  # (num_processes, d_z, H)
            policy_trainer.policy_storage.insert_latent(latent=traj_latent,
                                                        task_idx=task_storage_idx,
                                                        meta_traj_idx=0)

        policy_trainer.policy_storage.update_masks(task_idx=task_storage_idx,
                                                   meta_traj_idx=0)

    def _encode_task_meta_traj(self, return_latest_latent=True, return_prior=False, starting_state=None,
                                      transformer=None):
        # FIXME This won't work if the exploration policy was trained without the latent history
        if not self.args.ablation_use_history:
            return None
        return self._encode_meta_traj(return_latest_latent, return_prior, starting_state, transformer)

    def _encode_meta_traj(self, return_latest_latent=True, return_prior=False, starting_state=None,
                          transformer=None):
        assert transformer is not None
        return super()._encode_meta_traj(return_latest_latent, return_prior, starting_state, transformer)

    def _check_exploration_tasks(self, infos, exploration_tasks, task_idx):
        if not self.args.ablation_use_context:
            return
        check_ids = [info['task_id'] for info in infos] == exploration_tasks['task_ids'][task_idx]
        if 'MuJoCo' in self.args.env_name:
            # FIXME Make sure metalearner_helpers.check_tasks() stores the exploration tasks
            check_task_transitions = True
        else:
            if 'task' in infos[0]:
                check_task_transitions = [info['task'] for info in infos] == exploration_tasks['tasks'][task_idx]
            else:
                check_task_transitions = True
        assert check_ids and check_task_transitions, f'IDs {check_ids}; Transitions {check_task_transitions}'

    def _check_all_saved_tasks(self, loaded_tasks):
        if loaded_tasks is not None:
            assert loaded_tasks[0] == self.envs.train_tasks[0]
            assert loaded_tasks[1] == self.eval_envs.test_tasks[0]

    def _check_saved_tasks(self, loaded_tasks, envs):
        if self.args.save_train_tasks:
            assert envs.train_tasks[0] == loaded_tasks[0]
        if self.args.save_test_tasks:
            assert envs.test_tasks[0] == loaded_tasks[1]

    def _watch_training(self):
        self.logger.watch(self.task_policy.actor_critic, log_freq=self.args.log_interval, idx=0)
        if self.args.finetune_to_task:
            self.logger.watch(self.finetune_transformer, log_freq=self.args.log_interval, idx=1)

    def log(self, policy_train_stats, env_infos, train_success):
        with torch.no_grad():
            # --- save models ---
            mlutl.save_models(self.args, self.iter_idx, self.logger.full_output_folder,
                              task_policy=self.task_policy, )

            # --- log some other things ---
            do_log = ((self.iter_idx + 1) % self.args.log_interval == 0 or self.iter_idx <= 0) and (
                    policy_train_stats is not None)
            if do_log:
                self._log_misc(policy_train_stats, env_infos, train_success)

            # FIXME Think about running the exploration policy on an evaluation environment
            # --- evaluate policy on multiple adapting episodes on multiple tasks (in parallel) ----
            do_eval = (self.args.eval_interval > 0) and (self.iter_idx > -1) and (
                    ((self.iter_idx + 1) % self.args.eval_interval == 0) or (self.iter_idx == 0))
            if not self.args.ablation_fixed_tasks:
                if do_eval:
                    self._log_evaluate()

                do_test = (self.args.test_interval > 0) and (self.iter_idx > -1) and (
                        ((self.iter_idx + 1) % self.args.test_interval == 0) or (self.iter_idx == 0))
                if do_test:
                    self._log_test()
            else:
                do_test = False
                if do_eval:
                    self._log_fixed_tasks_evaluate()

            if do_eval or do_test or do_log:
                # Push the log
                self.logger.log(self.iter_idx)

    def _log_evaluate(self):
        self.task_policy.actor_critic.eval()

        self.task_trainer.policy_storage.after_update()
        exploration_tasks = self._collect_exploration_traj(envs=self.eval_envs)
        env_infos, eval_success = self._collect_task_traj(self.task_trainer, exploration_tasks, envs=self.eval_envs)

        return_per_task = self.policy_storage.rewards_raw.sum(dim=-1)
        if 'MetaWorld' in self.args.env_name:
            self.logger.add('eval/success_mean', eval_success)
        self.logger.add('eval/return_mean', return_per_task.mean())

        # Make sure the policy is the same as before the evaluation
        self.task_trainer.policy_storage.after_update()
        self.task_policy.actor_critic.train()

        assert self.task_policy.actor_critic.training

    def _log_test(self):
        if not self.args.run_test:
            return

        test_stats = self._run_test(self.args.test_repeat_task, self.args.test_repeat_task_traj)

        last_idx = copy.deepcopy(self.iter_idx)
        for k, v in test_stats.items():
            if isinstance(v, Iterable):
                start = last_idx + 1
                for i in range(len(v)):
                    self.logger.add(f'test/{k}', v[i])
                    self.logger.log(start + i)
                last_idx += len(v)

        for k, v in test_stats.items():
            if not isinstance(v, Iterable):
                self.logger.add(f'test/{k}', v)
        assert self.task_policy.actor_critic.training
        self.logger.log(last_idx + 1)

    def _log_fixed_tasks_evaluate(self):
        eval_stats = self._run_test(self.args.eval_repeat_task, self.args.eval_repeat_task_traj, simple_stats=True)
        for k, v in eval_stats.items():
            self.logger.add(f'eval/{k}', v)
        assert self.task_policy.actor_critic.training

    def _run_test(self, repeat_task, repeat_task_traj, simple_stats=False):
        tester_type = self.envs.tester_type
        tester = tester_type(
            args=self.args,
            envs=self.envs,
            repeat_task=repeat_task,
            repeat_task_traj=repeat_task_traj,
            full_output_folder=self.logger.full_output_folder,
            transformer=self.task_transformer,
            exploration_policy=self.exploration_policy,
            task_policy=self.task_policy,
            shared_latent=self.shared_latent,
        )
        return tester.run_test(simple_stats=simple_stats)

    def _log_misc(self, policy_train_stats, env_infos, train_success):
        self.logger.add('timer/exploration_traj_collection', round(self.timer_exploration_collect, 2))
        self.logger.add('timer/task_traj_collection', round(self.timer_task_collect, 2))

        if policy_train_stats is not None:
            # Encoder
            if self.policy_storage.latent_traj is not None:
                self.logger.add('encoder/latent_mean', self.policy_storage.latent_traj.mean())
                self.logger.add('encoder/latent_min', self.policy_storage.latent_traj.min())
                self.logger.add('encoder/latent_max', self.policy_storage.latent_traj.max())
            if self.policy_storage.task_context is not None:
                self.logger.add('encoder/task_context_mean', self.policy_storage.task_context.mean())
                self.logger.add('encoder/task_context_min', self.policy_storage.task_context.min())
                self.logger.add('encoder/task_context_max', self.policy_storage.task_context.max())

            # Environment
            # Average return over all tasks (using environment rewards)
            return_per_task = self.policy_storage.rewards_raw.sum(dim=-1)
            self.logger.add('environment/return_max', return_per_task.max())
            self.logger.add('environment/return_min', return_per_task.min())
            self.logger.add('environment/return_mean', return_per_task.mean())
            self.logger.add('environment/return_std', return_per_task.std())
            self.logger.add('environment/rew_max', self.policy_storage.rewards_raw.max())
            self.logger.add('environment/rew_min', self.policy_storage.rewards_raw.min())
            self.logger.add('environment/rew_mean', self.policy_storage.rewards_raw.mean())

            self.logger.add('environment/state_max', self.policy_storage.prev_state.max())
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min())
            self.logger.add('environment/state_mean', self.policy_storage.prev_state.mean())

            # FIXME This only makes sense for the MEWA env
            actions = self.policy_storage._actions.argmax(dim=2).float()
            self.logger.add('environment/action_max', actions.max())
            self.logger.add('environment/action_min', actions.min())
            self.logger.add('environment/action_mean', actions.mean())

            self.logger.add('environment/len_max', self.policy_storage.lens.max())
            self.logger.add('environment/len_min', self.policy_storage.lens.min())
            self.logger.add('environment/len_mean', self.policy_storage.lens.mean())
            self.logger.add('environment/len_std', self.policy_storage.lens.std())

            self.logger.add('environment/total_task_timesteps', self.frames)

            # Additional environment metrics
            if 'MetaWorld' in self.args.env_name:
                self.logger.add('environment/success_mean', train_success)

            # Task policy
            for k, v in policy_train_stats.items():
                self.logger.add(f'task_{k}', v)

            if hasattr(self.task_policy.actor_critic, 'logstd'):
                self.logger.add('task_policy/action_logstd',
                                self.task_policy.actor_critic.dist.logstd.mean())
            self.logger.add('task_policy/action_logprob', self.policy_storage.action_log_probs.mean())
            action_prob = torch.pow(2, self.policy_storage.action_log_probs)
            self.logger.add('task_policy/action_prob', action_prob.mean())
            self.logger.add('task_policy/action_prob_std', action_prob.std())
            self.logger.add('task_policy/action_prob_min', action_prob.min())
            self.logger.add('task_policy/action_prob_max', action_prob.max())
            quantiles = [0.001, 0.01, 0.05]
            for qant in quantiles:
                self.logger.add(f'task_policy/action_prob_quantile_{qant}', action_prob.quantile(qant))

            # FIXME For this to work, we would need to compute and store the log prob of each action, at each time-step
            #  Currently, we are only doing the log prob of one action per time-step
            # action_log_probs = self.policy_storage.action_log_probs
            # action_probs = torch.pow(2, action_log_probs)
            # entropy = -torch.matmul(action_probs.permute(0, 1, 3, 2), action_log_probs)
            # self.logger.add('task_policy/entropy', entropy.mean())

            self.logger.add('timer/task_policy_update', round(self.timer_task_policy_update, 2))
