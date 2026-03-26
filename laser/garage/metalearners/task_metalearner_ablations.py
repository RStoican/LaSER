import copy
import os
import time
from typing import Iterable

import numpy as np
import torch

from laser.garage.trainers.task_policy_trainer import TaskPolicyTrainer
from laser.garage.torch._functions import global_device
# from laser.garage.utils import evaluation as trans_utl_eval
from laser.garage.utils import helpers as utl
from laser.garage.utils import metalearner_helpers as mlutl
from laser.garage.metalearners.task_metalearner import TaskMetaLearner


class TaskMetaLearnerAblations(TaskMetaLearner):
    def __init__(self, args, logger):
        assert args.ablation_true_task
        assert not args.ablation_use_history
        super().__init__(args, logger)

    def _init_model(self):
        # The dimension of the task context used by the task policy
        self.args.context_type = 'latest'
        self.args.latent_dim = int(np.prod(self.envs.action_space.shape))
        self.args.policy_task_context_embedding_dim = self.args.latent_dim
        self.args.policy_context_hidden_layers = None

        metalearner_initialiser = mlutl.MetaLearnerInitialiser(args=self.args,
                                                               envs=self.envs,
                                                               exploration_train_mode=False,
                                                               get_iter_idx=lambda: self.iter_idx,
                                                               logger=self.logger, )

        # Transformer and policies
        self.transformer = metalearner_initialiser.initialise_autoencoder()
        self.exploration_policy = None

        self.args.pass_latent_to_policy = False
        self.task_policy = metalearner_initialiser.initialise_policy(policy_type='task')
        self.shared_latent = metalearner_initialiser.initialise_shared_latent()

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

    def _collect_task_traj(self, policy_trainer, exploration_tasks,
                           envs=None, batches_task_collect=None, tasks_per_iter=None,
                           success_tensor=None):
        envs = self.envs if envs is None else envs
        batches_task_collect = self.args.batches_task_collect if batches_task_collect is None else batches_task_collect
        tasks_per_iter = self.args.tasks_per_iter if tasks_per_iter is None else tasks_per_iter

        assert policy_trainer.policy_storage.is_empty()
        assert self.transformer.dataset_storage.is_empty()

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
                context = torch.zeros((self.args.num_processes, int(np.prod(envs.action_space.shape))),
                                      device=global_device())

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

                infos = None
                traj_done_flag = np.zeros(self.args.num_processes, dtype=int)

                # Run for a single trajectory per task
                updated_context = False
                while traj_done_flag.sum() != self.args.num_processes:
                    # sample actions from policy
                    value, action = self._select_action(
                        policy_trainer.policy,
                        prev_state,
                        traj_done_flag,
                        latent=None,
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

                    if self.args.ablation_use_context and not updated_context:
                        context = []
                        for proc in range(self.args.num_processes):
                            task = infos[proc]['task']
                            # FIXME Replace with a for loop
                            context.append(torch.tensor([task[0][0], task[1][0], task[2][0], task[3][0]]).unsqueeze(0))
                        context = torch.cat(context, dim=0)
                        context = context.to(global_device())
                        updated_context = True

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

                    prev_state = next_state
                    self.frames += self.args.num_processes - traj_done_flag.sum()

                    # Add the value of the last state for any finished trajectories
                    if len(traj_done_indices) > 0:
                        # Get the value of the last state
                        value, action = self._select_action(
                            policy_trainer.policy,
                            prev_state,
                            traj_done_flag,
                            latent=None,
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
        return None

    def _check_exploration_tasks(self, infos, exploration_tasks, task_idx):
        pass
