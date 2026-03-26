import time

import numpy as np
import torch
from tqdm import trange

from laser.garage.metalearners.metalearner import MetaLearner
from laser.garage.torch._functions import global_device
from laser.garage.trainers.exploration_policy_trainer import ExplorationPolicyTrainer
from laser.garage.trainers.transformer_trainer import TransformerTrainer
from laser.garage.utils import helpers as utl
from laser.garage.utils import metalearner_helpers as mlutl


class ExplorationMetaLearner(MetaLearner):
    def __init__(self, args, logger):
        self.collect_policy_data = False
        self.use_starting_dataset = args.starting_dataset is not None
        self.log_shared_latent, self.log_task_latent, self.log_traj_latent = None, None, None

        super().__init__(args, logger)

    def _init_model(self):
        metalearner_initialiser = mlutl.MetaLearnerInitialiser(args=self.args,
                                                               envs=self.envs,
                                                               exploration_train_mode=True,
                                                               get_iter_idx=lambda: self.iter_idx,
                                                               logger=self.logger,
                                                               use_starting_dataset=self.use_starting_dataset, )

        # Transformer
        self.transformer, self.target_coeff_head = metalearner_initialiser.initialise_autoencoder()

        # Exploration policy
        self.policy_storage = metalearner_initialiser.initialise_policy_storage(policy_type='exploration')
        self.exploration_policy = metalearner_initialiser.initialise_policy(policy_type='exploration')

        # Transformer and exploration policy trainers
        self.transformer_trainer = TransformerTrainer(self.args, self.transformer, self.target_coeff_head,
                                                      lambda: self.iter_idx, self.logger)
        self.exploration_trainer = ExplorationPolicyTrainer(self.args, self.exploration_policy, self.policy_storage)

        utl.create_global_state_rms(self.args)

    def _create_envs(self):
        super()._create_envs()

        loaded_tasks = mlutl.save_tasks(self.args, self.envs, self.logger.full_output_folder)
        # Make sure all environments have the same list of tasks
        if self.args.save_train_tasks or self.args.save_test_tasks:
            self.envs = mlutl.ml_make_vec_envs(self.args, tasks=None, loaded_tasks=loaded_tasks)
            self._check_saved_tasks()

        if self.args.single_task_mode:
            self.envs, self.train_tasks = mlutl.create_single_task_envs(self.args, self.envs,
                                                                        self.logger.full_output_folder)
        else:
            self.train_tasks = None

    def train(self):
        """ Main Meta-Training loop """

        # reset environments
        _ = utl.reset_env(self.envs, self.args)

        # log once before training
        self.log(None, None)

        # The static shared latent gives us a description of our task distribution. Initialize to a random vector
        self.transformer.set_static_shared_latent(random=True)

        # Pre-collect a few batches of data before we start training
        self._precollect_data()

        # The main training loop
        self._train()

    def _train(self):
        for self.iter_idx in trange(self.args.pre_train_epochs):
            self._collect_traj(self.exploration_trainer)
            transformer_log_stats = self._train_encoder()
            policy_train_stats, self.timer_exploration_policy_update = self.exploration_trainer.train_policy(self.iter_idx)

            # Log and cleanup
            self.log(transformer_log_stats, policy_train_stats)
            self.policy_storage.after_update()

    def _train_encoder(self):
        self.transformer_trainer.check_target_coeff_head_update()

        log_stats = None
        start_time = time.time()

        for train_step in range(self.args.num_transformer_updates):
            # Perform one update step
            log_stats, (self.log_shared_latent, self.log_task_latent, self.log_traj_latent) \
                = self.transformer_trainer.update(log=train_step == self.args.num_transformer_updates - 1)

        self.timer_enc_update = (time.time() - start_time) / self.args.num_transformer_updates
        return log_stats

    def _collect_traj(self, policy_trainer=None):
        if self.use_starting_dataset:
            if self.iter_idx < self.args.transformer_pre_train_epochs:
                return
            self.use_starting_dataset = False
            if self.args.reset_starting_dataset:
                self.transformer.dataset_storage.clear()

        assert self.policy_storage.is_empty()

        # Check if we are just collecting trajectories or doing encoder pre-training
        self.collect_policy_data = self.iter_idx != -1 and self.iter_idx >= self.args.transformer_pre_train_epochs
        if self.collect_policy_data:
            assert policy_trainer is not None

        self.previous_task_ids, self.previous_tasks = None, None
        start_time = time.time()

        # TODO Think about using a target network (updated every few iterations) for collecting trajectories
        for task_idx in range(self.args.tasks_per_iter):  # i \in [p]
            # The dataset should expect meta-trajectories for a new task
            self.transformer.dataset_storage.new_task()

            # Keep track of the current tasks being run. These shouldn't change for this iteration of the outermost loop
            self.initial_task_ids, self.initial_tasks = None, None

            # Reset the environment to new tasks
            utl.reset_env(self.envs, self.args)

            for meta_traj_idx in range(self.args.meta_traj_per_task):  # k \in [q]
                # The dataset should expect a new meta-trajectory for the current task
                self.transformer.dataset_storage.new_meta_traj()
                meta_steps = self.transformer.dataset_storage.meta_step_index.clone()

                # Reset the whole BAMDP environment, but to the same tasks for the current task_idx
                prev_state = utl.reset_meta_traj(self.envs)  # (num_processes, d_obs)
                self.transformer.dataset_storage.insert_initial_state(prev_state)

                # The latent prior
                # FIXME We might want to consider normalising the environment data before giving it to the
                #  transformer. This should avoid issues with large magnitude outputs
                traj_latent = self._encode_meta_traj(
                    return_prior=True,
                    starting_state=prev_state
                )  # (num_processes, d_z)

                infos = None
                self.meta_traj_done_flag = np.zeros(self.args.num_processes, dtype=int)

                while self.meta_traj_done_flag.sum() != self.args.num_processes:
                    # Sample actions from policy
                    value, action = self._select_exploration_action(
                        prev_state,
                        traj_latent,
                    )  # (n_proc, 1), (n_proc, act_dim)

                    # Insert the (predicted) value of the current state to the policy storage
                    if self.collect_policy_data:
                        self.policy_storage.insert_value(value,
                                                         task_idx=task_idx * self.args.num_processes,
                                                         meta_traj_idx=meta_traj_idx,
                                                         meta_step=meta_steps,
                                                         infos=infos)

                    # Update the current meta-steps for each meta-trajectory
                    meta_steps = self.transformer.dataset_storage.meta_step_index.clone()

                    # Take step in the environment
                    #   meta_episode_done is true only when the last traj of the meta-traj ends (default 10 trajs)
                    #   next_state will contain the episode_done flag appended at the end of the next_state features
                    # If the current trajectory is finished, env_step will automatically reset the environment
                    # (but NOT the task) and return:
                    #   the last state of the previous traj in next_state
                    #   the first state of the new traj in info['start_state']
                    # FIXME Look into normalising obs, acts, rews
                    next_state, (rew_raw, _), meta_episode_done, infos \
                        = utl.env_step(self.envs, action, self.args)

                    # Check that the tasks we are currently using are valid
                    self._check_tasks(infos)

                    self.meta_traj_done_flag = np.array(
                        [max(self.meta_traj_done_flag[i], meta_episode_done[i]) for i in
                         range(self.args.num_processes)],
                        dtype=int)

                    # Add the new step to the dataset storage
                    self.transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw, infos,
                                                                      self.meta_traj_done_flag)

                    # Update the latent with the new step
                    traj_latent = self._encode_meta_traj()  # (num_processes, d_z)

                    # A flag for whether the current traj (not meta-traj) is done
                    traj_done = next_state[:, -1].cpu()
                    traj_done_indices = torch.argwhere(traj_done == 1).cpu()

                    prev_state = next_state
                    self.frames += self.args.num_processes

                    if len(traj_done_indices) > 0:
                        # The dataset should expect a new trajectory for this meta-trajectory and task
                        self.transformer.dataset_storage.new_traj(indices=traj_done_indices)

                        # Get the value of the last state of this episode
                        if self.collect_policy_data:
                            value, _ = self._select_exploration_action(
                                prev_state,
                                traj_latent,
                            )  # (n_proc, 1), _

                        for proc in range(self.args.num_processes):
                            if traj_done[proc]:
                                # Insert the value of the last state of this episode
                                if self.collect_policy_data:
                                    self.policy_storage.insert_value(value,
                                                                     task_idx=task_idx * self.args.num_processes,
                                                                     meta_traj_idx=meta_traj_idx,
                                                                     meta_step=meta_steps,
                                                                     process=proc)
                                    meta_steps[proc] = self.transformer.dataset_storage.meta_step_index[proc].clone()

                                # Insert the first state of the new episode
                                if self.meta_traj_done_flag[proc] == 0:
                                    prev_state[proc] = torch.from_numpy(infos[proc]['start_state']).to(global_device())
                                    self.transformer.dataset_storage.insert_initial_state(prev_state[proc], proc)
                                    # FIXME: traj_latent = self._encode_meta_traj()  # (num_processes, d_z) ???
                                else:
                                    infos[proc]['padding'] = True

                # Insert the current meta-trajectories and latents for num_processes into the policy storage
                if self.collect_policy_data:
                    self._insert_policy_data(policy_trainer=policy_trainer,
                                             task_idx=task_idx,
                                             meta_traj_idx=meta_traj_idx)

            self.previous_task_ids = self.initial_task_ids
            self.previous_tasks = self.initial_tasks
        self.timer_collect = (time.time() - start_time) / self.args.tasks_per_iter

    def _insert_policy_data(self, policy_trainer, task_idx, meta_traj_idx):
        meta_traj, lens = self.transformer.dataset_storage.get_working_meta_traj(split=True, lens=True)
        self.policy_storage.insert_meta_traj(meta_traj=meta_traj,
                                             task_idx=task_idx * self.args.num_processes,
                                             meta_traj_idx=meta_traj_idx,
                                             lens=lens)

        # Computed with unidirectional attention; these latents are the same as the ones computed online
        traj_latent = self._encode_meta_traj(
            return_latest_latent=False,
        )  # (num_processes, d_z, m*H)
        self.policy_storage.insert_latent(latent=traj_latent,
                                          task_idx=task_idx * self.args.num_processes,
                                          meta_traj_idx=meta_traj_idx)

        # Some policies will not use the env rewards (e.g. use intrinsic exploration rewards)
        exploration_rewards = policy_trainer.create_rewards(traj_latent,
                                                            num_tasks=self.args.num_processes)  # (num_processes, m)
        self.policy_storage.insert_exploration_rewards(exploration_rewards,
                                                       task_idx=task_idx * self.args.num_processes,
                                                       meta_traj_idx=meta_traj_idx)

    def _select_exploration_action(self, prev_state, latent):
        return self._select_action(
            policy=self.exploration_policy,
            prev_state=prev_state,
            meta_traj_done_flag=self.meta_traj_done_flag,
            latent=latent,
        )

    def _precollect_data(self):
        print(f'Pre-collecting {self.args.batches_pre_collect * self.args.num_processes} tasks')
        if not self.use_starting_dataset:
            for _ in trange(int(self.args.batches_pre_collect)):
                self._collect_traj()
                print(f'Pre-Collected data from {len(self.transformer.dataset_storage)} tasks')
                if len(self.transformer.dataset_storage) + self.args.num_processes >= self.args.dataset_size:
                    return

    def _check_saved_tasks(self):
        if self.args.save_train_tasks:
            train_tasks = self.envs.train_tasks
            for i in range(len(train_tasks) - 1):
                assert train_tasks[i] == train_tasks[i + 1]
        if self.args.save_test_tasks:
            test_tasks = self.envs.test_tasks
            for i in range(len(test_tasks) - 1):
                assert test_tasks[i] == test_tasks[i + 1]

    def _watch_training(self):
        self.logger.watch(self.transformer, log_freq=self.args.log_interval, idx=0)
        if self.target_coeff_head is not None:
            self.logger.watch(self.target_coeff_head, log_freq=self.args.log_interval, idx=1)
        self.logger.watch(self.exploration_policy.actor_critic, log_freq=self.args.log_interval, idx=2)

    def log(self, transformer_log_stats, policy_train_stats):
        with torch.no_grad():
            # --- save models ---
            mlutl.save_models(self.args, self.iter_idx, self.logger.full_output_folder,
                              transformer=self.transformer,
                              exploration_policy=self.exploration_policy, )

            # --- log some other things ---
            initial_log = transformer_log_stats is None and policy_train_stats is None
            if ((self.iter_idx + 1) % self.args.log_interval == 0 or self.iter_idx <= 0) and not initial_log:
                self._log_misc(transformer_log_stats, policy_train_stats)
                # Push the log
                self.logger.log(self.iter_idx)

    def _log_misc(self, transformer_log_stats, policy_train_stats):
        if isinstance(transformer_log_stats, tuple) or isinstance(transformer_log_stats, list):
            transformer_loss_stats, transformer_latent_stats = transformer_log_stats
        else:
            transformer_loss_stats = transformer_log_stats
            transformer_latent_stats = None

        # Transformer
        self.logger.add('dataset/len', len(self.transformer.dataset_storage))
        if transformer_loss_stats is not None:
            # Transformer Losses
            for k, v in transformer_loss_stats.items():
                self.logger.add(f'transformer_losses/{k}', v)

            # Latent
            self.logger.add('latent/shared', self.log_shared_latent.abs().mean())
            self.logger.add('latent/task', self.log_task_latent.abs().mean())
            self.logger.add('latent/traj', self.log_traj_latent.abs().mean())

        if transformer_latent_stats is not None:
            for k, v in transformer_latent_stats.items():
                self.logger.add(f'latent_exploration/{k}', v)

        self.logger.add('timer/traj_collection', round(self.timer_collect, 2))
        self.logger.add('timer/enc_update', round(self.timer_enc_update, 2))

        if policy_train_stats is not None:
            # Encoder
            # self.logger.add('encoder/latent_mean', self.policy_storage._latent_traj.mean())

            # Environment
            # Log exploration rewards, not env rewards
            exploration_return_per_trajectory = self.policy_storage.rewards_exp.sum(dim=-2)
            exploration_return_per_trajectory = exploration_return_per_trajectory[:, :, 1:]
            exploration_return_per_trajectory = exploration_return_per_trajectory.mean(dim=-1)
            self.logger.add('environment/exploration_return_max', exploration_return_per_trajectory.max())
            self.logger.add('environment/exploration_return_min', exploration_return_per_trajectory.min())
            self.logger.add('environment/exploration_return_mean', exploration_return_per_trajectory.mean())
            self.logger.add('environment/exploration_rew_max', self.policy_storage.rewards_exp.max())
            self.logger.add('environment/exploration_rew_min', self.policy_storage.rewards_exp.min())
            self.logger.add('environment/exploration_rew_mean', self.policy_storage.rewards_exp.mean())

            p = self.policy_storage.rewards_exp.shape[0]
            m = self.policy_storage.rewards_exp.shape[-1]
            H = self.policy_storage.rewards_exp.shape[-2]
            env_return_per_trajectory = self.policy_storage.rewards_raw.reshape(p, -1, m, H)  # (p, q, m, H)
            env_return_per_trajectory = env_return_per_trajectory[:, :, 1:].sum(dim=-1)
            env_return_per_trajectory = env_return_per_trajectory.mean(dim=-1)
            self.logger.add('environment/env_return_mean', env_return_per_trajectory.mean())
            self.logger.add('environment/env_rew_mean', self.policy_storage.rewards_raw.mean())

            self.logger.add('environment/state_max', self.policy_storage.prev_state.max())
            self.logger.add('environment/state_min', self.policy_storage.prev_state.min())
            self.logger.add('environment/state_mean', self.policy_storage.prev_state.mean())

            # FIXME This only makes sense for the MEWA env
            if 'MEWA' or 'MuJoCo' in self.args.env_name:
                actions = self.policy_storage._actions.argmax(dim=2).float()
                self.logger.add('environment/action_max', actions.max())
                self.logger.add('environment/action_min', actions.min())
                self.logger.add('environment/action_mean', actions.mean())

            self.logger.add('environment/len_max', self.policy_storage.lens.max())
            self.logger.add('environment/len_min', self.policy_storage.lens.min())
            self.logger.add('environment/len_mean', self.policy_storage.lens.mean())
            self.logger.add('environment/len_std', self.policy_storage.lens.std())

            # Exploration policy
            for k, v in policy_train_stats.items():
                self.logger.add(f'exploration_{k}', v)

            if hasattr(self.exploration_policy.actor_critic, 'logstd'):
                self.logger.add('exploration_policy/action_logstd',
                                self.exploration_policy.actor_critic.dist.logstd.mean())
            self.logger.add('exploration_policy/action_logprob', self.policy_storage.action_log_probs.mean())
            action_prob = torch.pow(2, self.policy_storage.action_log_probs)
            self.logger.add('exploration_policy/action_prob', action_prob.mean())
            self.logger.add('exploration_policy/action_prob_std', action_prob.std())
            self.logger.add('exploration_policy/action_prob_min', action_prob.min())
            self.logger.add('exploration_policy/action_prob_max', action_prob.max())
            quantiles = [0.001, 0.01, 0.05]
            for qant in quantiles:
                self.logger.add(f'exploration_policy/action_prob_quantile_{qant}', action_prob.quantile(qant))

            self.logger.add('timer/exploration_policy_update', round(self.timer_exploration_policy_update, 2))
