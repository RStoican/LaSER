import time

import torch

from laser.garage.utils import helpers as utl


class TaskPolicyTrainer:
    def __init__(self, args, policy, policy_storage):
        self.args = args

        self._policy = policy
        self._policy_storage = policy_storage

        self._traj_per_meta_traj = self.args.traj_per_meta_traj  # m
        self._traj_len = self.args.horizon  # H

        # Used only when we fine-tune the transformer with the RL loss
        self._exploration_trajectories = None
        self._transformer = None

    @property
    def policy(self):
        return self._policy

    @property
    def policy_storage(self):
        return self._policy_storage

    def train_policy(self, iter_idx):
        start_time = time.time()
        # FIXME Use this:
        # if iter_idx >= self.args.transformer_pre_train_epochs and iter_idx > 0:

        if self.args.use_meta_entropy:
            latent = utl.get_latent_for_policy(self.args,
                                               self._policy_storage.latent_traj
                                               if self._policy_storage.latent_traj is not None else None)
            prev_state = self._policy_storage.prev_state.permute(0, 1, 3, 2)  # (p, q, m*H, d_obs)
            latent = latent.permute(0, 1, 3, 2) if latent is not None else None  # (p, q, m*H, d_z)
            task_context = self._policy_storage.task_context  # (p, q, H*d_z*m')

            with torch.no_grad():
                # FIXME Compare these with the ones computed online in metalearner_task
                value_no_context, _ = self._policy.actor_critic.forward(
                    state=prev_state,
                    latent=(latent * 0) if latent is not None else None,
                    task_context=(task_context * 0) if task_context is not None else None,
                )  # (p, q, m*H, 1)
                value_no_context = value_no_context.permute(0, 1, 3, 2)  # (p, q, 1, m*H)
        else:
            value_no_context = None

        # compute returns for current rollouts
        self._policy_storage.compute_returns(self.args.policy_use_gae, self.args.task_policy_gamma,
                                             self.args.task_policy_tau,
                                             value_no_context_preds=value_no_context)

        # update agent
        train_stats = self._policy.update(
            policy_storage=self._policy_storage,
            rlloss_through_encoder=self.args.finetune_to_task,
            exploration_trajectories=self._exploration_trajectories,
            transformer=self._transformer,
            requires_latent=self.args.ablation_use_history,
            requires_context=self.args.ablation_use_context,
            use_pfo=self.args.ppo_task_pfo,
        )

        # FIXME Make this algorithm agnostic
        # self._print(train_stats)
        return train_stats, time.time() - start_time

    def create_rewards(self, traj_latent, num_tasks):
        return None

    def setup_finetune(self, exploration_trajectories, transformer):
        self._exploration_trajectories = exploration_trajectories
        self._transformer = transformer

    def _print(self, train_stats):
        # FIXME Make this algorithm agnostic
        if train_stats is not None:
            print(f'Policy Losses')
            print(f'     Value Loss: {train_stats[0]}')
            print(f'     Action Loss: {train_stats[1]}')
            print(f'     Dist Entropy: {train_stats[2]}')
            print(f'     Sum: {train_stats[3]}')
