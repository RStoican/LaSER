import math
import time

import torch
from laser.garage.torch._functions import global_device
from laser.garage.utils import helpers as utl


class ExplorationPolicyTrainer:
    def __init__(self, args, policy, policy_storage):
        self.args = args

        self._policy = policy
        self._policy_storage = policy_storage

        self._traj_per_meta_traj = self.args.traj_per_meta_traj  # m
        self._traj_len = self.args.horizon  # H

    @property
    def policy(self):
        return self._policy

    @property
    def policy_storage(self):
        return self._policy_storage

    def train_policy(self, iter_idx):
        start_time = time.time()
        train_stats = None
        if iter_idx >= self.args.transformer_pre_train_epochs and iter_idx > 0:
            if self.args.use_meta_entropy:
                latent = utl.get_latent_for_policy(self.args,
                                                   self._policy_storage.latent_traj
                                                   if self._policy_storage.latent_traj is not None else None)
                prev_state = self._policy_storage.prev_state.permute(0, 1, 3, 2)  # (p, q, m*H, d_obs)
                latent = latent.permute(0, 1, 3, 2) if latent is not None else None  # (p, q, m*H, d_z)

                with torch.no_grad():
                    # FIXME Compare these with the ones computed online in metalearner_task
                    value_no_context, _ = self._policy.actor_critic.forward(
                        state=prev_state,
                        latent=(latent * 0) if latent is not None else None,
                        task_context=None,
                    )  # (p, q, m*H, 1)
                    value_no_context = value_no_context.permute(0, 1, 3, 2)  # (p, q, 1, m*H)
            else:
                value_no_context = None

            # compute returns for current rollouts
            self._policy_storage.compute_returns(self.args.policy_use_gae, self.args.exploration_policy_gamma,
                                                 self.args.exploration_policy_tau,
                                                 value_no_context_preds=value_no_context)

            # update agent
            train_stats = self._policy.update(
                policy_storage=self._policy_storage,
                # TODO Consider using the exploration RL loss to update the transformer
                rlloss_through_encoder=False,
                # FIXME Use args to set requires_latent (requires_context is always False)
                requires_latent=True,
                requires_context=False,
                use_pfo=self.args.ppo_exp_pfo,
            )

        # FIXME Make this algorithm agnostic
        # self._print(train_stats)
        return train_stats, time.time() - start_time

    def _get_value(self, state, latent):
        latent = utl.get_latent_for_policy(self.args, latent=latent)
        if state.shape[0] == 1:
            state = state.squeeze(0)
        return self._policy.actor_critic.get_value(state=state, latent=latent).detach()

    def create_rewards(self, traj_latent, num_tasks):
        """Create intrinsic rewards for exploration
        Args:
            :param traj_latent: (p, d_model, m*H) trajectory-specific latents
            :param num_tasks: the number of tasks p

        Returns:
            :return rewards: (p, 1, m*H) new task-agnostic intrinsic rewards containing meta-learned exploration
            information
        """
        return self._create_exploration_rewards(traj_latent.clone(), num_tasks)

    def _create_exploration_rewards(self, traj_latent, num_tasks):
        # Get both the rewards and latent as a collection of m trajectories of H steps each
        traj_latent = utl.step_to_traj(traj_latent, self._traj_len)  # (p', H*d_z, m)

        # FIXME Create method
        if self.args.exploration_similarity_measure == 'dot_product':
            # Given the latent of all the trajectories used, find their pairwise dot product
            exploration_sim = torch.matmul(traj_latent.permute(0, 2, 1), traj_latent)  # (p, m, m)

        elif self.args.exploration_similarity_measure == 'cosine_similarity':
            # Given the latent of all the trajectories used, find their cosine similarity
            traj_latent_transpose = traj_latent.permute(0, 2, 1)
            exploration_sim = torch.nn.functional.cosine_similarity(traj_latent_transpose.unsqueeze(1),
                                                                    traj_latent_transpose.unsqueeze(2),
                                                                    dim=-1)  # (p, m, m)

        else:
            raise ValueError(f'Unknown similarity measure: {self.args.exploration_similarity_measure}')

        # Create a mask that hides the dot-product between a trajectory and all future trajectories (unfeasible in an
        # online setting). Also, we don't care about the dot product between a trajectory with itself, so hide that too
        traj_mask = self._create_traj_mask(self._traj_per_meta_traj, num_tasks)  # (p, m, m)
        masked_traj_dot_prod = exploration_sim * traj_mask

        # For each trajectory, combine the computed dot products into a reward
        if self.args.exploration_avg_rewards:
            traj_weight = masked_traj_dot_prod.sum(dim=-1)  # (p, m)
            traj_weight /= traj_mask.sum(dim=-1)
            traj_weight = traj_weight.nan_to_num(nan=0, posinf=0)
        else:
            traj_weight = masked_traj_dot_prod.sum(dim=-1)  # (p, m)

        rewards = self._compute_rewards(traj_weight)  # (p, m)
        # The reward for the first trajectory is always 0
        rewards[:, 0] = 0
        return rewards

    def _compute_rewards(self, traj_weight):
        if self.args.exploration_reward == 'gauss':
            # Make sure that values close to 0 give the largest reward
            exploration_rewards = traj_weight.pow(2) / pow(self.args.exploration_sigma, 2)
            exploration_rewards *= -1
            exploration_rewards /= 2
            exploration_rewards = torch.exp(exploration_rewards)

            if self.args.normalise_exploration_rewards:
                norm_coeff = 1 / (self.args.exploration_sigma * math.sqrt(2 * torch.pi))
                return norm_coeff * exploration_rewards
            return exploration_rewards

        elif self.args.exploration_reward == 'square':
            return self.args.exploration_sigma * traj_weight.pow(2)

        else:
            raise ValueError(f'Unrecognized exploration reward {self.args.exploration_reward}')

    # Create a mask where the top-right triangle, including the main diagonal, is 0. Otherwise, it's 1
    def _create_traj_mask(self, size, batch_size):
        ones = torch.ones(size, size).to(global_device())  # (s, s)
        mask = torch.triu(ones).transpose(0, 1)  # (s, s)
        mask[range(size), range(size)] = 0
        return mask.unsqueeze(0).repeat(batch_size, 1, 1)  # (b, s, s)

    def _print(self, train_stats):
        # FIXME Make this algorithm agnostic
        if train_stats is not None:
            print(f'Policy Losses')
            print(f'     Value Loss: {train_stats[0]}')
            print(f'     Action Loss: {train_stats[1]}')
            print(f'     Dist Entropy: {train_stats[2]}')
            print(f'     Sum: {train_stats[3]}')
