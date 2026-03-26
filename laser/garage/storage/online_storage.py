"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""
import torch
from laser.garage.torch._functions import global_device
from laser.garage.utils import helpers as utl
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])


class OnlineStorage(object):
    def __init__(self,
                 args,
                 num_tasks, meta_traj_per_task, traj_per_meta_traj, horizon, num_processes,
                 state_dim,
                 action_space,
                 latent_dim,
                 normalise_rewards,
                 use_exploration_rewards,
                 task_context_dim=None,
                 use_meta_entropy=False,
                 gamma=None,
                 tau=None,
                 ):
        self.args = args
        self.state_dim = state_dim

        # Number of meta-trajectories per update (from different tasks): tasks_per_iter * num_processes * q
        self.num_processes = num_processes  # number of parallel processes
        self.num_tasks = num_tasks  # p
        self.meta_traj_per_task = meta_traj_per_task  # q
        self._traj_per_meta_traj = traj_per_meta_traj  # m
        self._horizon = horizon  # H
        self._meta_traj_len = self._traj_per_meta_traj * self._horizon  # m*H
        self._batch = self.num_tasks * self.meta_traj_per_task  # p*q
        # self._meta_traj_len_ex = self._meta_traj_len + self._traj_per_meta_traj  # m*(H+1)
        self._use_exploration_rewards = use_exploration_rewards
        self._use_task_context = task_context_dim is not None
        self._task_context_dim = task_context_dim
        self._meta_entropy_coeff = self.args.meta_entropy_coeff if use_meta_entropy else 0
        self._exploit_tradeoff = self.args.exploration_exploit_tradeoff if self._use_exploration_rewards else None
        self._gamma = gamma
        self._tau = tau

        self._task_step = 0  # keep track of the current task

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards

        # inputs to the policy
        # this will include s_0 when state was reset
        self._prev_state = torch.zeros(self.num_tasks, self.meta_traj_per_task, state_dim,
                                       self._meta_traj_len)  # (p, q, d_obs, m*H)
        # self.prev_state = torch.zeros(self.num_batches, state_dim, self._meta_traj_len_ex)  # (p*q, d_obs, m*(H+1))

        if self.args.pass_latent_to_policy:
            # latent variables (of VAE)
            self.latent_dim = latent_dim
            self._latent_traj = torch.zeros(self.num_tasks, self.meta_traj_per_task, latent_dim,
                                            self._meta_traj_len)  # (p, q, d_z, m*H)
            # next_state will include s_N when state was reset, skipping s_0
            # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
            # self.next_state = torch.zeros(self.num_batches, state_dim, self._meta_traj_len)  # (p*q, d_obs, m*H)
        else:
            self._latent_traj = None

        # rewards and end of episodes
        self.rewards_raw = torch.zeros(self.num_tasks, self.meta_traj_per_task, 1,
                                       self._meta_traj_len)  # (p, q, 1, m*H)
        self.rewards_normalised = None
        if self.normalise_rewards:
            self.rewards_normalised = torch.zeros(self.num_tasks, self.meta_traj_per_task, 1,
                                                  self._meta_traj_len)  # (p, q, 1, m*H)
        if self._use_exploration_rewards:
            self.rewards_exp = torch.zeros(self.num_tasks, self.meta_traj_per_task, self._horizon,
                                           self._traj_per_meta_traj)  # (p, q, H, m)
        else:
            self.rewards_exp = None

        # Mask trajectories. A mask element is 1 if the corresponding trajectory element is padding. Otherwise, 0
        self.masks = torch.zeros(self.num_tasks, self.meta_traj_per_task,
                                 self._traj_per_meta_traj, self._horizon)  # (p, q, m, H)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            self.action_shape = 1
        else:
            self.action_shape = action_space.shape[0]
        self._actions = torch.zeros(self.num_tasks, self.meta_traj_per_task, self.action_shape,
                                    self._meta_traj_len)  # (p, q, d_act, m*H)
        if action_space.__class__.__name__ == 'Discrete':
            self._actions = self._actions.long()
        self.action_log_probs = None

        # The (unpadded) lengths of trajectories
        self._lens = torch.zeros(self.num_tasks, self.meta_traj_per_task, self._traj_per_meta_traj)  # (p, q, m)
        self._lens = self._lens.long()

        # values and returns
        self._value_preds \
            = torch.zeros(self.num_tasks, self.meta_traj_per_task, 1, self._meta_traj_len)  # (p, q, 1, m*H)
        self.returns = torch.zeros(self.num_tasks, self.meta_traj_per_task, 1, self._meta_traj_len)  # (p, q, 1, m*H)

        # Used for computing the meta-entropy advantages
        self._value_no_context = None
        if use_meta_entropy:
            self.meta_entropy_returns = torch.zeros(self.num_tasks, self.meta_traj_per_task, 1,
                                                    self._meta_traj_len)  # (p, q, 1, m*H)
            assert self._gamma is not None and self._tau is not None

        if self._use_task_context:
            self._task_context \
                = torch.zeros(self.num_tasks, self.meta_traj_per_task, self._task_context_dim)  # (p, q, H*d_z*m')
        else:
            self._task_context = None

        self.to_device()

    def to_device(self):
        if self.args.pass_state_to_policy:
            self._prev_state = self._prev_state.to(global_device())
        if self.args.pass_latent_to_policy:
            self._latent_traj = self._latent_traj.to(global_device())
            # self.next_state = self.next_state.to(global_device())
        self.rewards_raw = self.rewards_raw.to(global_device())
        if self.normalise_rewards:
            self.rewards_normalised = self.rewards_normalised.to(global_device())
        if self._use_exploration_rewards:
            self.rewards_exp = self.rewards_exp.to(global_device())
        self.masks = self.masks.to(global_device())
        self._value_preds = self._value_preds.to(global_device())
        self.returns = self.returns.to(global_device())
        if self._meta_entropy_coeff != 0:
            self.meta_entropy_returns = self.meta_entropy_returns.to(global_device())
        self._actions = self._actions.to(global_device())
        if self._use_task_context:
            self._task_context = self._task_context.to(global_device())
        self._lens = self._lens.to(global_device())

    def insert_meta_traj(self, meta_traj, task_idx, meta_traj_idx, lens=None, norm_rewards=None):
        """Insert a meta-trajectory of size (num_processes, d, m*H), where each step is ((obs+done), act, rew)"""
        states, actions, rewards = meta_traj
        self._prev_state[task_idx:task_idx + self.num_processes, meta_traj_idx].copy_(states)
        self._actions[task_idx:task_idx + self.num_processes, meta_traj_idx].copy_(actions.detach())
        self.rewards_raw[task_idx:task_idx + self.num_processes, meta_traj_idx].copy_(rewards)
        if self.normalise_rewards:
            self.rewards_normalised[task_idx:task_idx + self.num_processes, meta_traj_idx].copy_(norm_rewards)
        if lens is not None:
            self._lens[task_idx:task_idx + self.num_processes, meta_traj_idx].copy_(lens)

    def insert_latent(self, latent, task_idx, meta_traj_idx):
        """Insert the trajectory-specific latent for a single meta-trajectory"""
        if self.args.pass_latent_to_policy:
            self._latent_traj[task_idx:task_idx + self.num_processes, meta_traj_idx].copy_(latent.detach())

    def insert_task_context(self, task_context, task_idx, meta_traj_idx):
        assert self._use_task_context
        self._task_context[task_idx:task_idx + self.num_processes, meta_traj_idx].copy_(task_context.detach())

    def insert_value(self,
                     value,
                     task_idx,
                     meta_traj_idx,
                     meta_step,
                     process=None,
                     infos=None,
                     ):
        if process is None:
            # FIXME Do this without for loop
            for proc in range(self.num_processes):
                if infos is None or 'padding' not in infos[proc] or not infos[proc]['padding']:
                    if isinstance(value, list):
                        self._value_preds[task_idx + proc, meta_traj_idx, :, meta_step[proc]] \
                            .copy_(value[0][proc].detach())
                    else:
                        self._value_preds[task_idx + proc, meta_traj_idx, :, meta_step[proc]] \
                            .copy_(value[proc].detach())
        else:
            self._value_preds[task_idx + process, meta_traj_idx, :, meta_step[process]].copy_(value[process].detach())

    def insert_exploration_rewards(self, rewards, task_idx, meta_traj_idx):
        """Update the rewards of the current meta-trajectory with the given rewards. The given tensor contains a single
        reward per trajectory, so replace the final trajectory reward with that, and set all other rewards to 0
        Args:
            :param rewards: (p, m) exploration rewards, one per trajectory (with m trajectories per task)
        """

        if self._use_exploration_rewards:
            traj_lens = self._lens[task_idx:task_idx + self.num_processes, meta_traj_idx, :]  # (p, m)

            env_rew = 0
            if self._exploit_tradeoff != 0.0:
                if not self.normalise_rewards:
                    env_rew = self.rewards_raw.clone()  # (p, q, 1, m*H)
                else:
                    env_rew = self.rewards_normalised.clone()  # (p, q, 1, m*H)
                env_rew = env_rew[task_idx:task_idx + self.num_processes, meta_traj_idx]  # (p, 1, m*H)
                env_rew = (env_rew
                           .reshape(-1, self._traj_per_meta_traj, self._horizon)  # (p, m, H)
                           .permute(0, 2, 1))  # (p, H, m)

            self.rewards_exp[task_idx:task_idx + self.num_processes, meta_traj_idx] *= 0
            self.rewards_exp[task_idx:task_idx + self.num_processes, meta_traj_idx] += self.args.exploration_step_reward
            self.rewards_exp[task_idx:task_idx + self.num_processes, meta_traj_idx] \
                += self._exploit_tradeoff * env_rew

            # FIXME Remove for loop
            for proc in range(self.num_processes):
                for m in range(traj_lens.shape[-1]):
                    if m == 0:
                        assert rewards[proc, m] == 0
                        self.rewards_exp[task_idx + proc, meta_traj_idx, traj_lens[proc, m] - 1, m] \
                            += self.args.exploration_first_traj_reward
                    else:
                        # Only 1 reward per trajectory
                        self.rewards_exp[task_idx + proc, meta_traj_idx, traj_lens[proc, m] - 1, m] \
                            += rewards[proc, m].clone()
                        # All future rewards for this episode are 0
                    self.rewards_exp[task_idx + proc, meta_traj_idx, traj_lens[proc, m]:, m] *= 0

                    self.masks[task_idx + proc, meta_traj_idx, m, :traj_lens[proc, m]] = 1

        else:
            if rewards is not None:
                raise ValueError('The policy storage is not expecting exploration rewards')

    def update_masks(self, task_idx, meta_traj_idx):
        """ Only call this if not using exploration rewards """
        if self._use_exploration_rewards:
            raise ValueError('Masks are automatically updated when using exploration rewards. You should not call this')

        traj_lens = self._lens[task_idx:task_idx + self.num_processes, meta_traj_idx, :]  # (p, m)
        for proc in range(self.num_processes):
            for m in range(traj_lens.shape[-1]):
                self.masks[task_idx + proc, meta_traj_idx, m, :traj_lens[proc, m]] = 1

    def compute_returns(self, use_gae, gamma, tau, value_no_context_preds=None):
        if self._use_exploration_rewards:
            rewards = self.rewards_exp.clone() \
                .permute(0, 1, 3, 2) \
                .reshape(self.num_tasks, self.meta_traj_per_task, -1).unsqueeze(2)  # (p, q, 1, m*H)

        else:
            if self.normalise_rewards:
                rewards = self.rewards_normalised.clone()
            else:
                # Rewards for a single trajectory (since the task policy only collects a single traj per meta-traj)
                rewards = self.rewards_raw.clone()  # (p, 1, 1, H)

        padding_mask = self.masks.reshape(self.num_tasks, self.meta_traj_per_task, -1).unsqueeze(2)  # (p, q, 1, m*H)

        if self._meta_entropy_coeff == 0:
            self._compute_returns(rewards=rewards, value_preds=self._value_preds,
                                  returns=self.returns, padding_mask=padding_mask,
                                  gamma=gamma, tau=tau, use_gae=use_gae)
        else:
            assert value_no_context_preds is not None
            self._value_no_context = value_no_context_preds

    def _compute_returns(self, rewards, value_preds, returns, padding_mask, gamma, tau, use_gae):
        if use_gae:
            # Use the forward-view TD(lambda) to compute advantages.
            # Tau gives the amount of future value to take into account
            gae = 0
            zero = torch.tensor(0, dtype=value_preds.dtype, device=global_device())
            for step in reversed(range(self._meta_traj_len)):
                mask = padding_mask[:, :, :, step]
                value = value_preds[:, :, :, step]

                # FIXME Maybe we can do this outside the for loop, for all steps
                # Compute the value at the next step (either timestep or episode)
                if step + 1 < self._meta_traj_len:
                    # Check if the next timestep is padding
                    next_mask = padding_mask[:, :, :, step + 1]

                    # If not padding, next step is next timestep. Else, next step is first timestep of next episode
                    next_meta_step = self._horizon * (1 + step // self._horizon)
                    next_step = next_mask * (step + 1) + (1 - next_mask) * next_meta_step  # (p, q, 1)
                    next_step = next_step.unsqueeze(-1).long()  # (p, q, 1, 1)

                    # If next episode doesn't exist (i.e. this is the last episode), replace the next step with padding
                    next_step = torch.where(condition=next_step < self._meta_traj_len,
                                            input=next_step,
                                            other=0)

                    # The next value is indexed by the next step. If the next step doesn't exist, then the value is 0
                    potential_next_value = torch.gather(value_preds, dim=-1, index=next_step)
                    next_value = torch.where(condition=next_step != 0,
                                             input=potential_next_value,
                                             other=zero)  # (p, q, 1, 1)
                    next_value = next_value[:, :, :, 0]  # (p, q, 1)
                else:
                    next_value = 0

                delta = rewards[:, :, :, step] + gamma * next_value - value

                # Only update non-padding entries for the gae
                gae_update = delta + gamma * tau * gae
                gae = mask * gae_update + (1 - mask) * gae

                # Only compute the return for non-padding entries
                returns[:, :, :, step] = gae + value  # (p, q, 1, 1)
                returns[:, :, :, step] *= mask
        else:
            raise NotImplementedError
            returns[-1] = next_value
            for meta_step in reversed(range(rewards.size(0))):
                returns[meta_step] = returns[meta_step + 1] * gamma * self.masks[meta_step + 1] + rewards[meta_step]

    def _compute_returns_entropy(self, rewards, value_preds, returns, padding_mask, gamma, tau, use_gae,
                                 value_no_context_preds, entropy_action_log):
        log_val_diffs = (self.args.meta_entropy_value_weight == 'exp_diff'
                         or self.args.meta_entropy_value_weight == 'tanh_diff')
        val_weights, values, meta_entropies = [], [], []
        val_diffs = [] if log_val_diffs else None

        if use_gae:
            # Use the forward-view TD(lambda) to compute advantages.
            # Tau gives the amount of future value to take into account
            gae = 0
            zero = torch.tensor(0, dtype=value_preds.dtype, device=global_device())
            for step in reversed(range(self._meta_traj_len)):
                mask = padding_mask[:, :, :, step]
                value = value_preds[:, :, :, step]
                value_no_context = value_no_context_preds[:, :, :, step]
                # FIXME Double check which is correct (probably the second option, since we care about the first action,
                #  but not about the action in the terminal state)
                # action_entropy = entropy_action_log[:, :, :, step - 1] if step > 0 else 0
                action_entropy = entropy_action_log[:, :, :, step] if step + 1 < self._meta_traj_len \
                    else torch.zeros(entropy_action_log.shape[:-1], device=global_device())

                # FIXME Maybe we can do this outside the for loop, for all steps
                # Compute the value at the next step (either timestep or episode)
                if step + 1 < self._meta_traj_len:
                    # Check if the next timestep is padding
                    next_mask = padding_mask[:, :, :, step + 1]

                    # If not padding, next step is next timestep. Else, next step is first timestep of next episode
                    next_meta_step = self._horizon * (1 + step // self._horizon)
                    next_step = next_mask * (step + 1) + (1 - next_mask) * next_meta_step  # (p, q, 1)
                    next_step = next_step.unsqueeze(-1).long()  # (p, q, 1, 1)

                    # If next episode doesn't exist (i.e. this is the last episode), replace the next step with padding
                    next_step = torch.where(condition=next_step < self._meta_traj_len,
                                            input=next_step,
                                            other=0)

                    # The next value is indexed by the next step. If the next step doesn't exist, then the value is 0
                    potential_next_value = torch.gather(value_preds, dim=-1, index=next_step)
                    next_value = torch.where(condition=next_step != 0,
                                             input=potential_next_value,
                                             other=zero)  # (p, q, 1, 1)
                    next_value = next_value[:, :, :, 0]  # (p, q, 1)
                else:
                    next_value = 0

                if self.args.meta_entropy_value_weight == 'ratio':
                    value_weight = value_no_context / (value + 1e-8)
                elif self.args.meta_entropy_value_weight == 'exp_diff':
                    diff = value_no_context - value
                    if self.args.meta_entropy_mask_diff:
                        diff = diff * mask
                    value_weight = torch.exp(self.args.meta_entropy_exp_coeff * diff) - (1 - self.args.meta_entropy_clip)
                    value_weight = torch.relu(value_weight)
                elif self.args.meta_entropy_value_weight == 'tanh_diff':
                    diff = value_no_context - value
                    if self.args.meta_entropy_mask_diff:
                        diff = diff * mask
                    # value_weight = torch.tanh(self.args.meta_entropy_exp_coeff * diff) + self.args.meta_entropy_clip
                    value_weight = torch.tanh(self.args.meta_entropy_exp_coeff * diff - self.args.meta_entropy_clip)
                    value_weight = torch.relu(value_weight)
                else:
                    raise ValueError(f'Unknown meta-entropy value weight type: {self.args.meta_entropy_value_weight}')
                value_weight *= mask

                if self.args.meta_entropy_equation == 'default':
                    meta_entropy = value_weight * action_entropy
                elif self.args.meta_entropy_equation == 'entropy_only':
                    meta_entropy = action_entropy
                elif self.args.meta_entropy_equation == 'weight_only':
                    meta_entropy = -1 * value_weight
                else:
                    raise ValueError(f'Unknown meta-entropy equation: {self.args.meta_entropy_equation}')

                delta = rewards[:, :, :, step] + gamma * next_value - value
                delta += self._meta_entropy_coeff * meta_entropy

                # Only update non-padding entries for the gae
                gae_update = delta + gamma * tau * gae
                gae = mask * gae_update + (1 - mask) * gae

                # Only compute the return for non-padding entries
                # FIXME Should the target returns use the entropy at the current timestep or future timestep?
                returns[:, :, :, step] = gae + value  # (p, q, 1, 1)
                returns[:, :, :, step] *= mask

                # Logging
                val_weights.append(value_weight)
                if log_val_diffs:
                    val_diffs.append(diff * mask)
                values.append(rewards[:, :, :, step] + gamma * next_value)
                meta_entropies.append(self._meta_entropy_coeff * meta_entropy)
        else:
            raise ValueError('Using meta-entropy without GAE')

        val_weights = torch.cat(val_weights, dim=0).unsqueeze(-1)
        val_diffs = torch.cat(val_diffs, dim=0).unsqueeze(-1) if val_diffs is not None else None
        values = torch.cat(values, dim=0).unsqueeze(-1)
        meta_entropies = torch.cat(meta_entropies, dim=0).unsqueeze(-1)
        return {'val_weights': val_weights,
                'val_diffs': val_diffs,
                'value': values,
                'meta_entropy': meta_entropies,
                }

    def _compute_advantages(self):
        advantages = self.returns[:, :, :, :-1] - self._value_preds[:, :, :, :-1]  # (p, q, 1, m*H-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return advantages

    def _recompute_advantages(self, policy):
        stats = None
        if self._meta_entropy_coeff != 0:
            latent \
                = utl.get_latent_for_policy(self.args,
                                            self._latent_traj[:, :, :, :-1] if self._latent_traj is not None else None)
            prev_state = self._prev_state[:, :, :, :-1].permute(0, 1, 3, 2)  # (p, q, m*H, d_obs)
            latent = latent.permute(0, 1, 3, 2) if latent is not None else None  # (p, q, m*H, d_z)
            actions = self._actions[:, :, :, 1:].permute(0, 1, 3, 2)  # (p, q, m*H, d_act)

            evaluated_actions = policy.actor_critic.evaluate_actions(prev_state, latent, actions, self.task_context,
                                                                     return_dist=True,
                                                                     )  # (p, q, m*H, 1)

            if self.args.meta_entropy_loss == 'entropy':
                _, action_log_probs, _, policy_dist = evaluated_actions
                entropy_action_log = policy_dist.entropy().unsqueeze(-1).permute(0, 1, 3, 2)  # (p, q, 1, m*H)
            elif self.args.meta_entropy_loss == 'neg_log':
                _, action_log_probs, _, policy_dist = evaluated_actions
                entropy_action_log = -1 * action_log_probs.permute(0, 1, 3, 2)  # (p, q, 1, m*H)
            elif self.args.meta_entropy_loss == 'kl_div':
                _, _, _, policy_dist = evaluated_actions
                with torch.no_grad():
                    _, _, _, policy_dist_no_context = policy.actor_critic.evaluate_actions(
                        prev_state, latent, actions, self.task_context * 0, return_dist=True)  # (p, q, m*H, 1)
                # kl_div = torch.distributions.kl.kl_divergence(policy_dist_no_context, policy_dist)  # (p, q, m*H)
                kl_div = torch.distributions.kl.kl_divergence(policy_dist, policy_dist_no_context)  # (p, q, m*H)
                entropy_action_log = kl_div.unsqueeze(-2)  # (p, q, 1, m*H)
            else:
                raise ValueError(f'Unknown meta-entropy loss: {self.args.meta_entropy_loss}')

            if self._use_exploration_rewards:
                rewards = self.rewards_exp.clone() \
                    .permute(0, 1, 3, 2) \
                    .reshape(self.num_tasks, self.meta_traj_per_task, -1).unsqueeze(2)  # (p, q, 1, m*H)

            else:
                if self.normalise_rewards:
                    rewards = self.rewards_normalised.clone()
                else:
                    # Rewards for a single trajectory (since the task policy only collects a single traj per meta-traj)
                    rewards = self.rewards_raw.clone()  # (p, 1, 1, H)

            padding_mask \
                = self.masks.reshape(self.num_tasks, self.meta_traj_per_task, -1).unsqueeze(2)  # (p, q, 1, m*H)

            # Delete the previous computation graph, to release memory
            del self.returns
            self.returns = torch.zeros(self.num_tasks, self.meta_traj_per_task, 1,
                                       self._meta_traj_len)  # (p, q, 1, m*H)
            self.returns = self.returns.to(global_device())

            stats = self._compute_returns_entropy(rewards=rewards, value_preds=self._value_preds,  # (p, q, 1, m*H)
                                                  returns=self.returns, padding_mask=padding_mask,
                                                  gamma=self._gamma, tau=self._tau,
                                                  use_gae=self.args.policy_use_gae,
                                                  value_no_context_preds=self._value_no_context,  # (p, q, 1, m*H)
                                                  entropy_action_log=entropy_action_log)  # (p, q, 1, m*H)
        return self._compute_advantages(), stats

    def num_transitions(self):
        return len(self._prev_state) * self.num_processes

    def before_update(self, policy):
        latent = utl.get_latent_for_policy(self.args,
                                           self._latent_traj[:, :, :, :-1] if self._latent_traj is not None else None)
        prev_state = self._prev_state[:, :, :, :-1].permute(0, 1, 3, 2)  # (p, q, m*H, d_obs)
        latent = latent.permute(0, 1, 3, 2) if latent is not None else None  # (p, q, m*H, d_z)
        actions = self._actions[:, :, :, 1:].permute(0, 1, 3, 2)  # (p, q, m*H, d_act)
        _, action_log_probs, _ = policy.evaluate_actions(prev_state, latent, actions, self._task_context)
        self.action_log_probs = action_log_probs.detach()  # (p, q, m*H, 1)

    def after_update(self):
        self._prev_state *= 0
        self._actions *= 0
        self.rewards_raw *= 0
        if self.normalise_rewards:
            self.rewards_normalised *= 0
        self._lens *= 0
        if self._use_exploration_rewards:
            self.rewards_exp *= 0
        if self.args.pass_latent_to_policy:
            self._latent_traj *= 0
        self.masks *= 0
        self.action_log_probs = None
        self._value_preds *= 0
        self._value_no_context = None
        self.returns *= 0
        if self._meta_entropy_coeff != 0:
            self.meta_entropy_returns *= 0
        self._task_step *= 0
        if self._use_task_context:
            self._task_context *= 0

    def is_empty(self):
        return self._prev_state.abs().sum() == 0 \
            and self._actions.abs().sum() == 0 \
            and self.rewards_raw.abs().sum() == 0 \
            and (not self._use_exploration_rewards or self.rewards_exp.abs().sum() == 0) \
            and (not self.normalise_rewards or self.rewards_normalised.abs().sum() == 0) \
            and (not self.args.pass_latent_to_policy or self._latent_traj.abs().sum() == 0) \
            and self.action_log_probs is None \
            and self._value_preds.abs().sum() == 0 \
            and self._value_no_context is None \
            and self.returns.abs().sum() == 0 \
            and (self._meta_entropy_coeff == 0 or self.meta_entropy_returns.abs().sum() == 0) \
            and self.masks.abs().sum() == 0 \
            and self._task_step == 0 \
            and (not self._use_task_context or self._task_context.abs().sum() == 0) \
            and self._lens.abs().sum() == 0

    def feed_forward_generator(self,
                               policy,
                               num_mini_batch=None,
                               mini_batch_size=None,
                               return_context=True,
                               ppo_epoch=0,
                               ):
        """Data generator that selects a sample of given size and reshapes the data in the form expected by PPO:
            (batch, dim_data)
        """
        batch_size = self.num_tasks * self.meta_traj_per_task * (
                self._traj_per_meta_traj * self._horizon - 1)  # p*q*(m*H-1)

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the total number of steps (p*q*(m*H-1)) = ({}*{}*({}*{}-1)) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(self.num_tasks, self.meta_traj_per_task, self._traj_per_meta_traj, self._horizon, batch_size,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        # FIXME Make sure we only use non-padding steps
        # raise NotImplementedError

        advantages, stats = self._recompute_advantages(policy)

        i = 0
        for indices in sampler:
            if self.args.pass_state_to_policy:
                state_batch = self._get_batch(self._prev_state[:, :, :, :-1],
                                              self.state_dim, indices)  # (len(indices), d_obs)
            else:
                state_batch = None

            if self.args.pass_latent_to_policy:
                latent_batch = self._get_batch(self._latent_traj[:, :, :, :-1],
                                               self.latent_dim, indices)  # (len(indices), d_z)
            else:
                latent_batch = None

            actions_batch = self._get_batch(self._actions[:, :, :, 1:],
                                            self.action_shape, indices)  # (len(indices), d_act)

            value_preds_batch = self._get_batch(self._value_preds[:, :, :, :-1], 1, indices)  # (len(indices), 1)
            return_batch = self._get_batch(self.returns.detach()[:, :, :, :-1], 1, indices)  # (len(indices), 1)

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]  # (len(indices), 1)
            if advantages is None:
                adv_targ, v_diffs = None, None
            else:
                if self._meta_entropy_coeff != 0 and (ppo_epoch > 0 or i > 0):
                    # We need fresh gradients, so recompute the advantages, using the updated policy
                    advantages, stats = self._recompute_advantages(policy)
                i += 1
                adv_targ = self._get_batch(advantages, 1, indices)  # (len(indices), 1)

                if stats is not None:
                    for k in stats.keys():
                        stats[k] = self._get_batch(stats[k], 1, indices) \
                            if stats[k] is not None else None  # (len(indices), 1)

            if return_context:
                if self._use_task_context:
                    # Each step in a trajectory uses the same context, so repeat it for the whole trajectory
                    task_context \
                        = self._task_context.unsqueeze(-1).repeat(1, 1, 1, self._horizon - 1)  # (p, q, d_ctx, H)
                    task_context_batch = self._get_batch(task_context,
                                                         self._task_context_dim, indices)  # (len(indices), d_ctx)
                else:
                    task_context_batch = None

                yield state_batch, actions_batch, latent_batch, \
                    value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, stats, task_context_batch
            else:
                yield state_batch, actions_batch, latent_batch, \
                    value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, stats, indices

    def _get_batch(self, data, dim_data, indices=None):
        data = data.permute(0, 1, 3, 2).reshape(-1, dim_data)  # (p*q*m*H, dim)
        if indices is None:
            return data
        return data[indices]  # (len(indices), dim)

    @property
    def task_step(self):
        return self._task_step

    @property
    def value_preds(self):
        return self._value_preds.reshape(self._batch, -1)  # (p*q, m*H)

    @property
    def prev_state(self):
        return self._prev_state

    @property
    def actions(self):
        # return self._actions.reshape(self._batch, self.action_shape, -1)  # (p*q, d_act, m*H)
        return self._actions.permute(0, 1, 3, 2).reshape(-1, self.action_shape)  # (p*q*m*H, d_act)

    @property
    def latent_traj(self):
        return self._latent_traj

    @property
    def task_context(self):
        return self._task_context

    @property
    def lens(self):
        return self._lens.float()
