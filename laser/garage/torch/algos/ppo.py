import copy
import warnings
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from laser.garage.torch._functions import global_device
from laser.garage.utils import helpers as utl


class PPO:
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 pfo_coef,
                 policy_optimiser,
                 policy_anneal_lr,
                 train_steps,
                 lr=None,
                 critic_lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 update_state_normaliser=False,
                 update_context_normaliser=False,
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.pfo_coef = pfo_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss
        self.huber_beta = self.args.ppo_huber_beta

        self.update_state_normaliser = update_state_normaliser
        self.update_context_normaliser = update_context_normaliser

        # optimiser
        self.lr = lr
        self.critic_lr = critic_lr
        if self.critic_lr is None:
            actor_params = actor_critic.parameters()
        else:
            actor_params = chain(*(actor_critic.actor_params() + actor_critic.encoder_params()))
            critic_params = chain(*actor_critic.critic_params())
            if policy_optimiser == 'adam':
                self.critic_optimiser = optim.Adam(critic_params, lr=critic_lr, eps=eps)
            elif policy_optimiser == 'rmsprop':
                self.critic_optimiser = optim.RMSprop(critic_params, lr=critic_lr, eps=eps, alpha=0.99)

        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_params, lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_params, lr=lr, eps=eps, alpha=0.99)

        self.lr_scheduler_policy = None
        self.lr_scheduler_encoder = None
        if policy_anneal_lr:
            if self.critic_lr is not None:
                raise NotImplementedError
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)
            # FIXME Move this to transformer.py
            if hasattr(self.args, 'rlloss_through_encoder') and self.args.rlloss_through_encoder:
                self.lr_scheduler_encoder = optim.lr_scheduler.LambdaLR(self.optimiser_vae, lr_lambda=lam)

    def update(self,
               policy_storage,
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               exploration_trajectories=None,
               transformer=None,  # encoder
               requires_latent=False,
               requires_context=False,
               use_pfo=False,
               ):
        # if this is true, we will update the VAE at every PPO update
        # otherwise, we update it after we update the policy
        if rlloss_through_encoder:
            assert exploration_trajectories is not None and transformer is not None
            # recompute embeddings
            # (to build computation graph, because the original embeddings were computed with torch.no_grad())
            task_context = self._recompute_embeddings(policy_storage, exploration_trajectories, transformer,
                                                      update_idx=0)
            horizon = policy_storage.prev_state.shape[-1]
            task_context = task_context.unsqueeze(-1).repeat(1, 1, 1, horizon)

        if self.update_state_normaliser:
            utl.update_state_rms(policy_storage.prev_state, policy_storage.lens, self.args)
        if self.update_context_normaliser:
            self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        # Keep a copy of the policy used to collect the data
        # FIXME We can make this better by storing the pre-computed features during data collection ==> no need to
        #  clone the policy
        collection_actor_critic = None
        if use_pfo:
            collection_actor_critic = copy.deepcopy(self.actor_critic)

        value_loss_epoch, action_loss_epoch, dist_entropy_epoch, pfo_loss_epoch, loss_epoch = 5 * (0,)
        ratio_mean, ratio_min, ratio_max = 3 * (0,)
        val_weights_mean, val_weights_min, val_weights_max = 3 * (None,)
        val_diffs_mean, val_diffs_min, val_diffs_max = 3 * (None,)
        value_mean, value_min, value_max = 3 * (None,)
        meta_entropy_mean, meta_entropy_min, meta_entropy_max = 3 * (None,)
        for e in range(self.ppo_epoch):
            data_generator = policy_storage.feed_forward_generator(policy=self,
                                                                   num_mini_batch=self.num_mini_batch,
                                                                   return_context=not rlloss_through_encoder,
                                                                   ppo_epoch=e)
            first_sample = True
            i = 0
            for sample in data_generator:
                if not rlloss_through_encoder:
                    # Get all data from the data generator
                    state_batch, actions_batch, latent_batch, \
                        value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, stats, \
                        task_context_batch = sample
                else:
                    # Get most data from the data generator, but use a batch of task context that has computed gradients
                    state_batch, actions_batch, latent_batch, \
                        value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ, stats, \
                        indices = sample
                    task_context_batch = task_context.permute(0, 1, 3, 2).reshape(-1, policy_storage._task_context_dim)
                    # The context batch should match the batch of the other data (given by indices)
                    task_context_batch = task_context_batch[indices]

                if requires_latent:
                    assert latent_batch is not None
                if requires_context:
                    assert task_context_batch is not None
                else:
                    assert task_context_batch is None, task_context_batch

                if not rlloss_through_encoder:
                    state_batch = state_batch.detach()
                    if latent_batch is not None:
                        latent_batch = latent_batch.detach()
                    if task_context_batch is not None:
                        task_context_batch = task_context_batch.detach()

                latent_batch = utl.get_latent_for_policy(args=self.args, latent=latent_batch)

                # FIXME Maybe use these action_log_probs when computing the meta-entropy GAE?
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, pre_activations = \
                    self.actor_critic.evaluate_actions(state=state_batch,
                                                       latent=latent_batch,
                                                       action=actions_batch,
                                                       task_context=task_context_batch,
                                                       use_pre_activations=True)  # (batch, 1), (batch, 1), (0), ()

                # adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-8)

                # ratio = torch.exp(action_log_probs -
                #                   old_action_log_probs_batch)
                # surr1 = -adv_targ * ratio
                # surr2 = -adv_targ * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                # action_loss = torch.max(surr1, surr2).mean()

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                using_rms = self.update_state_normaliser or self.update_context_normaliser
                if (e == 0 and i == 0 and not torch.all(torch.isclose(ratio, torch.tensor(1.0), atol=1e-4))
                       and not rlloss_through_encoder and not using_rms):
                   warnings.warn(f'Expected the PPO policy ratio to be 1 in the first iteration'
                                 f'\n{ratio}\n{ratio.min()}\n{ratio.max()}')
                i += 1

                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none', beta=self.huber_beta)
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none',
                                                            beta=self.huber_beta)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch, beta=self.huber_beta)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Compute the PFO loss on the pre-activations of the penultimate policy layer, as in
                # https://arxiv.org/pdf/2405.00662
                if use_pfo:
                    with torch.no_grad():
                        _, _, old_pre_activations = collection_actor_critic(state=state_batch,
                                                                            latent=latent_batch,
                                                                            task_context=task_context_batch,
                                                                            use_pre_activations=True)
                    pfo_loss = torch.linalg.vector_norm(pre_activations - old_pre_activations)
                else:
                    pfo_loss = torch.zeros(1, device=global_device())

                # zero out the gradients
                self.optimiser.zero_grad()
                if self.critic_lr is not None:
                    self.critic_optimiser.zero_grad()
                if rlloss_through_encoder:
                    transformer.optimiser_transformer.zero_grad()

                # compute policy loss and backprop
                loss = value_loss * self.value_loss_coef + \
                       action_loss - \
                       dist_entropy * self.entropy_coef + \
                       pfo_loss * self.pfo_coef

                # compute gradients (will attach to all networks involved in this computation)
                loss.backward()

                # clip gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)

                # update
                self.optimiser.step()
                if self.critic_lr is not None:
                    self.critic_optimiser.step()
                if rlloss_through_encoder:
                    if self.args.encoder_max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(transformer.parameters(), self.args.encoder_max_grad_norm)
                    # update
                    transformer.optimiser_transformer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                pfo_loss_epoch += pfo_loss.item()
                loss_epoch += loss.item()

                if e > 0 or not first_sample:
                    ratio_mean += ratio.mean()
                    ratio_min += ratio.min()
                    ratio_max += ratio.max()
                first_sample = False

                if stats is not None:
                    if stats['val_weights'] is not None:
                        if val_weights_mean is None:
                            val_weights_mean, val_weights_min, val_weights_max = 3 * (0,)
                        val_weights_mean += stats['val_weights'].mean()
                        val_weights_min += stats['val_weights'].min()
                        val_weights_max += stats['val_weights'].max()

                    if stats['val_diffs'] is not None:
                        if val_diffs_mean is None:
                            val_diffs_mean, val_diffs_min, val_diffs_max = 3 * (0,)
                        val_diffs_mean += stats['val_diffs'].mean()
                        val_diffs_min += stats['val_diffs'].min()
                        val_diffs_max += stats['val_diffs'].max()

                    if stats['value'] is not None:
                        if value_mean is None:
                            value_mean, value_min, value_max = 3 * (0,)
                        value_mean += stats['value'].mean()
                        value_min += stats['value'].min()
                        value_max += stats['value'].max()

                    if stats['meta_entropy'] is not None:
                        if meta_entropy_mean is None:
                            meta_entropy_mean, meta_entropy_min, meta_entropy_max = 3 * (0,)
                        meta_entropy_mean += stats['meta_entropy'].mean()
                        meta_entropy_min += stats['meta_entropy'].min()
                        meta_entropy_max += stats['meta_entropy'].max()

                if rlloss_through_encoder:
                    # recompute embeddings
                    # (to build computation graph, because the original embeddings were computed with torch.no_grad())
                    task_context = self._recompute_embeddings(policy_storage, exploration_trajectories, transformer,
                                                              update_idx=e + 1)
                    horizon = policy_storage.prev_state.shape[-1]
                    task_context = task_context.unsqueeze(-1).repeat(1, 1, 1, horizon)

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()
        if self.lr_scheduler_encoder is not None:
            self.lr_scheduler_encoder.step()

        last_lr = self.lr if self.lr_scheduler_policy is None else self.lr_scheduler_policy.get_last_lr()[0]

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        pfo_loss_epoch = (pfo_loss_epoch / num_updates) if use_pfo else None
        loss_epoch /= num_updates

        ratio_mean /= num_updates
        ratio_min /= num_updates
        ratio_max /= num_updates

        if val_weights_mean is not None:
            val_weights_mean /= num_updates
            val_weights_min /= num_updates
            val_weights_max /= num_updates

        if val_diffs_mean is not None:
            val_diffs_mean /= num_updates
            val_diffs_min /= num_updates
            val_diffs_max /= num_updates

        if value_mean is not None:
            value_mean /= num_updates
            value_min /= num_updates
            value_max /= num_updates

        if meta_entropy_mean is not None:
            meta_entropy_mean /= num_updates
            meta_entropy_min /= num_updates
            meta_entropy_max /= num_updates

        return self._get_train_stats(value_loss_epoch, action_loss_epoch, dist_entropy_epoch, pfo_loss_epoch,
                                     loss_epoch, last_lr, (ratio_mean, ratio_min, ratio_max),
                                     (val_weights_mean, val_weights_min, val_weights_max),
                                     (val_diffs_mean, val_diffs_min, val_diffs_max),
                                     (value_mean, value_min, value_max),
                                     (meta_entropy_mean, meta_entropy_min, meta_entropy_max),
                                     )

    def act(self, state, latent, task_context=None, deterministic=False):
        return self.actor_critic.act(state=state, latent=latent, task_context=task_context, deterministic=deterministic)

    def encode_task_context(self, task_context, state_shape):
        return self.actor_critic.encode_task_context(task_context, state_shape)

    def _recompute_embeddings(self, policy_storage, exploration_trajectories, encoder, update_idx):
        return utl.recompute_embeddings(self.args, policy_storage, exploration_trajectories, encoder,
                                        update_idx=update_idx)

    def save(self, save_path):
        torch.save(self.actor_critic.state_dict(), save_path)

    def _get_train_stats(self, value_loss, action_loss, dist_entropy, pfo_loss, loss, last_lr, ratios,
                         val_weights, val_diffs, value, meta_entropy):
        stats = {
            'policy_losses/value_loss': value_loss,
            'policy_losses/action_loss': action_loss,
            'policy_losses/dist_entropy': dist_entropy,
            'policy_losses/sum': loss,
            'policy/lr': last_lr,
            'policy/ratio_mean': ratios[0],
            'policy/ratio_min': ratios[1],
            'policy/ratio_max': ratios[2],
        }
        if pfo_loss is not None:
            stats['policy_losses/pfo_loss'] = pfo_loss
        if val_weights[0] is not None:
            stats['policy/val_weights_mean'] = val_weights[0]
            stats['policy/val_weights_min'] = val_weights[1]
            stats['policy/val_weights_max'] = val_weights[2]
        if val_diffs[0] is not None:
            stats['policy/val_diffs_mean'] = val_diffs[0]
            stats['policy/val_diffs_min'] = val_diffs[1]
            stats['policy/val_diffs_max'] = val_diffs[2]
        if value[0] is not None:
            stats['policy/value_mean'] = value[0]
            stats['policy/value_min'] = value[1]
            stats['policy/value_max'] = value[2]
        if meta_entropy[0] is not None:
            stats['policy/meta_entropy_mean'] = meta_entropy[0]
            stats['policy/meta_entropy_min'] = meta_entropy[1]
            stats['policy/meta_entropy_max'] = meta_entropy[2]
        return stats
