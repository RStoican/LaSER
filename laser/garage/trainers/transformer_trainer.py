import copy

import numpy as np
import torch
from laser.garage.torch.modules.mlp_module import _get_activation_fn
from laser.garage.utils import helpers as utl
from laser.garage.utils.data_masker import DataMasker
from laser.garage.utils.helpers import step_to_traj, traj_to_step


class TransformerTrainer:
    def __init__(self,
                 args,
                 transformer,
                 target_coeff_head,
                 get_iter_idx,
                 logger,
                 ):
        self.args = args
        self.transformer = transformer
        self.target_coeff_head = target_coeff_head
        self.get_iter_idx = get_iter_idx
        self.logger = logger

        if self.args.task_reconstruction_type == 'non-linear':
            self.nonlinear_activation = _get_activation_fn(self.args.task_reconstruction_activation)

        self.data_masker = DataMasker(mask_prob=self.args.mask_prob,
                                      mask_prob_token=self.args.mask_prob_token,
                                      mask_prob_replace=self.args.mask_prob_replace,
                                      mask_token=self.args.mask_token,
                                      mask_latest_action=self.args.mask_latest_action,
                                      obs_dim=self.args.state_dim, act_dim=self.args.action_dim,
                                      )

    def check_target_coeff_head_update(self):
        update_iter = self.args.target_net_update_iterations == 0 \
                      or self.get_iter_idx() % self.args.target_net_update_iterations == 0
        if update_iter and self.target_coeff_head is not None:
            self.target_coeff_head.load_state_dict(copy.deepcopy(self.transformer.linear_coeff_head.state_dict()))

    def update(self, log=False):
        # Get a batch of data
        tasks_batch = self.transformer.dataset_storage.get_batch(batch_size=self.args.batch_size,
                                                                 lens=self.args.norm_state_exploration)  # (p, q, d, m*H)

        # First, update the normalisation parameters of the state inputs
        if self.args.norm_state_exploration:
            tasks_batch, lens = tasks_batch
            utl.update_state_rms(tasks_batch, lens, self.args)

        # Randomly mask some of the steps (i.e. so the model learns to predict them)
        masked_tasks_batch, mask = self.data_masker.mask_input_traj(tasks_batch)  # (p, q, d, m*H)

        # (p, q, H*d_z, m), (H*d_z), (p, q, H*d_z), (p, q, H*d_z, m), (p, q, d, m*H)
        latent, shared_latent, task_latent, traj_latent, reconstructed_tasks_batch \
            = self._run_encoder(masked_tasks_batch, tasks_batch)

        # Compute the coefficients required for the task reconstruction linear combination
        # (p', 1, d, m*H), (p', q, m, m), (p', q, d, m*H)
        partial_tasks_batch, linear_coeff, expected_task_data \
            = self._compute_task_reconstruction_coeff(tasks_batch, task_latent, traj_latent)

        exploration_traj_latent, latent_exploration_target = None, None
        use_contrastive_loss = self.get_iter_idx() >= self.args.epoch_start_contrastive_loss
        if use_contrastive_loss:
            exploration_traj_latent, latent_exploration_target \
                = self._compute_exploration_component(tasks_batch, task_latent, traj_latent)  # (p', H*d_z, m), (p')
        assert task_latent.requires_grad and traj_latent.requires_grad

        log_stats = self.transformer.compute_loss(data=tasks_batch,
                                                  reconstructed_data=reconstructed_tasks_batch,

                                                  shared_latent=shared_latent,
                                                  task_latent=task_latent,
                                                  traj_latent=traj_latent,

                                                  partial_task_data=partial_tasks_batch,
                                                  linear_coeff=linear_coeff,
                                                  expected_task_data=expected_task_data,

                                                  use_contrastive_loss=use_contrastive_loss,
                                                  exploration_traj_latent=exploration_traj_latent,
                                                  latent_exploration_target=latent_exploration_target,

                                                  mask=mask,
                                                  update=True,

                                                  print_info=False)

        if log:
            self._log_update(linear_coeff, use_contrastive_loss, latent_exploration_target)
        return log_stats, (shared_latent, task_latent, traj_latent)

    def _run_encoder(self, masked_tasks_batch, tasks_batch=None):
        if self.args.mask_entire_loss:
            # (p, q, H*d_z, m), (H*d_z), (p, q, H*d_z), (p, q, H*d_z, m), (p, q, d, m*H)
            latent, shared_latent, task_latent, traj_latent, reconstructed_tasks_batch \
                = self.transformer.forward(masked_tasks_batch, return_reconstructed_input=True)
        else:
            assert tasks_batch is not None
            # (p, q, d, m*H)
            _, _, _, _, reconstructed_tasks_batch \
                = self.transformer.forward(masked_tasks_batch, return_reconstructed_input=True)
            # (p, q, H*d_z, m), (H*d_z), (p, q, H*d_z), (p, q, H*d_z, m)
            latent, shared_latent, task_latent, traj_latent \
                = self.transformer.forward(tasks_batch, return_reconstructed_input=False)
        return latent, shared_latent, task_latent, traj_latent, reconstructed_tasks_batch


    # Select a meta-trajectory from several tasks in the data. Compute a set of coefficients, such that the linear
    # combination of the selected meta-trajectory gives back the entire dataset for its task
    def _compute_task_reconstruction_coeff(self, tasks_batch, task_latent, traj_latent):
        # Sample both the collected data and their latent representations:
        #   partial_tasks_batch: the meta-trajectory whose combination we will compute
        #   expected_task_data: the entire dataset for the meta-trajectory's task
        #   coeff_task_latent: the task-specific latent component of the selected tasks
        #   coeff_traj_latent: the traj-specific latent component of the selected meta-traj and tasks
        # (p', 1, d, m*H), (p', q, d, m*H), (p', q, H*d_z), (p', H*d_z, m)
        (partial_tasks_batch, expected_task_data), (coeff_task_latent, coeff_traj_latent), (task_idx, meta_traj_idx) \
            = self._sample_input_and_latent(tasks_batch, task_latent, traj_latent, return_idx=True)

        # Check if we want to backprop the task reconstruction loss through these 2 latent spaces
        if not self.args.task_rec_loss_through_task_latent:
            coeff_task_latent = coeff_task_latent.detach().clone()
        if not self.args.task_rec_loss_through_traj_latent:
            coeff_traj_latent = coeff_traj_latent.detach().clone()

        # Decide whether to use the task latent or the traj latent to represent the selected meta-episode
        if self.args.no_traj_latent_for_task_rec:
            coeff_traj_latent = task_latent[task_idx, meta_traj_idx].unsqueeze(1).clone()  # (p', 1, H*d_z)
        else:
            if not self.args.task_rec_loss_through_traj_latent:
                coeff_traj_latent = coeff_traj_latent.detach().clone()

        # Compute the coefficients of the linear combination for the given meta-trajectories
        linear_coeff = self.transformer.compute_linear_coeff(coeff_task_latent, coeff_traj_latent)  # (p', q, m, m)

        return partial_tasks_batch, linear_coeff, expected_task_data

    def _compute_exploration_component(self, tasks_batch, task_latent, traj_latent):
        # TODO The repulsive term might be better if we sample several (e.g. q') meta-traj per task.
        #  That way, trajectories that appear in multiple will be attracted to some and repulsed by some

        # TODO Maybe use the target transformer to compute the task latent
        task_latent = task_latent.detach().clone()

        # Sample both the collected data and their latent representations:
        # (p', 1, d, m*H), (p', q, d, m*H), (p', q, H*d_z), (p', H*d_z, m)
        ((target_partial_tasks_batch, target_expected_task_data), (partial_task_latent, partial_traj_latent),
         (task_idx, meta_traj_idx)) \
            = self._sample_input_and_latent(tasks_batch, task_latent, traj_latent, return_idx=True)

        if self.args.no_traj_latent_for_task_rec:
            traj_representation = task_latent[task_idx, meta_traj_idx].unsqueeze(1).clone()  # (p', 1, H*d_z)
        else:
            traj_representation = partial_traj_latent

        with torch.no_grad():
            # Compute the coefficients of the linear combination for the given meta-trajectories
            # Use the target encoder's parameters
            target_linear_coeff \
                = self.transformer.compute_linear_coeff(partial_task_latent,
                                                        traj_representation,
                                                        linear_coeff_head=self.target_coeff_head)  # (p', q, m, m)

            # Rearrange the input into a sequence of trajectories instead of a sequence of steps
            partial_task_data = step_to_traj(target_partial_tasks_batch, self.args.horizon)  # (p', 1, H*d, m)

            # Compute the reconstructed task data
            if self.args.task_reconstruction_type == 'linear':
                reconstructed_task_data = torch.matmul(partial_task_data, target_linear_coeff)  # (p', q, H*d, m)
            elif self.args.task_reconstruction_type == 'non-linear':
                w0, b0, w1, b1 = target_linear_coeff  # (p', q, m, w), (p', q, 1, w), (p', q, w, m), (p', q, 1, m)
                x = self.nonlinear_activation(torch.matmul(partial_task_data, w0) + b0)  # (p', q, H*d, w)
                reconstructed_task_data = torch.matmul(x, w1) + b1  # (p', q, H*d, m)

            # Rearrange back into a sequence of steps
            reconstructed_task_data = traj_to_step(reconstructed_task_data, self.args.horizon)  # (p', q, d, m*H)

            # Compute a distance metric using the difference between reconstructed task data and real task data
            latent_exploration_target = self._compute_exploration_target(target_expected_task_data,
                                                                         reconstructed_task_data)  # (p')
            if self.args.normalise_exploration_target:
                # FIXME Use 0 as min and find a suitable max
                target_min = latent_exploration_target.min()
                latent_exploration_target = (latent_exploration_target - target_min) / \
                                            (latent_exploration_target.max() - target_min)
                raise NotImplementedError('Use 0 as min and find a suitable max')

            return partial_traj_latent, latent_exploration_target  # (p', H*d_z, m), (p')

    def _sample_input_and_latent(self, tasks_batch, task_latent, traj_latent, task_latent_one_meta_traj=False,
                                 return_idx=False):
        # Select several random tasks
        task_idx = np.random.choice(self.args.batch_size, size=self.args.coeff_batch_size, replace=False)  # (p')
        # For each task, select (at random) a single meta-trajectory
        meta_traj_idx = np.random.choice(self.args.meta_traj_per_task, size=self.args.coeff_batch_size,
                                         replace=True)  # (p')

        # Sample the input
        partial_tasks_batch = tasks_batch[task_idx, meta_traj_idx].unsqueeze(1)  # (p', 1, d, m*H)
        partial_task_data = tasks_batch[task_idx]  # (p', q, d, m*H)

        # Sample the latent
        if task_latent_one_meta_traj:
            partial_task_latent = task_latent[task_idx, meta_traj_idx].unsqueeze(1)  # (p', 1, H*d_z)
        else:
            partial_task_latent = task_latent[task_idx]  # (p', q, H*d_z)
        partial_traj_latent = traj_latent[task_idx, meta_traj_idx]  # (p', H*d_z, m)

        data = (partial_tasks_batch, partial_task_data), (partial_task_latent, partial_traj_latent)
        if return_idx:
            data += ((task_idx, meta_traj_idx),)
        return data

    def _compute_exploration_target(self, expected, reconstructed):
        avg_distance = (expected - reconstructed).pow(2).mean(dim=(1, 2, 3))  # (p')

        if self.args.exploration_target_type == 'squared_diff':
            return avg_distance

        elif self.args.exploration_target_type == 'exponential':
            return torch.exp(avg_distance) - 1

        elif self.args.exploration_target_type == 'negative_exponential':
            # FIXME Set self.args.exp_target_coeff to 1, 1.5, or 2
            # raise NotImplementedError('Set self.args.exp_target_coeff to 1, 1.5, or 2')
            return 1 - torch.exp(self.args.exp_target_coeff * -1 * avg_distance)

        elif self.args.exploration_target_type == 'arctan':
            return torch.arctan(self.args.exp_target_coeff * avg_distance) / (torch.pi / 2)
        else:
            raise ValueError(f'Invalid value for exploration_target_type: {self.args.exploration_target_type}')

    def _log_update(self, linear_coeff, use_contrastive_loss, latent_exploration_target):
        with torch.no_grad():
            if self.args.task_reconstruction_type == 'linear':
                self.logger.add('latent_exploration/linear_coeff_max', linear_coeff.max())
                self.logger.add('latent_exploration/linear_coeff_min', linear_coeff.min())
                self.logger.add('latent_exploration/linear_coeff_absmean', linear_coeff.abs().mean())

                # FIXME Move eye to cuda instead of the other way around. Also use a static eye on cuda, instead of recreating it everytime
                eye = torch.zeros(linear_coeff.shape) + torch.eye(linear_coeff.shape[-2], linear_coeff.shape[-1])
                self.logger.add('latent_exploration/linear_coeff_vs_unit_matrix', (linear_coeff.cpu() - eye).pow(2).mean())
            elif self.args.task_reconstruction_type == 'non-linear':
                w0, b0, w1, b1 = linear_coeff
                self.logger.add('latent_exploration/linear_coeff_max', torch.mean(torch.cat((
                    w0.max().unsqueeze(0),
                    b0.max().unsqueeze(0),
                    w1.max().unsqueeze(0),
                    b1.max().unsqueeze(0)
                ))))
                self.logger.add('latent_exploration/linear_coeff_min', torch.mean(torch.cat((
                    w0.min().unsqueeze(0),
                    b0.min().unsqueeze(0),
                    w1.min().unsqueeze(0),
                    b1.min().unsqueeze(0)
                ))))
                self.logger.add('latent_exploration/linear_coeff_absmean', torch.mean(torch.cat((
                    w0.abs().mean().unsqueeze(0),
                    b0.abs().mean().unsqueeze(0),
                    w1.abs().mean().unsqueeze(0),
                    b1.abs().mean().unsqueeze(0)
                ))))

                self.logger.add('latent_exploration/linear_coeff_vs_unit_matrix', torch.mean(torch.cat((
                    (w0.cpu() - self._create_eye_tensor(w0.shape)).pow(2).mean().unsqueeze(0),
                    (b0.cpu() - self._create_eye_tensor(b0.shape)).pow(2).mean().unsqueeze(0),
                    (w1.cpu() - self._create_eye_tensor(w1.shape)).pow(2).mean().unsqueeze(0),
                    (b1.cpu() - self._create_eye_tensor(b1.shape)).pow(2).mean().unsqueeze(0)
                ))))

            if use_contrastive_loss:
                self.logger.add('latent_exploration/target_max', latent_exploration_target.max())
                self.logger.add('latent_exploration/target_min', latent_exploration_target.min())
                self.logger.add('latent_exploration/target_mean', latent_exploration_target.mean())

    def _create_eye_tensor(self, shape):
        return torch.zeros(shape) + torch.eye(shape[-2], shape[-1])
