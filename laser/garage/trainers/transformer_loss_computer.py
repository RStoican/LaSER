import torch
from laser.garage.torch import global_device
from laser.garage.torch.modules.mlp_module import _get_activation_fn
from laser.garage.utils import helpers as utl


class TransformerLossComputer:
    def __init__(self,
                 args,
                 masked_reconstruction_loss,
                 task_latent_type,
                 traj_per_meta_traj, traj_len,
                 obs_dim, act_dim):
        self.args = args
        self._masked_reconstruction_loss = masked_reconstruction_loss
        self._task_latent_type = task_latent_type

        self._traj_per_meta_traj = traj_per_meta_traj
        self._traj_len = traj_len

        self._obs_dim = obs_dim
        self._act_dim = act_dim

        if self.args.task_reconstruction_type == 'non-linear':
            self.nonlinear_activation = _get_activation_fn(self.args.task_reconstruction_activation)

    def compute_loss(self,
                     data, reconstructed_data,
                     shared_latent, task_latent, traj_latent,
                     partial_task_data, linear_coeff, expected_task_data,
                     use_contrastive_loss, exploration_traj_latent, latent_exploration_target,
                     mask,
                     print_info=True):
        """

        :param data: (p, q, d, m*H)
        :param reconstructed_data: (p, q, d, m*H)

        :param shared_latent: (H*d_z)
        :param task_latent: (p, q, H*d_z)
        :param traj_latent: (p, q, H*d_z, m)

        :param partial_task_data: (p', 1, d, m*H)
        :param linear_coeff: (p', q, m, m)
        :param expected_task_data: (p', q, d, m*H)

        :param use_contrastive_loss:
        :param exploration_traj_latent: (p', H*d_z, m)
        :param latent_exploration_target: (p')

        :param mask:
        :param print_info
        :return:
        """

        # The loss between the (possibly masked) input and reconstructed output
        reconstruction_loss = self._compute_reconstruction_loss(data,
                                                                reconstructed_data,
                                                                mask)

        obs_reconstruction_loss, act_reconstruction_loss, rew_reconstruction_loss = None, None, None
        if self.args.split_reconstruction_loss:
            obs_reconstruction_loss, act_reconstruction_loss, rew_reconstruction_loss = reconstruction_loss
            reconstruction_loss = (self.args.state_loss_coeff * obs_reconstruction_loss +
                                   self.args.act_loss_coeff * act_reconstruction_loss +
                                   self.args.rew_loss_coeff * rew_reconstruction_loss)

        # The loss between a (linear combination of a) single meta-trajectory and all meta-trajectories in its task
        task_reconstruction_loss = self._compute_task_reconstruction_loss(partial_task_data,
                                                                          linear_coeff,
                                                                          expected_task_data)

        obs_task_rec_loss, act_task_rec_loss, rew_task_rec_loss = None, None, None
        if self.args.split_reconstruction_loss:
            obs_task_rec_loss, act_task_rec_loss, rew_task_rec_loss = task_reconstruction_loss
            task_reconstruction_loss = (self.args.state_task_loss_coeff * obs_task_rec_loss +
                                        self.args.act_task_loss_coeff * act_task_rec_loss +
                                        self.args.rew_task_loss_coeff * rew_task_rec_loss)

        contrastive_loss, exploration_sim = None, None
        if use_contrastive_loss:
            contrastive_loss, exploration_sim = self._compute_contrastive_loss(exploration_traj_latent,
                                                                              latent_exploration_target)

        # Regularise the latent space
        regulariser = self._compute_regulariser(shared_latent,
                                                task_latent,
                                                traj_latent)

        ctr_loss = contrastive_loss if contrastive_loss is not None else 0
        loss = (self.args.reconstruction_coeff * reconstruction_loss
                + self.args.task_reconstruction_coeff * task_reconstruction_loss
                + self.args.contrastive_coeff * ctr_loss
                + self.args.regulariser_coeff * regulariser)
        return loss, self._create_loss_stats(loss=loss,
                                             reconstruction_loss=reconstruction_loss,
                                             obs_reconstruction_loss=obs_reconstruction_loss,
                                             act_reconstruction_loss=act_reconstruction_loss,
                                             rew_reconstruction_loss=rew_reconstruction_loss,
                                             task_reconstruction_loss=task_reconstruction_loss,
                                             obs_task_reconstruction_loss=obs_task_rec_loss,
                                             act_task_reconstruction_loss=act_task_rec_loss,
                                             rew_task_reconstruction_loss=rew_task_rec_loss,
                                             contrastive_loss=contrastive_loss,
                                             regulariser=regulariser,
                                             latent_dot_prod=exploration_sim,
                                             print_info=print_info)

    def _compute_reconstruction_loss(self, data, reconstructed_data, mask):
        if self._masked_reconstruction_loss:
            if mask is None:
                raise ValueError('Computing masked reconstruction loss. However, the mask is None')

        if not self.args.split_reconstruction_loss:
            if not self.args.use_action_loss:
                raise NotImplementedError('We need to compare the action in the next step, since the current action '
                                          'might be part of the input, depending on args.mask_latest_action')

            # Average over all the steps in a meta-trajectory
            if self._masked_reconstruction_loss:
                # To compute the actual mean of each meta-episode, we need to compute the number of comparisons we made,
                # taking only the masked steps into account
                reconstruction_loss = (data[mask] - reconstructed_data[mask]).pow(2).mean(dim=-1)  # (p, q, d)
            else:
                reconstruction_loss = (data - reconstructed_data).pow(2).mean(dim=-1)  # (p, q, d)

            # Compute the loss over the features of a step
            if self.args.loss_avg_features:
                if self.args.mask_latest_action:
                    reconstruction_loss = reconstruction_loss.mean(dim=-1)  # (p, q)
                else:
                    reconstruction_loss = reconstruction_loss.sum(dim=-1) / (self.args.state_dim + 1)  # (p, q)
            else:
                reconstruction_loss = reconstruction_loss.sum(dim=-1)  # (p, q)

            # Average over meta-trajectories and tasks
            # FIXME Small issue: if there is an entire meta-trajectory that had no mask, the avg loss will be slightly
            #  wrong. Divide the loss sum by the number of meta-trajectories (summed from all p tasks) that were used in
            #  computing the loss. E.g. for
            #                   tensor([[0., 1., 1.],
            #                           [1., 1., 0.]])
            #  the [0,0] and [1,2] meta-trajectories were not used, so divide the loss sum by 4, not 6
            return reconstruction_loss.mean()  # (1)

        else:
            obs_mask, act_mask, rew_mask = 3 * (None,)
            if self._masked_reconstruction_loss:
                obs_mask = mask[:, :, :self._obs_dim, :]
                act_mask = mask[:, :, :self._obs_dim:self._obs_dim + self._act_dim, :] \
                    if self.args.use_action_loss else None
                rew_mask = mask[:, :, -1:, :]

            obs, act, rew = self._split_data(data)
            rec_obs, rec_act, rec_rew = self._split_data(reconstructed_data)

            obs_reconstruction_loss = self._compute_separate_reconstruction_loss(obs, rec_obs, obs_mask)
            act_reconstruction_loss = self._compute_separate_reconstruction_loss(act, rec_act, act_mask) \
                if self.args.use_action_loss else 0
            rew_reconstruction_loss = self._compute_separate_reconstruction_loss(rew, rec_rew, rew_mask)

            return obs_reconstruction_loss, act_reconstruction_loss, rew_reconstruction_loss

    def _compute_task_reconstruction_loss(self, partial_task_data, linear_coeff, expected_task_data):
        # Rearrange the input into a sequence of trajectories instead of a sequence of steps
        partial_task_data = utl.step_to_traj(partial_task_data, traj_len=self._traj_len)  # (p', 1, H*d, m)

        # Reconstruct the entire dataset for this task from one meta-trajectory and the given coefficients
        if self.args.task_reconstruction_type == 'linear':
            reconstructed_task_data = torch.matmul(partial_task_data, linear_coeff)  # (p', q, H*d, m)
        elif self.args.task_reconstruction_type == 'non-linear':
            # FIXME Use a method to do this
            w0, b0, w1, b1 = linear_coeff  # (p', q, m, w), (p', q, 1, w), (p', q, w, m), (p', q, 1, m)
            x = self.nonlinear_activation(torch.matmul(partial_task_data, w0) + b0)  # (p', q, H*d, w)
            reconstructed_task_data = torch.matmul(x, w1) + b1  # (p', q, H*d, m)
        else:
            raise ValueError(f'Unknown reconstruction type: {self.args.task_reconstruction_type}')

        # Rearrange back into a sequence of steps
        reconstructed_task_data = utl.traj_to_step(reconstructed_task_data, self._traj_len)  # (p', q, d, m*H)

        # FIXME Create method to do this (i.e. use the reconstruction loss method)
        if not self.args.split_reconstruction_loss:
            if not self.args.use_action_loss:
                raise NotImplementedError

            # FIXME Why use sum here?
            # task_reconstruction_loss = (expected_task_data - reconstructed_task_data).pow(2).sum(dim=-1)  # (p', q, d)

            # Compute the loss over the features of a step
            if self.args.loss_avg_features:
                task_reconstruction_loss = (expected_task_data - reconstructed_task_data).pow(2).mean(
                    dim=-1)  # (p', q, d)
                task_reconstruction_loss = task_reconstruction_loss.mean(dim=-1)  # (p, q)
            else:
                task_reconstruction_loss = (expected_task_data - reconstructed_task_data).pow(2).sum(
                    dim=-1)  # (p', q, d)
                task_reconstruction_loss = task_reconstruction_loss.sum(dim=-1)  # (p, q)
            return task_reconstruction_loss.mean()
        else:

            obs, act, rew = self._split_data(expected_task_data)
            rec_obs, rec_act, rec_rew = self._split_data(reconstructed_task_data)

            obs_reconstruction_loss = self._compute_separate_reconstruction_loss(obs, rec_obs)
            act_reconstruction_loss = self._compute_separate_reconstruction_loss(act, rec_act) \
                if self.args.use_action_loss else 0
            rew_reconstruction_loss = self._compute_separate_reconstruction_loss(rew, rec_rew)

            return obs_reconstruction_loss, act_reconstruction_loss, rew_reconstruction_loss

    def _compute_separate_reconstruction_loss(self, true, rec, mask=None):
        if mask is None:
            reconstruction_loss = (true - rec).pow(2).mean(dim=-1)
        else:
            # Only compute the loss between masked steps
            reconstruction_loss = (true[mask] - rec[mask]).pow(2).mean(dim=-1)

        if self.args.loss_avg_features:
            reconstruction_loss = reconstruction_loss.mean(dim=-1)  # (p, q)
        else:
            reconstruction_loss = reconstruction_loss.sum(dim=-1)  # (p, q)
        return reconstruction_loss.mean()

    def _compute_contrastive_loss(self, traj_latent, latent_exploration_target):
        """
        Given the latent of all the trajectories used, increase/decrease the dot product between different
        trajectories, based on the target computed for that task
        """
        # FIXME Create method
        if self.args.exploration_similarity_measure == 'dot_product':
            # Compute the dot product matrix between (the latent of) all given trajectories
            exploration_sim = torch.matmul(traj_latent.permute(0, 2, 1), traj_latent)  # (p', m, m)

        elif self.args.exploration_similarity_measure == 'cosine_similarity':
            # Compute the cosine similarity matrix product between (the latent of) all given trajectories
            traj_latent_transpose = traj_latent.permute(0, 2, 1)
            exploration_sim = torch.nn.functional.cosine_similarity(traj_latent_transpose.unsqueeze(1),
                                                                    traj_latent_transpose.unsqueeze(2),
                                                                    dim=-1)  # (p, m, m)

        else:
            raise ValueError(f'Unknown similarity measure: {self.args.exploration_similarity_measure}')

        # The dot product of all different vectors from the same task will have the same target
        latent_exploration_target = latent_exploration_target.unsqueeze(-1).unsqueeze(-1)  # (p', 1, 1)
        latent_exploration_target = latent_exploration_target \
            .repeat_interleave(self._traj_per_meta_traj, dim=1) \
            .repeat_interleave(self._traj_per_meta_traj, dim=-1)  # (p', m, m)

        loss = (exploration_sim - latent_exploration_target).pow(2)  # (p', m, m)

        # The diagonal gives the (squared) norm of each latent trajectory.
        # The regulariser deals with this, so no need to compute its loss here
        loss[:, range(self._traj_per_meta_traj), range(self._traj_per_meta_traj)] = 0

        return loss.mean(), exploration_sim.detach()

    def _compute_regulariser(self, shared_latent, task_latent, traj_latent):
        total_tasks = task_latent.shape[0]  # p
        reg1_count = 1

        shared_norm_loss = 0
        if self.args.normalise_shared_latent:
            # Regularise the norm of the shared latent to 1
            shared_norm = torch.linalg.vector_norm(shared_latent)
            shared_norm_loss = (shared_norm - 1).pow(2)
            reg1_count += 1

        task_norm_loss = 0
        if self.args.normalise_task_latent:
            # Regularise the norm of the rows in the task latent to 1
            task_norm = torch.linalg.vector_norm(task_latent, dim=-1)  # (p, q) / (p, q, m*H)
            task_norm_loss = (task_norm - 1).pow(2).mean()
            reg1_count += 1

        if self._task_latent_type == 'orthogonal':
            raise NotImplementedError('The dimensions of the shared and task latents need to be readjusted')
            # Ensure both the shared and task-specific latents are orthogonal, and orthogonal to each other
            shared_latent = shared_latent.reshape(1, -1, 1).repeat_interleave(total_tasks, dim=0)  # (p, d_model, 1)
            partial_latent = torch.cat((shared_latent, task_latent), dim=-1)  # (p, d_model, d_z+1)

            # The dot product of all (d_z + 1) d_model-dimensional vectors
            latent_dot_prod = torch.matmul(partial_latent.permute(0, 2, 1), partial_latent)  # (p, d_z+1, d_z+1)

            # Create p (d_z+1, d_z+1) unit matrices to compare the latent dot product to
            eyes = torch.eye(partial_latent.shape[-1]).unsqueeze(0).to(global_device())  # (1, d_z+1, d_z+1)
            eyes = eyes.repeat_interleave(total_tasks, dim=0)  # (p, d_z+1, d_z+1)

            reg1 = (latent_dot_prod - eyes).pow(2).reshape(total_tasks, -1).mean(dim=-1)  # (p)
            reg1 = reg1.sum()  # (1)
        elif self._task_latent_type == 'diagonal':
            # Orthogonalise the shared and task latents
            latent_dot_prod = torch.matmul(shared_latent, torch.diag_embed(task_latent))  # (p, q, H*d_z)
            ortho_loss = latent_dot_prod.pow(2).mean()
        else:  # self._task_latent_type == 'attention'
            if not self.args.simple_shared_latent:
                # FIXME Does it make sense to compare every element of shared_latent to the single task_latent???
                #  Should it be the other way around (i.e. a multi-component task and single-component shared)???
                # TODO This is only well-defined when H <= d_z-1 and works better when H << d_z.
                #  Intuitively, H-d_z is the number of different orthogonal task representations.
                #  Maybe we should put a warning at the start for H > d_z or H > (d_z - threshold) ???

                # shared_latent   (H*d_z)
                # task_latent   (p, q, d_z)
                shared_latent = shared_latent.view(self._traj_len, -1)  # (H, d_z)
                latent_dot_prod = torch.matmul(shared_latent, task_latent.unsqueeze(-1))  # (p, q, H, 1)
            else:
                # FIXME When task_latent_multidim == True, we regularize the d_z-dimensional vector shared_latent to be
                #  orthogonal to all the p*q*m*H d_z-dimensional vectors in task_latent. This is difficult to
                #  approximate (and impossible to get exact) when d_z < p*q*m*H
                # shared_latent   (d_z)
                # task_latent   (p, q, d_z) / (p, q, m*H, d_z)
                latent_dot_prod = torch.matmul(task_latent, shared_latent)  # (p, q) / (p, q, m*H)
            ortho_loss = latent_dot_prod.pow(2).mean()

        reg1 = (shared_norm_loss + task_norm_loss + ortho_loss) / reg1_count

        if self.args.regularise_traj_latent:
            # Ensure the traj-specific latents are normalised
            latent_traj_norm = torch.linalg.vector_norm(traj_latent, dim=-2).reshape(-1)  # (p, q, m)
            reg2 = (latent_traj_norm - 1).pow(2)  # (p, q, m)

            if self.args.loss_avg_features:
                reg2 = reg2.mean()  # (1)
            else:
                reg2 = reg2.reshape(total_tasks, -1).mean(dim=-1)  # (p)
                reg2 = reg2.sum()  # (1)
        else:
            reg2 = 0

        return reg1 + reg2

    def _split_data(self, data):
        obs = data[:, :, :self._obs_dim]  # (p, q, d_obs, m*H)
        act = data[:, :, self._obs_dim:self._obs_dim + self._act_dim]  # (p, q, d_act, m*H)
        rew = data[:, :, self._obs_dim + self._act_dim:self._obs_dim + self._act_dim + 1]  # (p, q, 1, m*H)
        return obs, act, rew

    def _create_loss_stats(self,
                           loss,
                           reconstruction_loss,
                           obs_reconstruction_loss, act_reconstruction_loss, rew_reconstruction_loss,
                           task_reconstruction_loss,
                           obs_task_reconstruction_loss, act_task_reconstruction_loss, rew_task_reconstruction_loss,
                           contrastive_loss,
                           regulariser,
                           latent_dot_prod,
                           print_info=True):
        loss_dict = {
            'loss': loss,
            'reconstr_loss': reconstruction_loss,
            'task_reconstr_loss': task_reconstruction_loss,
            'regulariser': regulariser,
        }
        if self.args.split_reconstruction_loss:
            loss_dict['observation_reconstr_loss'] = obs_reconstruction_loss
            loss_dict['reward_reconstr_loss'] = rew_reconstruction_loss
            loss_dict['observation_task_loss'] = obs_task_reconstruction_loss
            loss_dict['reward_task_loss'] = rew_task_reconstruction_loss
            if self.args.use_action_loss:
                loss_dict['action_reconstr_loss'] = act_reconstruction_loss
                loss_dict['action_task_loss'] = act_task_reconstruction_loss
        if contrastive_loss is not None:
            loss_dict['contrastive_loss'] = contrastive_loss

        if print_info:
            print(f'Loss: {loss}')
            print(f'     Reconstruction Loss: {reconstruction_loss}')
            if self.args.split_reconstruction_loss:
                print(f'          Obs Loss: {obs_reconstruction_loss}')
                print(f'          Rew Loss: {rew_reconstruction_loss}')
                if self.args.use_action_loss:
                    print(f'          Act Loss: {act_reconstruction_loss}')
            print(f'     Task Reconstruction Loss: {task_reconstruction_loss}')
            if contrastive_loss is not None:
                print(f'     Contrastive Loss: {contrastive_loss}')
            print(f'     Regulariser: {regulariser}')

        latent_dict = None
        if latent_dot_prod is not None:
            latent_dict = {
                'latent_dot_prod_mean': latent_dot_prod.mean(),
                'latent_dot_prod_min': latent_dot_prod.min(),
                'latent_dot_prod_max': latent_dot_prod.max(),
            }

        return loss_dict, latent_dict
