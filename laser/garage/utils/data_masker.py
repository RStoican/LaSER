import numpy as np
import torch
from laser.garage.torch._functions import global_device


class DataMasker:
    def __init__(self,
                 mask_prob,
                 mask_prob_token,
                 mask_prob_replace,
                 mask_token,
                 mask_latest_action,
                 obs_dim=None, act_dim=None):
        self.mask_prob = mask_prob
        self.mask_prob_token = mask_prob_token
        self.mask_prob_replace = mask_prob_replace
        self.mask_token = mask_token
        self.mask_latest_action = mask_latest_action
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if not self.mask_latest_action:
            assert self.obs_dim is not None and self.act_dim is not None

    def mask_input_traj(self, data):
        """Mask the dataset for self-supervised training. Given a dataset of RL trajectories, mask some of them based
        on the probability given to the model

                Args:
                    :param data: (torch.Tensor) a dataset collected from p tasks. For each task, we have q
                    meta-trajectories. Each meta-trajectory has m trajectories, each of length H steps. Each step is a
                    feature vector ((observation, done), action, reward) of size d
                        data: shape (p, q, d, m*H)

                Returns:
                    :return masked_data: the masked dataset
                        masked_data: shape (p, q, d, m*H)

                """

        # FIXME Look at BERT Ch. 3.1 again
        #  collect the indices of the steps selected (i.e. replaced with 0, replaced with random steps, kept the same).
        #  Return those indices and only use the loss computed between the input and reconstucted input at those indices

        data = data.clone()

        step_mask_shape = list(data.shape)  # (p, q, d, m*H)
        step_dim = 2
        p = step_mask_shape[0]
        q = step_mask_shape[1]
        step_size = step_mask_shape[step_dim]  # d
        h = step_mask_shape[3]
        step_mask_shape[step_dim] = 1  # (p, q, 1, m*H)
        total_steps = p * q * h  # (p*q*m*H)

        # # Get the number of steps to mask
        # total_steps = step_mask_shape[0] * step_mask_shape[1] * step_mask_shape[2]  # (p*q*m*H)
        # num_selected = int(0.9915 * total_steps)
        #
        # # Shuffle the indices and select ...% of them
        # selected_indices = torch.randperm(total_steps)[:num_selected]
        #
        # # Convert selected_indices to 3D indices
        # p_indices = selected_indices // (step_mask_shape[1] * step_mask_shape[3])
        # remainder = selected_indices % (step_mask_shape[1] * step_mask_shape[3])
        # q_indices = remainder // step_mask_shape[3]
        # horizon_indices = remainder % step_mask_shape[3]
        #
        # # Replace ...% of selected vectors with the [MASK] token
        # num_replace_token = int(0 * num_selected)
        # token_mask = torch.zeros_like(data, dtype=torch.bool, device=global_device())
        # token_mask[
        # p_indices[:num_replace_token],
        # q_indices[:num_replace_token],
        # :,
        # horizon_indices[:num_replace_token]
        # ] = True
        # if not self.mask_latest_action:
        #     token_mask[:, :, self.obs_dim:self.obs_dim + self.act_dim, :] = False
        # data[token_mask] = self.mask_token
        #
        # # Replace ...% with other steps from the data
        # num_replace_steps = int(0.99 * num_selected)
        # permuted_data = data.view(-1, step_size).clone()  # Flatten to 2D tensor for permutation
        # permuted_data = permuted_data[torch.randperm(permuted_data.size(0))].view(data.shape)  # Random permutation
        # print(permuted_data[0, 0, :-1, :10])

        mask_prob = torch.rand((p, q, 1, h), device=global_device())
        mask_prob = mask_prob.repeat_interleave(repeats=step_size, dim=step_dim)
        mask = torch.zeros_like(data, dtype=torch.bool, device=global_device())
        mask[mask_prob <= self.mask_prob] = 1
        # print(f'{50 * "="}\n{50 * "="}\n{50 * "="}')
        # print(mask.long()[0, 0, :-1, :10])

        prob = torch.rand((p, q, 1, h), device=global_device())
        prob = prob.repeat_interleave(repeats=step_size, dim=step_dim)
        token_mask = torch.zeros_like(data, dtype=torch.bool, device=global_device())
        token_mask[(prob < self.mask_prob_token) & (mask == 1)] = 1
        # print(f'{50 * "="}\n{50 * "="}\n{50 * "="}')
        # print(token_mask.long()[0, 0, :-1, :10])

        if not self.mask_latest_action:
            # Only [MASK] the states and rewards
            token_mask[:, :, self.obs_dim:self.obs_dim + self.act_dim, :] = 0

        replace_mask = torch.zeros_like(data, dtype=torch.bool, device=global_device())
        replace_prob = (self.mask_prob_token <= prob) & (prob < self.mask_prob_token + self.mask_prob_replace)
        valid_replace = (token_mask == 0) & (mask == 1)
        replace_mask[replace_prob & valid_replace] = 1

        # print(f'{50 * "="}\n{50 * "="}\n{50 * "="}')
        # print(replace_mask.long()[0, 0, :-1, :10])

        # FIXME We should only use actual steps as replacement, not padding
        shuffled_data = data\
            .permute(0, 1, 3, 2)\
            .reshape(-1, step_size)[torch.randperm(total_steps)]\
            .view(p, q, h, step_size)\
            .permute(0, 1, 3, 2)

        # print(f'{50 * "="}\n{50 * "="}\n{50 * "="}')
        # print(data[0, 0, :-1, :10])

        data[token_mask] = self.mask_token
        data[replace_mask] = shuffled_data[replace_mask]

        # print(f'{50 * "="}\n{50 * "="}\n{50 * "="}')
        # print(data[0, 0, :-1, :10])

        # FIXME Add a third mask that replaces an entire trajectory with the [MASK] token. The model should learn to
        #  predict that from the other (meta-)trajectories in that task
        pass

        # FIXME See if it makes sense to add a fourth mask that replaces an entire trajectory with a trajectory from
        #  another task
        pass

        return data, mask

        # # Compute the number of elements for each value
        # num_ones = int(0.8 * total_elements)
        # num_twos = int(0.1 * total_elements)
        # num_threes = total_elements - num_ones - num_twos  # Remaining elements will be threes
        #
        # # Create a 1D tensor with the specified proportions of 1s, 2s, and 3s
        # values = torch.tensor([1] * num_ones + [2] * num_twos + [3] * num_threes)
        #
        # # Shuffle the values to randomize their positions
        # shuffled_values = values[torch.randperm(total_elements)]
        #
        # # Reshape the 1D tensor to the desired 3D shape
        # tensor_3d = shuffled_values.view(shape)
        #
        # prob = torch.rand((p, q, 1, h), device=global_device())
        # prob = prob.repeat_interleave(repeats=step_size, dim=step_dim)
        # token_idx = prob <= 0.5
        # mask[token_idx] += 1
        # print(f'{50 * "="}\n{50 * "="}\n{50 * "="}')
        # print(mask[0, 0, :-1, :10])
        #
        # # if not self.mask_latest_action:
        # #     token_mask[:, :, self.obs_dim:self.obs_dim + self.act_dim, :] = False
        #
        # data[mask] = self.mask_token
        #
        # # Mask the data. FOr each index, if the mask value is
        # #   0, leave it unchanged
        # #   1, use [MASK] token
        # #   2, use a random token
        # #   3, leave it unchanged
        #
        # # print(f'{50 * "="}\n{50 * "="}\n{50 * "="}')
        # # print(data[0, 0, :-1, :10])

        # Create a mask of 1's for each step in the trajectory, then randomly replace some of them with 0, with
        # probability self.args.step_mask_prob_token. Replace all the 0 positions with the [MASK] token
        token_mask, token_idx = self._create_random_zero_mask(step_mask_shape,
                                                              dim=step_dim,
                                                              dim_size=step_size,
                                                              prob_zero=self.step_mask_prob_token)

        # Create a mask that, with prob self.args.step_mask_prob_other, will replace each step with another random step
        # in the meta-trajectory
        other_mask, replacement_mask, _ = self._create_random_step_mask(data,
                                                                        step_mask_shape,
                                                                        dim=step_dim,
                                                                        dim_size=step_size,
                                                                        prob_zero=self.step_mask_prob_other)

        data = data * other_mask + replacement_mask
        data[token_idx] = self.mask_token

        # FIXME Add a third mask that replaces an entire trajectory with the [MASK] token. The model should learn to
        #  predict that from the other (meta-)trajectories in that task
        pass

        # FIXME See if it makes sense to add a fourth mask that replaces an entire trajectory with a trajectory from
        #  another task
        pass

        return data, (token_mask, other_mask)

    def _create_random_zero_mask(self, mask_shape, dim, dim_size, prob_zero):
        assert mask_shape[dim] == 1
        mask = np.random.choice(2, mask_shape, p=[prob_zero, 1 - prob_zero])  # (p, q, 1, m*H)
        mask = np.repeat(mask, repeats=dim_size, axis=dim)  # (p, q, d, m*H)
        if not self.mask_latest_action:
            mask[:, :, self.obs_dim:self.obs_dim + self.act_dim, :] = 1
        mask = torch.from_numpy(mask).to(global_device())  # (p, q, d, m*H)
        idx = torch.nonzero(1 - mask, as_tuple=True)  # Index of all steps that are zero
        return mask, idx

    def _create_random_step_mask(self, data, mask_shape, dim, dim_size, prob_zero):
        # A 0-1 mask will be used to decide which of the steps are replaced
        step_zero_mask, idx = self._create_random_zero_mask(mask_shape, dim, dim_size, prob_zero)

        # Get a random permutation of the steps in the dataset. To be used to replace the steps given in idx
        replacement_idx = torch.randperm(torch.tensor(mask_shape).prod())  # (p*q*1*m*H)

        # FIXME Use dim and dim_size
        # Create a matrix of all the replacement steps, and 0 everywhere else
        replacement = data.permute(0, 1, 3, 2)  # (p, q, m*H, d)
        replacement = replacement.reshape(-1, dim_size)[replacement_idx].view(replacement.size())  # (p, q, m*H, d)
        replacement = replacement.permute(0, 1, 3, 2)  # (p, q, d, m*H)
        replacement = replacement * (1 - step_zero_mask)

        idx = torch.nonzero(replacement, as_tuple=True)
        return step_zero_mask, replacement, idx
