import os
import pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from laser.garage.torch._functions import global_device
from laser.garage.utils.running_stats import TorchRunningMeanStd


def create_split_dict(args, train_task_count, test_task_count):
    if "split" not in args or args.split is None or len(args.split) <= 0:
        return None

    return {
        "split": args.split,
        "train_count": train_task_count,
        "split_train": args.split_train,
        "eval_count": test_task_count,
        "split_eval": args.split_eval
    }


def reset_env(env, args, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset().float().to(global_device())

    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)
    return state


# Reset the environment without resetting the task
def reset_meta_traj(env):
    state = env.reset_meta_traj().float().to(global_device())
    return state


def squash_action(action, args):
    if args.norm_actions_post_sampling:
        return torch.tanh(action)
    else:
        return action


def env_step(env, action, args):
    act = squash_action(action.detach(), args)
    next_obs, reward, done, infos = env.step(act)

    if isinstance(next_obs, list):
        next_obs = [o.to(global_device()) for o in next_obs]
    else:
        next_obs = next_obs.to(global_device())
    if isinstance(reward, list):
        reward = [r.to(global_device()) for r in reward]
    else:
        reward = reward.to(global_device())

    return next_obs, reward, done, infos


def select_action(args,
                  policy,
                  deterministic,
                  state=None,
                  latent=None,
                  task_context=None):
    """ Select action using the policy. """
    latent = get_latent_for_policy(args=args, latent=latent)
    if state is not None and state.shape[0] == 1:
        state = state.squeeze(0)

    action = policy.act(state=state, latent=latent, task_context=task_context, deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action = action
    else:
        raise NotImplementedError
    action = action.to(global_device())
    return value, action


def padding_action(action_space):
    if hasattr(action_space, 'padding_token'):
        return action_space.padding_token

    # FIXME Find a better way to do this
    PADDING_TOKEN = -1234.5678
    return np.full(action_space.shape, PADDING_TOKEN, dtype=float)


def get_latent_for_policy(args, latent=None):
    if latent is None:
        return None

    if args.add_nonlinearity_to_latent:
        latent = F.relu(latent)

    while latent.shape[0] == 1:
        latent = latent.squeeze(0)
    return latent


def step_to_traj(seq_of_steps, traj_len):
    """
    Rearrange a sequence of steps (*, d, m*H) into a sequence of trajectories (*, H*d, m),
    where * is any number of batch dimensions

    :param seq_of_steps: (*, d, m*H); a sequence of steps
    :param traj_len: H; the length of a trajectory
    :return: (*, H*d, m); a sequence of trajectories
    """
    new_shape = seq_of_steps.shape[:-1] + (-1, traj_len)
    seq_of_trajs = seq_of_steps.reshape(new_shape)  # (*, d, m, H)

    seq_of_trajs = seq_of_trajs.permute(tuple(range(seq_of_trajs.dim() - 3)) + (-1, -3, -2))  # (*, H, d, m)
    return seq_of_trajs.reshape(seq_of_trajs.shape[:-3] + (-1, seq_of_trajs.shape[-1]))  # (*, H*d, m)


def traj_to_step(seq_of_trajs, traj_len):
    """
    Rearrange a sequence of trajectories (*, H*d, m) into a sequence of steps (*, d, m*H),
    where * is any number of batch dimensions

    :param seq_of_trajs: (*, H*d, m); a sequence of trajectories
    :param traj_len: H; the length of a trajectory
    :return: (*, d, m*H); a sequence of steps
    """
    new_shape = seq_of_trajs.shape[:-2] + (traj_len, -1, seq_of_trajs.shape[-1])
    seq_of_steps = seq_of_trajs.reshape(new_shape)  # (*, H, d, m)

    seq_of_steps = seq_of_steps.permute(tuple(range(seq_of_steps.dim() - 3)) + (-2, -1, -3))  # (*, d, m, H)
    return seq_of_steps.reshape(seq_of_steps.shape[:-2] + (-1,))  # (*, d, m*H)


def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'Use only one process and set --deterministic_execution to True if you want identical results '
              '(only recommended for debugging).')


def recompute_embeddings(args,
                         policy_storage,
                         exploration_trajectories,
                         transformer,
                         update_idx,
                         ):
    assert args.ablation_use_context

    # Recompute the context, and keep gradients
    task_context, _, _, _ = transformer(exploration_trajectories,
                                        compute_shared_latent=False,
                                        use_static_shared_latent=True)  # (p, 1, H*d_z, m)
    assert task_context.requires_grad

    assert task_context.shape[1] == 1
    task_context = task_context.squeeze(1)  # (p, H*d_z, m)

    if args.context_type != 'full':
        task_context = task_context.reshape(
            task_context.shape[0], args.horizon, -1, task_context.shape[-1])  # (p, H, d_z, m)
        if args.context_type == 'traj_latest':
            task_context = task_context[:, -1, :, :]  # (p, d_z, m)
        elif args.context_type == 'latest':
            task_context = task_context[:, -1, :, -1]  # (p, d_z)
    task_context = task_context.reshape(task_context.shape[0], -1)  # (p, H*d_z*m / d_z*m / d_z)

    task_context = task_context.repeat((args.batches_task_collect,) + (len(task_context.shape) - 1) * (1,))

    if update_idx == 0:
        diff = (policy_storage.task_context[:, 0] - task_context).pow(2).sum()
        try:
            assert diff <= 1e-07
        except AssertionError:
            warnings.warn(f'You are not recomputing the embeddings correctly!. Difference: {diff}')
            import pdb
            pdb.set_trace()

    assert policy_storage.task_context.shape[0] == task_context.shape[0]

    # TODO Think about also updating the transformer that encodes the history of the current task (instead of just the
    #  one that encodes the exploration data)

    return task_context


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs.float()))
        else:
            return torch.zeros(0, ).to(global_device())


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        return pickle.load(f)


_state_rms = None
_rew_rms = None


def create_global_state_rms(args, force=False):
    global _state_rms
    _state_rms = _create_global_rms(_state_rms, args, shape=args.state_dim, force=force)


def create_global_rew_rms(args, force=False):
    global _rew_rms
    _rew_rms = _create_global_rms(_rew_rms, args, shape=(), force=force, device='cpu')


def _create_global_rms(rms, args, shape, force=False, device=None):
    assert rms is None
    if args.norm_state_exploration or force:
        return TorchRunningMeanStd(shape=shape, device=device)
    return None


def update_state_rms(tasks_batch, lens, args):
    """ Update normalisation parameters for inputs with current data """
    if args.norm_state_exploration:
        states = tasks_batch[..., :args.state_dim, :]
        _state_rms.masked_update(states, lens, h=args.horizon)


def update_rew_rms(ret):
    for a in [i == 0 or ret.shape[i] == 1 for i in range(len(ret.shape))]:
        assert a, 'Wrong shape'
    ret = ret.reshape(ret.shape[0], 1, 1, 1)  # (p, 1, 1, 1)
    _rew_rms.update(ret)


def save_rms(rms_type, save_path, mean_name, var_name):
    if rms_type == 'state':
        rms = _state_rms
    elif rms_type == 'rew':
        rms = _rew_rms
    else:
        raise ValueError(f'Unknown rms type {rms_type}')
    save_obj(rms.mean, save_path, mean_name)
    save_obj(rms.var, save_path, var_name)


def load_rms(save_path, mean_name, var_name):
    # TODO Extend to other rms
    mean = load_obj(save_path, mean_name).to(global_device())
    var = load_obj(save_path, var_name).to(global_device())

    global _state_rms
    if _state_rms is not None:
        warnings.warn('Attempting to load state RMS multiple times')
    _state_rms = TorchRunningMeanStd(shape=mean.shape)
    _state_rms.mean = mean
    _state_rms.var = var


def state_rms():
    global _state_rms
    return _state_rms


def rew_rms():
    global _rew_rms
    return _rew_rms


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))
