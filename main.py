"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""

import argparse
import warnings

import numpy as np
import torch

from config.metaworld import args_metaworld_ml1
from config.mewa import args_mewa
from config.mujoco import args_mujoco
from laser.garage.envs.parallel_envs import make_vec_envs
from laser.runner import Runner


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--env-type', type=str, required=True)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--exp-train', action=argparse.BooleanOptionalAction, default=True,
                        help='run exploration pre-training')
    parser.add_argument('--task-train', action=argparse.BooleanOptionalAction, default=True,
                        help='run task policy training')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='enable wandb logging')
    setup_args, rest_args = parser.parse_known_args()

    assert setup_args.exp_train or setup_args.task_train, 'At least one of exp-train or task-train must be True'

    if setup_args.env_type == 'mewa':
        args = args_mewa.get_args(rest_args)
        args.mode = None
        if args.ablation_fixed_tasks:
            args.env_name = 'MEWATaskAblation-v0'
    elif setup_args.env_type == 'metaworld_ml1':
        args = args_metaworld_ml1.get_args(rest_args)
        args.mode = 'train'
    elif setup_args.env_type == 'metaworld_ml10':
        args = args_metaworld_ml1.get_args(rest_args)
        args.env_name = 'MetaWorldML10-v0'
        args.mode = 'train'
    elif setup_args.env_type == 'mujoco':
        args = args_mujoco.get_args(rest_args)
        args.env_name = f'MuJoCo{args.task_type}-v5'
        args.mode = 'train'
    else:
        raise Exception(f"Invalid Environment: {setup_args.env_type}")

    for key, value in vars(setup_args).items():
        setattr(args, key, value)

    check_args(args)

    runner = Runner(args, exp_train=setup_args.exp_train, task_train=setup_args.task_train)
    runner.run()


def check_args(args):
    if not args.final_transformer_projection:
        if args.latent_dim != args.d_model:
            args.latent_dim = args.d_model
            warnings.warn(f'When final_transformer_projection is False, '
                          f'latent_dim({args.latent_dim}) == d_model({args.d_model}) must hold. '
                          f'Setting latent_dim = d_model')
    else:
        raise NotImplementedError

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        # FIXME Pass environment arguments as kwargs
        if args.env_name == 'MEWASymbolic-v0':
            envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                                 gamma=args.task_policy_gamma, device='cpu',
                                 episodes_per_task=args.max_rollouts_per_task,
                                 normalise_rew=args.norm_rew_task,
                                 tasks=None,
                                 task_path=args.task,
                                 wide_tasks=args.wide if 'wide' in args else 1,
                                 narrow_tasks=args.narrow,
                                 complex_worker=(args.worker == 'complex')
                                 )
        elif args.env_name == 'MetaWorldML1-v0' or args.env_name == 'MetaWorldML10-v0':
            envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                                 gamma=args.task_policy_gamma, device='cpu',
                                 episodes_per_task=args.traj_per_meta_traj,
                                 normalise_rew=args.norm_rew_task,
                                 tasks=None,
                                 task_type=args.task_type,
                                 mode='train',
                                 )
        else:
            raise ValueError(f'Unknown environment {args.env_name}')
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False


if __name__ == '__main__':
    # Required for PyTorch multiprocessing to work with CUDA
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    main()
