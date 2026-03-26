import argparse

import numpy as np
from laser.garage.utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------------------------------------------------------
    # ----- GENERAL -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--num_frames', type=int, default=2e7, help='number of frames to train')
    parser.add_argument('--exp_label', default='LaSER_MetaWorld', help='label (typically name of method)')
    parser.add_argument('--env_name', default=None, help='environment to train on')

    # ------------------------------------------------------------------------------------------------------------------
    # ----- ENVIRONMENT PARAMETERS -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--task_type', type=str, default=None)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- TRANSFORMER AND EXPLORATION TRAINING -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--num_processes', type=int, default=16,
                        help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--policy_num_steps', type=int, default=200,
                        help='number of env steps to do (per process) before updating')
    parser.add_argument('--num_vae_updates', type=int, default=3,
                        help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--lr_enc', type=float, default=0.0001, help='encoder learning rate (default: 3e-05)')
    parser.add_argument('--lr_policy_exploration', type=float, default=0.00001,
                        help='exploration policy learning rate (default: 3e-05)')
    parser.add_argument('--lr_policy_task', type=float, default=3e-05,
                        help='task policy learning rate (default: 3e-05)')
    parser.add_argument('--lr_critic_task', type=float, default=None,
                        help='task policy learning rate (default: 3e-05)')
    parser.add_argument('--pre_train_epochs', type=int, default=400,
                        help='the number of epochs to train the encoder and exploration policy')
    parser.add_argument('--task_train_epochs', type=int, default=1000)
    parser.add_argument('--transformer_pre_train_epochs', type=int, default=2000,
                        help='the number of epochs to train the encoder WITHOUT the exploration policy')
    parser.add_argument('--num_transformer_updates', type=int, default=50,
                        help='how many transformer update steps to take per meta-iteration')
    parser.add_argument('--tasks_per_iter', type=int, default=1,
                        help='from how many tasks to collect trajectories per iteration, per process')
    parser.add_argument('--batches_pre_collect', type=float, default=2.0,
                        help='number in [0, inf) of batches of task data to collect before we start training')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='probability of masking each step')
    parser.add_argument('--mask_prob_token', type=float, default=0.8,
                        help='probability of the masked step being a [MASK] token')
    parser.add_argument('--mask_prob_replace', type=float, default=0.1,
                        help='probability of the masked step being another step in the sequence')
    # FIXME Use these
    parser.add_argument('--traj_mask_prob_token', type=float, default=0.02,
                        help='self-supervised training: probability of replacing an entire trajectory in the input '
                             'with the [MASK] token')
    parser.add_argument('--traj_mask_prob_other', type=float, default=0.02,
                        help='self-supervised training: probability of replacing an entire trajectory in the input '
                             'with another trajectory from a different task')
    parser.add_argument('--mask_latest_action', type=boolean_argument, default=False,
                        help='mask or use the latest action when training the transformer. Masking means the '
                             'transformer must predict next obs without knowing the action')
    parser.add_argument('--mask_entire_loss', type=boolean_argument, default=False)
    parser.add_argument('--task_rec_loss_through_task_latent', type=boolean_argument, default=False,
                        help='backpropagate the task reconstruction loss through the task-specific latent')
    parser.add_argument('--task_rec_loss_through_traj_latent', type=boolean_argument, default=False,
                        help='backpropagate the task reconstruction loss through the traj-specific latent')
    parser.add_argument('--no_traj_latent_for_task_rec', type=boolean_argument, default=True)
    parser.add_argument('--use_target_net', type=boolean_argument, default=True)
    parser.add_argument('--target_net_update_iterations', type=int, default=250)
    parser.add_argument('--exploration_step_reward', type=float, default=0)
    parser.add_argument('--exploration_first_traj_reward', type=float, default=0)
    parser.add_argument('--exploration_exploit_tradeoff', type=float, default=0.0)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- DATASET -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--horizon', type=int, default=400,
                        help='the maximum length (i.e. number of steps) of a trajectory (i.e. H)')
    parser.add_argument('--traj_per_meta_traj', type=int, default=4,
                        help='number of trajectories per meta-trajectory (i.e. m)')
    parser.add_argument('--meta_traj_per_task', type=int, default=20,
                        help='the number of meta-trajectories per task (i.e. q)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='how many tasks to process at the same time (i.e. p)')
    parser.add_argument('--dataset_size', type=int, default=100000,
                        help='how many tasks to keep in the dataset buffer')
    parser.add_argument('--static_shared_latent_batches', type=int, default=10,
                        help='how many batches of tasks to use to compute a static shared latent. This latent will be '
                             'used when we do not have enough tasks to compute a proper shared latent (e.g. when '
                             'running a policy on a single meta-trajectory')
    parser.add_argument('--coeff_batch_size', type=int, default=5,
                        help='how many tasks to compute the linear coefficients matrix C for')
    parser.add_argument('--static_latent_num_batches', type=int, default=5)
    parser.add_argument('--static_shared_latent_transformation_type', type=str, default=None,
                        help='Choose: None, mean, single. How to reduce the static shared latent to one element')
    parser.add_argument('--starting_dataset', type=str, default=None,
                        help='Path to a pre-collected dataset. If None, do not use a starting dataset')
    parser.add_argument('--starting_dataset_len', type=int, default=-1,
                        help='How many tasks from the pre-collected dataset to load. Use -1 for loading all')
    parser.add_argument('--reset_starting_dataset', type=boolean_argument, default=False,
                        help='Empty the starting dataset when starting to collect new data')
    parser.add_argument('--save_train_tasks', type=boolean_argument, default=True,
                        help='save the meta-training task descriptions sampled during pre-training')
    parser.add_argument('--save_test_tasks', type=boolean_argument, default=True,
                        help='save the meta-testing task descriptions sampled during pre-training')

    # ------------------------------------------------------------------------------------------------------------------
    # ----- TRANSFORMER ARCHITECTURE -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--dim_ff', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=128, help='size of latent space (i.e. d_z)')
    parser.add_argument('--mlp_hidden_sizes', nargs='*', type=int,  default=(64,))
    parser.add_argument('--transformer_encoder_activation', type=str, default='gelu', help='relu/gelu')
    parser.add_argument('--latent_hidden_nonlinearity', type=str, default='tanh', help='tanh/relu/gelu')
    parser.add_argument('--latent_out_nonlinearity', type=str, default=None, help='tanh/relu/gelu')
    parser.add_argument('--reconstruction_hidden_nonlinearity', type=str, default='gelu', help='tanh/relu/gelu')
    # parser.add_argument('--latent_traj_dim', default=32, help='size of latent trajectory space (i.e. d_k)')
    parser.add_argument('--shared_latent_pooling_type', type=str, default='avg',
                        help='choose: max, avg, none. The type of pooling used for the input to the shared latent MLP')
    parser.add_argument('--simple_shared_latent', type=boolean_argument, default=True)
    parser.add_argument('--multiheaded_latent_network', type=boolean_argument, default=False, help='')
    parser.add_argument('--reconstruction_hidden_sizes', nargs='*', type=int, default=[32, 32])
    parser.add_argument('--linear_coeff_hidden_sizes', nargs='*', type=int, default=[128, 128, 128])
    parser.add_argument('--task_reconstruction_type', type=str, default='linear', help='linear/non-linear')
    parser.add_argument('--task_reconstruction_activation', type=str, default='tanh', help='tanh/relu/gelu')
    parser.add_argument('--task_reconstruction_latent_size', type=int, default=16)
    # parser.add_argument('--linear_coeff_use_shared', type=boolean_argument, default=False,
    #                     help='whether to use the shared latent when computing the linear coefficients matrix C')
    # parser.add_argument('--linear_coeff_use_traj', type=boolean_argument, default=False,
    #                     help='whether to use the traj-specific latent when computing the linear coefficients matrix C')
    parser.add_argument('--task_latent_type', type=str, default='attention', help='diagonal/orthogonal/attention')
    parser.add_argument('--task_latent_multidim', type=boolean_argument, default=True)
    parser.add_argument('--use_traj_latent_for_context', type=boolean_argument, default=False)
    parser.add_argument('--individual_exploration_episodes', type=boolean_argument, default=False)
    parser.add_argument('--final_transformer_projection', type=boolean_argument, default=False)
    parser.add_argument('--remove_ln', type=boolean_argument, default=True)
    parser.add_argument('--tfixup', type=boolean_argument, default=True)
    parser.add_argument('--dropout', default=0.0)
    parser.add_argument('--normalized_wm', default=False)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- LOSS -----
    # ------------------------------------------------------------------------------------------------------------------
    # Loss Coefficients
    parser.add_argument('--reconstruction_coeff', type=float, default=0.5, help='weight for reconstruction loss')
    parser.add_argument('--task_reconstruction_coeff', type=float, default=1.0,
                        help='weight for task reconstruction loss')
    parser.add_argument('--contrastive_coeff', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--regulariser_coeff', type=float, default=0.125, help='weight for regulariser')
    # TODO Try setting this to True
    parser.add_argument('--use_action_loss', type=boolean_argument, default=False, help='compute action loss')
    parser.add_argument('--state_loss_coeff', type=float, default=0.05, help='weight for state loss')
    parser.add_argument('--act_loss_coeff', type=float, default=0.2, help='weight for action loss')
    parser.add_argument('--rew_loss_coeff', type=float, default=1.0, help='weight for reward loss')
    parser.add_argument('--state_task_loss_coeff', type=float, default=0.05, help='weight for state loss')
    parser.add_argument('--act_task_loss_coeff', type=float, default=0.2, help='weight for action loss')
    parser.add_argument('--rew_task_loss_coeff', type=float, default=1.0, help='weight for reward loss')

    # Reconstruction Loss
    parser.add_argument('--masked_reconstruction_loss', type=boolean_argument, default=True,
                        help='if true, only compute the reconstruction loss between masked tokens (similar to BERT)')
    parser.add_argument('--split_reconstruction_loss', type=boolean_argument, default=True,
                        help='split the reconstruction loss into obs, act and rew losses, and then summ them')

    # Exploration Loss
    parser.add_argument('--exploration_similarity_measure', type=str, default='cosine_similarity',
                        help='similarity measure for contrastive and exploration rewards. '
                             'Choose between: dot_product, cosine_similarity (default)')
    parser.add_argument('--epoch_start_contrastive_loss', type=int, default=1000,
                        help='after how many epochs to start computing the exploration target and contrastive loss')
    parser.add_argument('--exploration_target_type', type=str, default='negative_exponential',
                        help='choose: squared_diff, exponential, negative_exponential, arctan. '
                             'The type of target to compute when optimising the exploration latent space')
    parser.add_argument('--exp_target_coeff', type=float, default=0.05,
                        help='only when exploration_target_type is negative_exponential or arctan. '
                             'Controls how fast the function grows')
    # TODO Try different values (probaboly False makes the most sense, otherwise exploration target will be dependent
    #  of other random tasks in the batch)
    parser.add_argument('--normalise_exploration_target', type=boolean_argument, default=False,
                        help='min-max normalise the exploration target distance')

    # Misc
    parser.add_argument('--loss_avg_features', type=boolean_argument, default=True,
                        help='Average over the loss of the features in a step (instead of sum)')

    # Regulariser
    parser.add_argument('--regularise_traj_latent', type=boolean_argument, default=True)
    parser.add_argument('--normalise_shared_latent', type=boolean_argument, default=True)
    parser.add_argument('--normalise_task_latent', type=boolean_argument, default=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- EXPLORATION AND TASK POLICIES -----
    # ------------------------------------------------------------------------------------------------------------------
    # General
    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--pass_state_to_policy', type=boolean_argument, default=True, help='condition policy on state')
    parser.add_argument('--pass_latent_to_policy', type=boolean_argument, default=True,
                        help='condition policy on VAE latent')
    parser.add_argument('--policy_state_embedding_dim', type=int, default=64)
    parser.add_argument('--policy_latent_embedding_dim', type=int, default=64)
    # TODO Try different values
    parser.add_argument('--policy_task_context_embedding_dim', type=int, default=128)
    parser.add_argument('--policy_initialisation', type=str, default='normc', help='normc/orthogonal')
    parser.add_argument('--policy_anneal_lr', type=boolean_argument, default=False, help='anneal LR over time')
    parser.add_argument('--policy_optimiser', type=str, default='adam', help='choose: rmsprop, adam')

    # Exploration Policy
    parser.add_argument('--policy', type=str, default='ppo', help='choose: ppo, sac, discrete_sac')
    parser.add_argument('--exploration_policy_gamma', type=float, default=0.99,
                        help='discount factor for (exploration) rewards')
    parser.add_argument('--exploration_avg_rewards', type=boolean_argument, default=True)
    # TODO Try different values for this (e.g. 0.05)
    parser.add_argument('--exploration_sigma', type=float, default=0.025)
    parser.add_argument('--normalise_exploration_rewards', type=boolean_argument, default=False)
    parser.add_argument('--policy_layers', nargs='*', type=int, default=[128, 128, 128])
    parser.add_argument('--policy_activation_function', type=str, default='tanh', help='tanh/relu/leaky-relu')
    parser.add_argument('--exploration_reward', type=str, default='gauss')

    # Task Policy
    parser.add_argument('--task_policy', type=str, default=None,
                        help='choose: ppo, sac, discrete_sac, None (same as exploration policy)')
    parser.add_argument('--policy_gamma', type=float, default=1.0, help='discount factor for (task) rewards')
    parser.add_argument('--batches_task_collect', type=int, default=3,
                        help='how many batches of tasks to collect (per iteration) for training the task policy')
    parser.add_argument('--task_policy_activation_function', type=str, default='tanh', help='tanh/relu/leaky-relu')
    parser.add_argument('--policy_context_activation_function', type=str, default='leaky-relu',
                        help='relu/leaky-relu/selu')
    parser.add_argument('--policy_context_hidden_layers', nargs='*', type=int, default=[128, 128])
    parser.add_argument('--context_type', type=str, default='full',
                        help='how much context to use in the task policy (full, traj_latest, latest)')
    parser.add_argument('--task_policy_layers', nargs='*', type=int, default=None,
                        help='The hidden layers of the task policy. If None, same as --policy_layers')
    # Hypernetwork for Task Policy
    parser.add_argument('--hypernetwork_task_policy', type=boolean_argument, default=False)
    parser.add_argument('--hypernetwork_layers', nargs='*', type=int, default=[32, 32])
    parser.add_argument('--hypernetwork_chunk_size', type=int, default=16)

    # Normalising (inputs/rewards/outputs)
    parser.add_argument('--norm_state_exploration', type=boolean_argument, default=False,
                        help='normalise state input for both exploration and task policy')
    parser.add_argument('--norm_state_task', type=boolean_argument, default=False,
                        help='normalise state input for task policy only')
    parser.add_argument('--norm_latent_for_policy', type=boolean_argument, default=False, help='normalise latent input')
    parser.add_argument('--norm_context', type=boolean_argument, default=False,
                        help='normalise task context input for task policy')
    parser.add_argument('--norm_rew_task', type=boolean_argument, default=False,
                        help='normalise rew for RL train')
    parser.add_argument('--norm_actions_pre_sampling', type=boolean_argument, default=False,
                        help='normalise policy output')
    parser.add_argument('--norm_actions_post_sampling', type=boolean_argument, default=False,
                        help='normalise policy output')

    # PPO specific
    parser.add_argument('--ppo_num_epochs', type=int, default=4, help='number of epochs per PPO update')
    parser.add_argument('--ppo_num_minibatch', type=int, default=2, help='number of minibatches to split the data')
    parser.add_argument('--ppo_use_huberloss', type=boolean_argument, default=True, help='use huberloss instead of MSE')
    parser.add_argument('--ppo_use_clipped_value_loss', type=boolean_argument, default=True, help='clip value loss')
    parser.add_argument('--ppo_clip_param', type=float, default=0.1, help='clamp param')
    parser.add_argument('--ppo_huber_beta', type=float, default=1, help='huber loss threshold')
    parser.add_argument('--ppo_exp_pfo', type=boolean_argument, default=False, help='use PFO for exploration policy')
    parser.add_argument('--ppo_task_pfo', type=boolean_argument, default=False, help='use PFO for task policy')

    # SAC specific
    parser.add_argument('--replay_buffer_size', type=int, default=100000)
    parser.add_argument('--sac_num_epochs', type=int, default=2000, help='number of epochs per SAC update')
    parser.add_argument('--sac_num_minibatch', type=int, default=16, help='number of minibatches to split the data')
    parser.add_argument('--sac_size_minibatch', type=int, default=64, help='size of each minibatch')
    parser.add_argument('--sac_information_bottleneck', type=boolean_argument, default=False,
                        help='use KL information bottleneck')
    parser.add_argument('--soft_target_tau', type=float, default=0.005, help='SAC target network update')

    # Discrete SAC specific
    parser.add_argument('--sac_use_temperature', type=boolean_argument, default=True)
    parser.add_argument('--temperature_lr', type=float, default=3e-06)

    # Misc
    parser.add_argument('--mask_token', type=int, default=-1,
                        help='the token used to mask trajectories\' steps for masked self-supervised learning')
    parser.add_argument('--save_full_shared_latent', type=boolean_argument, default=True)

    # ------------------------------------------------------------------------------------------------------------------
    # ----- OTHER HYPERPARAMETERS -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--policy_eps', type=float, default=1e-8, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy_init_std', type=float, default=1.0, help='only used for continuous actions')
    parser.add_argument('--policy_value_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.01, help='entropy term coefficient')
    parser.add_argument('--policy_pfo_coef', type=float, default=0.001, help='pfo term coefficient')
    parser.add_argument('--policy_use_gae', type=boolean_argument, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--exploration_policy_tau', type=float, default=0.9, help='gae parameter')
    parser.add_argument('--task_policy_tau', type=float, default=0.9, help='gae parameter')
    parser.add_argument('--use_meta_entropy', type=boolean_argument, default=False)
    parser.add_argument('--meta_entropy_loss', type=str, default='entropy')
    parser.add_argument('--meta_entropy_coeff', type=float, default=0.2)
    parser.add_argument('--meta_entropy_clip', type=float, default=0.9)
    parser.add_argument('--meta_entropy_value_weight', type=str, default='ratio',
                        help='how to create the value function weight. Options: ratio (default), exp_diff')
    parser.add_argument('--meta_entropy_exp_coeff', type=float, default=1.0)
    parser.add_argument('--meta_entropy_mask_diff', type=boolean_argument, default=False)
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--encoder_max_grad_norm', type=float, default=None, help='max norm of gradients')

    # ------------------------------------------------------------------------------------------------------------------
    # ----- DATASET PRE-COLLECT -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--precollect_dataset_size', type=int, default=5000)
    parser.add_argument('--precollect_policy_type', type=str, default='uniform')
    parser.add_argument('--dataset_output_folder', type=str, default=None)
    parser.add_argument('--seed_dataset', type=int, default=int(np.random.choice(65536, replace=False, size=1)[0]))

    # ------------------------------------------------------------------------------------------------------------------
    # ----- ABLATIONS -----
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--ablation_use_history', type=boolean_argument, default=True)
    parser.add_argument('--ablation_use_context', type=boolean_argument, default=True,
                        help='whether to use latent task context during evaluation')
    parser.add_argument('--ablation_true_task', type=boolean_argument, default=False)
    parser.add_argument('--ablation_fixed_tasks', type=boolean_argument, default=False)

    # Transformer
    parser.add_argument('--disable_decoder', type=boolean_argument, default=False,
                        help='train without decoder')
    parser.add_argument('--decode_only_past', type=boolean_argument, default=False,
                        help='only decoder past observations, not the future')

    # Combining transformer and RL loss
    parser.add_argument('--rlloss_through_encoder', type=boolean_argument, default=False,
                        help='backprop rl loss through encoder')
    parser.add_argument('--finetune_to_task', type=boolean_argument, default=False,
                        help='fine-tune by backpropagating the RL loss of the task policy through the transformer')
    parser.add_argument('--lr_finetune', type=float, default=3e-06,
                        help='encoder learning rate for fine-tuning (default: 3e-06)')
    parser.add_argument('--add_nonlinearity_to_latent', type=boolean_argument, default=False,
                        help='Use relu before feeding latent to policy')
    parser.add_argument('--vae_loss_coeff', type=float, default=1.0,
                        help='weight for VAE loss (vs RL loss)')

    # Other
    parser.add_argument('--disable_metalearner', type=boolean_argument, default=False,
                        help='Train feedforward policy')
    parser.add_argument('--single_task_mode', type=boolean_argument, default=False,
                        help='train policy on one (randomly chosen) environment only')

    # ------------------------------------------------------------------------------------------------------------------
    # ----- OTHERS -----
    # ------------------------------------------------------------------------------------------------------------------
    # Logging, saving, evaluation
    # FIXME Check these values
    parser.add_argument('--log_interval', type=int, default=1, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=200, help='save interval, one save per n updates')
    parser.add_argument('--save_intermediate_models', type=boolean_argument, default=False, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=25, help='eval interval, one eval per n updates')
    parser.add_argument('--test_interval', type=int, default=-1, help='eval interval, one eval per n updates')
    parser.add_argument('--vis_interval', type=int, default=25, help='visualisation interval, one eval per n updates')
    parser.add_argument('--results_log_dir', help='directory to save results (None uses ./logs)')
    parser.add_argument('--eval_total_tasks', type=int, default=20, help='no. of distinct task to eval on')
    parser.add_argument('--eval_repeat_task', type=int, default=50)
    parser.add_argument('--eval_repeat_task_traj', type=int, default=10)
    parser.add_argument('--run_test', type=boolean_argument, default=False)
    parser.add_argument('--test_repeat_task', type=int, default=100)
    parser.add_argument('--test_repeat_task_traj', type=int, default=20)

    # Model names
    parser.add_argument('--task_policy_name', type=str, default='task_policy')

    # General settings
    seed = np.random.choice(65536, replace=False, size=1)
    seed = [int(s) for s in seed]
    parser.add_argument('--seed', nargs='+', type=int, default=seed)
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')
    parser.add_argument('--task_seed_type', type=str, default='generated',
                        help="seed to use for task training. Options: 'generated (default)', 'random', an integer")
    parser.add_argument('--task_use_gpu', type=boolean_argument, default=True, help='use GPU for task policy')
    parser.add_argument('--torch_envs', type=boolean_argument, default=False, help='use PyTorch for envs')

    return parser.parse_args(rest_args)
