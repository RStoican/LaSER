import copy
import os

import gymnasium as gym
import torch

from laser.garage._environment import EnvSpec
from laser.garage.envs.parallel_envs import make_vec_envs
from laser.garage.storage.online_storage import OnlineStorage
from laser.garage.torch._functions import global_device
from laser.garage.torch.algos.ppo import PPO
from laser.garage.torch.memory_transformers.bidirectional_transformer_encoder import BidirectionalTransformerEncoder
from laser.garage.torch.memory_transformers.unidirectional_transformer_encoder import UnidirectionalTransformerEncoder
from laser.garage.torch.policies.policy import Policy
from laser.garage.utils import helpers as utl
from transformer import Transformer


class MetaLearnerInitialiser:
    def __init__(self,
                 args,
                 envs,
                 exploration_train_mode,
                 get_iter_idx,
                 logger,
                 use_starting_dataset=None,
                 ):
        self.args = args
        self.envs = envs
        self.exploration_train_mode = exploration_train_mode
        self.get_iter_idx = get_iter_idx
        self.logger = logger
        self.use_starting_dataset = use_starting_dataset

        # The dimension of the task context used by the task policy
        if self.args.context_type == 'full':
            self.task_context_dim = self.args.latent_dim * self.args.horizon * self.args.traj_per_meta_traj  # H*d_z*m
        elif self.args.context_type == 'traj_latest':
            raise NotImplementedError  # d_z*m
        elif self.args.context_type == 'latest':
            self.task_context_dim = self.args.latent_dim  # d_z
        else:
            raise ValueError(f'Unknown context type {self.args.context_type}')

        assert not exploration_train_mode or (exploration_train_mode and use_starting_dataset is not None)

        # If we are training the transformer and exploration policy, we collect q meta-episodes per task.
        # If we are training the task policy or evaluating, we collect only 1
        self.meta_traj_per_task = args.meta_traj_per_task if exploration_train_mode else 1

        self.env_spec = EnvSpec(
            observation_space=envs.observation_space,
            action_space=envs.action_space,
            max_episode_length=args.horizon,
        )

        # FIXME This is the number of overall updates. Might need to change to number of ppo updates (after pretrain)
        self.ppo_total_epochs = self.args.pre_train_epochs if exploration_train_mode else self.args.task_train_epochs

    def initialise_autoencoder(self, lr=None, finetune_mode=False):
        uni_encoder = self._create_encoder(attention_type='unidirectional',
                                           name='unidirectional transformer')
        bi_encoder = self._create_encoder(attention_type='bidirectional',
                                          name='bidirectional transformer')

        transformer = self._create_transformer(uni_encoder, bi_encoder, lr, finetune_mode)
        transformer.to(global_device())

        if self.exploration_train_mode:
            target_coeff_head = self._create_target_net(transformer) if self.args.use_target_net else None
            return transformer, target_coeff_head
        return transformer

    def initialise_policy(self, policy_type):
        is_task_policy = self._get_policy_type(policy_type)

        policy_lr = self.args.lr_policy_task if is_task_policy else self.args.lr_policy_exploration
        critic_lr = self.args.lr_critic_task if is_task_policy else None

        task_context_dim = self.task_context_dim if is_task_policy else None
        task_context_dim = task_context_dim if self.args.ablation_use_context else None

        return self._ppo_policy(is_task_policy, policy_lr, critic_lr, task_context_dim)

    def initialise_policy_storage(self, policy_type):
        is_task_policy = self._get_policy_type(policy_type)

        # For the task policy, we will collect several (unrelated) trajectories from the same task
        num_tasks = self.args.tasks_per_iter * self.args.num_processes
        num_tasks *= self.args.batches_task_collect if is_task_policy else 1

        # We only collect one trajectory per task for the task policy
        traj_per_meta_traj = 1 if is_task_policy else self.args.traj_per_meta_traj

        # For the task policy, we also have to save the task context
        task_context_dim = self.task_context_dim if (is_task_policy and self.args.ablation_use_context) else None

        # The original VariBAD computes the latent space based on the last adaptation episodes, i.e. a meta-episode
        # (e.g. 10 episodes by default). When a meta-episode is finished, we reset the hidden states. This means the
        # policy buffer only needs to remember the last episode, as the meta-episode is already encoded in the RNN
        # encoder hidden state
        #
        # For the transformer case, we don't use the hidden states, so just recompute the latent space from the
        # (m*H, m*H) attention matrix, where m=len(meta_episode) and H=len(episode). So, the policy storage must
        # remember the entire meta_episode, not just the last episode + hidden_state
        return OnlineStorage(args=self.args,
                             num_tasks=num_tasks,
                             meta_traj_per_task=self.meta_traj_per_task,
                             traj_per_meta_traj=traj_per_meta_traj,
                             horizon=self.args.horizon,
                             num_processes=self.args.num_processes,
                             state_dim=self.args.state_dim,
                             latent_dim=self.args.latent_dim,
                             action_space=self.args.action_space,
                             normalise_rewards=is_task_policy and self.args.norm_rew_task,
                             use_exploration_rewards=not is_task_policy,
                             task_context_dim=task_context_dim,
                             use_meta_entropy=self.args.use_meta_entropy if is_task_policy else False,
                             gamma=self.args.task_policy_gamma,
                             tau=self.args.task_policy_tau,
                             )

    def initialise_shared_latent(self):
        dim = self.args.latent_dim * self.args.horizon
        if self.args.save_full_shared_latent:
            return torch.zeros(dim, self.args.static_shared_latent_batches)  # (H*d_z, num_batches)
        return torch.zeros(dim)  # (H*d_z)

    def _get_policy_type(self, policy_type):
        """
        Policy type is exploration or task.

        :return: is_task_policy, policy_algo
        """

        assert policy_type == 'exploration' or policy_type == 'task', \
            f'Expected the policy type to be: exploration or task. Got: {policy_type}'
        is_task_policy = policy_type == 'task'

        return is_task_policy

    def _create_encoder(self, attention_type, name):
        if attention_type == 'unidirectional':
            encoder_architecture = UnidirectionalTransformerEncoder
        elif attention_type == 'bidirectional':
            encoder_architecture = BidirectionalTransformerEncoder
        else:
            raise ValueError(f'Expected the type of attention to be unidirectional or bidirectional. '
                             f'Got {attention_type}')

        return encoder_architecture(name=name,
                                    env_spec=self.env_spec,
                                    action_dim=self.args.action_dim,
                                    latent_hidden_sizes=self.args.mlp_hidden_sizes,
                                    latent_hidden_nonlinearity=self.args.latent_hidden_nonlinearity,
                                    out_latent_dim=self.args.latent_dim,
                                    shared_latent_pooling_type=self.args.shared_latent_pooling_type,
                                    simple_shared_latent=self.args.simple_shared_latent,
                                    latent_output_nonlinearity=self.args.latent_out_nonlinearity,
                                    nhead=self.args.n_heads,
                                    d_model=self.args.d_model,
                                    dropout=self.args.dropout,
                                    num_encoder_layers=self.args.layers,
                                    dim_feedforward=self.args.dim_ff,
                                    transformer_encoder_activation=self.args.transformer_encoder_activation,
                                    traj_per_meta_traj=self.args.traj_per_meta_traj,
                                    max_trajectory_len=self.args.horizon,
                                    meta_train_tasks=self.args.batch_size,
                                    meta_train_meta_trajs=self.meta_traj_per_task,
                                    task_latent_type=self.args.task_latent_type,
                                    task_latent_multidim=self.args.task_latent_multidim,
                                    tfixup=self.args.tfixup,
                                    remove_ln=self.args.remove_ln,
                                    normalize_wm=self.args.normalized_wm,
                                    multiheaded_latent_network=self.args.multiheaded_latent_network,
                                    final_transformer_projection=self.args.final_transformer_projection,
                                    train_mode=self.exploration_train_mode,
                                    )

    def _create_transformer(self, uni_encoder, bi_encoder, lr=None, finetune_mode=False):
        lr = self.args.lr_enc if lr is None else lr

        # If not training, reset the transformer storage after each data collection phase. So, no need for large storage
        dataset_size = self.args.dataset_size if self.exploration_train_mode else self.args.tasks_per_iter * self.args.num_processes

        # We only use a starting dataset when training the transformer
        # TODO We can pre-collect exploration data at the end of training and use it to train the task policy
        dataset = {'use_starting_dataset': self.use_starting_dataset,
                   'path': self.args.starting_dataset,
                   'len': self.args.starting_dataset_len} if self.exploration_train_mode else None

        # The dataset size is the amount of data per iteration. Since we are not updating the transformer, we don't
        # need the full dataset_size dataset
        return Transformer(self.args, uni_encoder, bi_encoder, self.logger, self.get_iter_idx,
                           env_spec=self.env_spec,
                           action_dim=self.args.action_dim,
                           lr=lr,
                           dataset_size=dataset_size,
                           out_latent_dim=self.args.latent_dim,
                           mlp_hidden_nonlinearity=self.args.reconstruction_hidden_nonlinearity,
                           d_model=self.args.d_model,
                           traj_per_meta_traj=self.args.traj_per_meta_traj,
                           max_trajectory_len=self.args.horizon,
                           meta_train_tasks=self.args.batch_size,
                           meta_train_meta_trajs=self.meta_traj_per_task,
                           reconstruction_hidden_sizes=self.args.reconstruction_hidden_sizes,
                           linear_coeff_hidden_sizes=self.args.linear_coeff_hidden_sizes,
                           masked_reconstruction_loss=self.args.masked_reconstruction_loss,
                           use_traj_latent_for_context=self.args.use_traj_latent_for_context,
                           norm_state=self.args.norm_state_exploration,
                           dataset=dataset,
                           train_mode=self.exploration_train_mode,
                           finetune_mode=finetune_mode,
                           store_norm_rewards=not self.exploration_train_mode and self.args.norm_rew_task,
                           )

    def _create_target_net(self, transformer):
        return copy.deepcopy(transformer.linear_coeff_head)

    def _ppo_policy(self, is_task_policy, policy_lr, critic_lr, task_context_dim):
        # A neural network including both the actor and critic (i.e. policy and value function) networks
        # The task policy will use task context
        activation_function = self.args.policy_activation_function if not is_task_policy \
            else self.args.task_policy_activation_function

        hidden_layers = self.args.policy_layers \
            if not is_task_policy or self.args.task_policy_layers is None else self.args.task_policy_layers

        if self.exploration_train_mode:
            policy_initialisation = self.args.policy_initialisation
        else:
            policy_initialisation = self.args.policy_initialisation if activation_function != 'leaky-relu' else None

        pfo_coef = self.args.policy_task_pfo_coef if is_task_policy else self.args.policy_exp_pfo_coef
        policy_anneal_lr = self.args.policy_task_anneal_lr if is_task_policy else self.args.policy_exp_anneal_lr

        ppo_num_epochs = self.args.ppo_task_num_epochs if is_task_policy else self.args.ppo_exp_num_epochs
        ppo_num_minibatch = self.args.ppo_task_num_minibatch if is_task_policy else self.args.ppo_exp_num_minibatch
        ppo_clip_param = self.args.ppo_task_clip_param if is_task_policy else self.args.ppo_exp_clip_param

        if is_task_policy and self.args.hypernetwork_task_policy:
            from laser.garage.torch.policies.hyper_policy import HyperPolicy
            policy_type = HyperPolicy
        else:
            policy_type = Policy

        # A neural network including both the actor and critic (i.e. policy and value function) networks
        actor_critic_policy_net = policy_type(
            args=self.args,
            #
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=self.args.pass_latent_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim,
            # Use only in task policy:
            pass_task_context_to_policy=is_task_policy if self.args.ablation_use_context else False,
            dim_task_context=task_context_dim,
            #
            hidden_layers=hidden_layers,
            activation_function=activation_function,
            policy_initialisation=policy_initialisation,
            # The task policy will use task context
            context_activation_function=self.args.policy_context_activation_function if is_task_policy else None,
            context_hidden_layers=self.args.policy_context_hidden_layers if is_task_policy else None,
            #
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
            # normalisation
            norm_state=self.args.norm_state_exploration if not is_task_policy else self.args.norm_state_task,
            norm_latent=self.args.norm_latent_for_policy,
            norm_context=is_task_policy and self.args.norm_context,
        ).to(global_device())

        return PPO(
            self.args,
            actor_critic_policy_net,
            self.args.policy_value_loss_coef,
            self.args.policy_entropy_coef,
            pfo_coef=pfo_coef,
            policy_optimiser=self.args.policy_optimiser,
            policy_anneal_lr=policy_anneal_lr,
            train_steps=self.ppo_total_epochs,
            lr=policy_lr,
            critic_lr=critic_lr,
            eps=self.args.policy_eps,
            ppo_epoch=ppo_num_epochs,
            num_mini_batch=ppo_num_minibatch,
            use_huber_loss=self.args.ppo_use_huberloss,
            use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
            clip_param=ppo_clip_param,
            update_state_normaliser=is_task_policy and self.args.norm_state_task and not self.args.norm_state_exploration,
            update_context_normaliser=is_task_policy and self.args.norm_context,
        )


def ml_make_vec_envs(args, tasks, loaded_tasks=None, mode=None, eval=False):
    mode = args.mode if mode is None else mode
    # FIXME Pass environment arguments as kwargs
    if 'MEWA' in args.env_name:
        return make_vec_envs(env_name=args.env_name, seed=args.seed,
                             num_processes=args.num_processes,
                             gamma=args.task_policy_gamma, device=global_device(),
                             h=args.horizon - 1,
                             episodes_per_task=args.traj_per_meta_traj,
                             normalise_rew=args.norm_rew_task,
                             tasks=tasks,
                             task_path=args.task,
                             wide_tasks=args.wide if 'wide' in args else 1,
                             narrow_tasks=args.narrow,
                             complex_worker=args.worker == 'complex',
                             split_dict=utl.create_split_dict(args, train_task_count=args.narrow,
                                                              test_task_count=0),
                             uniform_human=args.uniform_human,
                             )
    if args.env_name == 'MetaWorldML1-v0'or args.env_name == 'MetaWorldML10-v0':
        return make_vec_envs(env_name=args.env_name, seed=args.seed,
                             num_processes=args.num_processes,
                             gamma=args.task_policy_gamma, device=global_device(),
                             h=args.horizon - 1,
                             episodes_per_task=args.traj_per_meta_traj,
                             normalise_rew=args.norm_rew_task,
                             tasks=tasks,
                             task_type=args.task_type,
                             mode=mode,
                             given_tasks=loaded_tasks,
                             eval=eval,
                             )
    if 'MuJoCo' in args.env_name:
        return make_vec_envs(env_name=args.env_name, seed=args.seed,
                             num_processes=args.num_processes,
                             gamma=args.task_policy_gamma, device=global_device(),
                             h=args.horizon - 1,
                             episodes_per_task=args.traj_per_meta_traj,
                             normalise_rew=args.norm_rew_task,
                             tasks=tasks,
                             mode=mode,
                             eval=eval,
                             )
    raise ValueError(f'Unknown environment {args.env_name}')


def save_models(args, iter_idx, full_output_folder,
                transformer=None, exploration_policy=None, task_policy=None, ):
    if (iter_idx + 1) % args.save_interval == 0:
        save_path = os.path.join(full_output_folder, 'models')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        idx_labels = ['']
        if args.save_intermediate_models:
            padding = (iter_idx >= 0) * (5 - len(str(iter_idx))) * '0'
            idx_labels.append(f'_i{padding}{iter_idx}')

        # Compute the shared latent from a batch of tasks
        shared_latent = transformer.compute_static_shared_latent(
            batch_size=args.batch_size,
            num_batches=args.static_latent_num_batches,
            return_full_latent=args.save_full_shared_latent,
            update=False) if transformer is not None else None

        for idx_label in idx_labels:
            # Save transformer (including training heads)
            if transformer is not None:
                save_model_state_dict(transformer.get_headless_state_dict(), save_path, 'transformer', idx_label)
                if idx_label == '':
                    save_model(transformer.unembedding_head, save_path, 'unembedding_head', idx_label)
                    save_model(transformer.linear_coeff_head, save_path, 'linear_coeff_head', idx_label)
                    save_model(transformer.uni_encoder, save_path, 'uni_encoder', idx_label)
                    save_model(transformer.bi_encoder, save_path, 'bi_encoder', idx_label)

            # Save the shared latent
            if shared_latent is not None:
                torch.save(shared_latent, os.path.join(save_path, f'shared_latent{idx_label}.pt'))

            # Save task and exploration policies
            if exploration_policy is not None:
                exploration_policy.save(os.path.join(save_path, f'exploration_policy{idx_label}.pt'))
            if task_policy is not None:
                task_policy.save(os.path.join(save_path, f'{args.task_policy_name}{idx_label}.pt'))

            if args.norm_state_exploration:
                if task_policy is None:
                    utl.save_rms('state',
                                 save_path,
                                 f'exploration_state_rms_mean{idx_label}',
                                 f'exploration_state_rms_var{idx_label}')

            # TODO Save task policy rms

            # Save normalisation params of envs
            if args.norm_rew_task:
                if task_policy is not None:
                    utl.save_rms('rew',
                                 save_path,
                                 f'reward_rms_mean{idx_label}',
                                 f'reward_rms_var{idx_label}')


def save_model(model, save_path, model_name, idx_label=''):
    torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}{idx_label}.pt'))


def save_model_state_dict(model_state_dict, save_path, model_name, idx_label=''):
    torch.save(model_state_dict, os.path.join(save_path, f'{model_name}{idx_label}.pt'))


def load_models(save_path,
                label,
                transformer=None,
                exploration_policy=None,
                shared_latent=None,
                task_policy=None,
                task_policy_name=None,
                set_eval=True,
                norm_state_exploration=False):
    models_path = os.path.join(save_path, 'models')

    label = '' if label is None else label
    if label != '':
        padding = (int(label) >= 0) * (5 - len(label)) * '0'
        label = f'_{padding}{label}'

    if transformer is not None:
        transformer.load_state_dict(torch.load(os.path.join(models_path, f'transformer{label}.pt')))

    if exploration_policy is not None:
        exploration_policy.actor_critic.load_state_dict(torch.load(os.path.join(models_path,
                                                                                f'exploration_policy{label}.pt')))

    if task_policy is not None:
        task_policy_name = f'task_policy{label}.pt' if task_policy_name is None else task_policy_name
        task_policy.actor_critic.load_state_dict(torch.load(os.path.join(models_path, task_policy_name)))

    if shared_latent is not None:
        assert transformer is not None
        shared_latent = torch.load(os.path.join(models_path, f'shared_latent{label}.pt'))
        transformer.set_static_shared_latent(shared_latent)

    if norm_state_exploration:
        utl.load_rms(models_path,
                     f'exploration_state_rms_mean{label}',
                     f'exploration_state_rms_var{label}')

    if set_eval:
        if transformer is not None:
            transformer.eval()
            assert not (transformer.training or transformer.uni_encoder.training or transformer.bi_encoder.training)
        if exploration_policy is not None:
            exploration_policy.actor_critic.eval()
            assert not exploration_policy.actor_critic.training
        if task_policy is not None:
            task_policy.actor_critic.eval()
            assert not task_policy.actor_critic.training


def create_single_task_envs(args, envs, output_dir):
    # get the current tasks (which will be num_process many different tasks)
    train_tasks = envs.get_task()
    # set the tasks to the first task (i.e. just a random task)
    train_tasks[1:] = train_tasks[0]
    # make it a list
    train_tasks = [t for t in train_tasks]
    # re-initialise environments with those tasks
    envs = ml_make_vec_envs(args, tasks=train_tasks)
    # save the training tasks, so we can evaluate on the same envs later
    utl.save_obj(train_tasks, output_dir, "train_tasks")
    return envs, train_tasks


def update_policy_input_dims(args, envs):
    args.state_dim = envs.observation_space.shape[0]  # Actually state_dim + 1 (for the "done" flag)
    args.num_states = envs.num_states
    # get policy output (action) dimensions
    args.action_space = envs.action_space
    if isinstance(envs.action_space, gym.spaces.discrete.Discrete):
        args.action_dim = 1
    else:
        args.action_dim = envs.action_space.shape[0]


def save_tasks(args, envs, output_dir):
    # Assumes all environments have the same list of tasks
    train_tasks, test_tasks = None, None

    if args.save_train_tasks:
        train_tasks = envs.train_tasks  # (num_processes)
        assert len(train_tasks) == args.num_processes
        train_tasks = train_tasks[0]
        utl.save_obj(train_tasks, output_dir, 'train_tasks')

    if args.save_test_tasks:
        test_tasks = envs.test_tasks  # (num_processes)
        assert len(test_tasks) == args.num_processes
        test_tasks = test_tasks[0]
        utl.save_obj(test_tasks, output_dir, 'test_tasks')

    return train_tasks, test_tasks


def load_tasks(args, output_dir):
    train_tasks, test_tasks = None, None
    if args.save_train_tasks:
        train_tasks = utl.load_obj(output_dir, 'train_tasks')
    if args.save_test_tasks:
        test_tasks = utl.load_obj(output_dir, 'test_tasks')
    return (train_tasks, test_tasks) if train_tasks is not None or test_tasks is not None else None


def check_tasks(infos,
                initial_tasks, initial_task_ids,
                previous_tasks, previous_task_ids):
    # Check that the tasks we are currently using are valid
    if initial_task_ids is None:
        initial_task_ids = [info['task_id'] for info in infos]
    if initial_tasks is None:
        initial_tasks = [info['task'] for info in infos]

    # Make sure the tasks haven't changed since the last step or trajectory
    assert [info['task_id'] for info in infos] == initial_task_ids
    assert [info['task'] for info in infos] == initial_tasks
    return initial_tasks, initial_task_ids
