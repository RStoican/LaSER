import abc

import torch

from laser.garage.envs._utils import gym_to_akro
from laser.garage.torch._functions import global_device
from laser.garage.utils import helpers as utl
from laser.garage.utils import metalearner_helpers as mlutl


class MetaLearner(abc.ABC):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.frames = 0
        self.iter_idx = -1

        self.transformer = None

        self._create_envs()

        # calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.traj_per_meta_traj

        # Update args to the input/output dimensions expected by the policy (i.e. state and action dimensions)
        mlutl.update_policy_input_dims(args, self.envs)

        # Time some of the methods
        self.timer_collect, self.timer_enc_update, self.timer_exploration_policy_update = 0, 0, 0
        self.timer_exploration_collect, self.timer_task_collect, self.timer_task_policy_update = 0, 0, 0

        # Make sure tasks don't change unexpectedly
        self.initial_task_ids, self.initial_tasks = None, None
        self.previous_task_ids, self.previous_tasks = None, None

        self._gym_to_akro()
        self._init_model()
        self._watch_training()

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def _init_model(self):
        pass

    @abc.abstractmethod
    def _watch_training(self):
        pass

    def _create_envs(self):
        self.envs = mlutl.ml_make_vec_envs(self.args, tasks=None)

    def _gym_to_akro(self):
        # We use akro-type environments
        self.envs = gym_to_akro(self.envs)

    def _encode_meta_traj(self, return_latest_latent=True, return_prior=False, starting_state=None,
                          transformer=None):
        """The exploration policy will never have to learn to work with multiple *dependent* meta-trajectories
            (i.e. online setting), so select only one from the latent
        Use the representation of the full meta-episode (i.e. of each timestep). This is different from
            "Transformers are Meta-Reinforcement Learners", Melo, 2022, where only the last step is used"""
        transformer = transformer if transformer is not None else self.transformer

        with torch.no_grad():
            if return_prior:
                return transformer.forward_exploration(
                    prior=True,
                    num_tasks=self.args.num_processes,
                    return_latest_latent=True,
                    starting_state=starting_state)  # (num_processes, d_z)
            assert starting_state is None

            # Get a dataset for the current tasks for which we collect data
            # This dataset only includes the current meta-trajectory. This is similar to the meta-testing case, where the
            # policy will not have additional meta-trajectories (i.e. one meta-trajectory per task)
            batch = transformer.dataset_storage.get_working_meta_traj().unsqueeze(1)  # (num_processes, 1, d, m*H)

            return transformer.forward_exploration(
                trajectories=batch,
                return_latest_latent=return_latest_latent)  # (num_processes, d_z)

    def _select_action(self, policy, prev_state, meta_traj_done_flag, latent, task_context=None):
        if latent is not None:
            assert prev_state.shape[0] == latent.shape[0], \
                f'Got shapes: {prev_state.shape}; {latent.shape}'
        if task_context is not None:
            assert prev_state.shape[0] == task_context.shape[0], \
                f'Got shapes: {prev_state.shape}; {task_context.shape}'

        with torch.no_grad():
            value, action = utl.select_action(
                args=self.args,
                policy=policy,
                state=prev_state,
                deterministic=False,
                latent=latent,
                task_context=task_context,
            )  # (num_processes, 1), (num_processes, action_dim)

            # If any of the meta-trajectories already have m episodes, then do not return an action for that
            if meta_traj_done_flag.sum() > 0:
                for proc in range(self.args.num_processes):
                    if meta_traj_done_flag[proc] == 1:
                        action_padding = torch.from_numpy(
                            utl.padding_action(self.envs.action_space)
                        ).float().to(global_device())
                        action[proc, :] = 0 * action[proc, :] + action_padding
            return value, action

    def _check_tasks(self, infos):
        self.initial_tasks, self.initial_task_ids = mlutl.check_tasks(infos,
                                                                      self.initial_tasks, self.initial_task_ids,
                                                                      self.previous_tasks, self.previous_task_ids)
        assert self.initial_tasks is not None and self.initial_task_ids is not None
