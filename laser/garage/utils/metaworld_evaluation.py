# FIXME Combine with evaluation.py

import copy

import numpy as np
from tqdm import trange

from laser.garage.torch import global_device
from laser.garage.utils import metalearner_helpers as mlutl
from metalearner_eval import EvalMetaLearner
from laser.garage.utils import helpers as utl
import torch


class MetaWorldTester(EvalMetaLearner):
    def __init__(self, args, envs, repeat_task, repeat_task_traj):
        args = copy.deepcopy(args)
        args.repeat_task = repeat_task
        args.repeat_task_traj = repeat_task_traj
        args.mode = self.task_mode

        # FIXME Not very efficient, but makes implementation easier. Maybe remove in the future
        args.num_processes = 1

        # FIXME This assumes the exploration policy was trained with pass_latent_to_policy == True
        args.pass_latent_to_policy = True

        args.task_policy_name = f'{args.task_policy_name}.pt'

        new_seed = int(np.random.choice(65536, replace=False, size=1)[0])
        print(f'Using evaluation seed {new_seed}')
        args.seed = new_seed

        super().__init__(args, use_logger=False, verbose=False)

        self.envs = mlutl.ml_make_vec_envs(self.args, tasks=None, loaded_tasks=self.get_eval_tasks(envs))
        for i in range(len(self.envs.envs[0].tasks_for_mode)):
            assert self.envs.envs[0].tasks_for_mode[i] == self._get_tasks_for_check(envs)[0][i], \
                f'{self.envs.envs[0].tasks_for_mode[i]}\n{self._get_tasks_for_check(envs)[0][i]}'

    @property
    def task_mode(self):
        return 'test'

    def _get_tasks_for_check(self, envs):
        return envs.venv.test_tasks

    def get_eval_tasks(self, envs):
        if self.task_mode == 'train':
            return envs.venv.train_tasks[0], None
        if self.task_mode == 'test':
            return None, envs.venv.test_tasks[0]
        raise ValueError('Task mode must be test or train')

    def run_test(self):
        self.envs.envs[0].set_mode(self.task_mode)
        self.task_policy.actor_critic.eval()

        with torch.no_grad():
            test_stats = self._run_test()

        self.task_policy.actor_critic.train()
        self.envs.envs[0].set_mode('train')
        return test_stats

    def _run_test(self):
        assert not self.args.ablation_use_history

        pre_adapt_task_returns_per_task, pre_adapt_success_per_task = 0, 0
        task_returns_per_task, success_per_task = 0, 0

        for task_id in trange(len(self.envs.envs[0].unwrapped.tasks_for_mode)):
            returns, success = self._exploit_task(task_id,
                                                  explore=False)
            pre_adapt_task_returns_per_task += returns
            pre_adapt_success_per_task += success

            returns, success = self._exploit_task(task_id,
                                                  explore=True)
            task_returns_per_task += returns
            success_per_task += success

        pre_adapt_task_returns = pre_adapt_task_returns_per_task / len(self.envs.envs[0].unwrapped.tasks_for_mode)
        pre_adapt_success = pre_adapt_success_per_task / len(self.envs.envs[0].unwrapped.tasks_for_mode)
        task_returns = task_returns_per_task / len(self.envs.envs[0].unwrapped.tasks_for_mode)
        success = success_per_task / len(self.envs.envs[0].unwrapped.tasks_for_mode)

        print('pre_adapt_task_returns', pre_adapt_task_returns)
        print('pre_adapt_success ', pre_adapt_success)
        print('task_returns ', task_returns)
        print('success ', success)

        return {
            # Success
            'test/success': success,
            'test/pre_adapt_success': pre_adapt_success,
            'test/delta_success': success - pre_adapt_success,

            # Returns
            'test/return': task_returns,
            'test/pre_adapt_return': pre_adapt_task_returns,
            'test/delta_return': task_returns - pre_adapt_task_returns,
        }

    def _exploit_task(self, task_id, explore):
        task_returns_per_episode = 0
        success_per_episode = 0

        for i in range(self.args.test_repeat_task):
            context = self._explore_task(task_id, explore)

            self.envs.envs[0].unwrapped.reset_task(task_id)
            for _ in range(self.args.repeat_task_traj):
                prev_state = utl.reset_meta_traj(self.envs)  # (num_processes, d_obs)
                if len(prev_state.shape) == 1:
                    prev_state = prev_state.unsqueeze(0)

                timestep = 0
                done = False
                while timestep < self.args.horizon - 1 and not done:
                    _, action = self._select_action(self.task_policy,
                                                        prev_state,
                                                        None,
                                                        np.array([int(done)]),
                                                        task_context=context,
                                                        )  # (num_processes, 1), (num_processes, action_dim)

                    next_state, (_, _), _, infos = utl.env_step(self.envs, action, self.args)

                    info = infos[0]
                    if len(next_state.shape) == 1:
                        next_state = next_state.unsqueeze(0)

                    # add rewards
                    task_returns_per_episode += info['unscaled_reward']

                    if int(info['success']) == 1:
                        done = True

                    prev_state = next_state

                    timestep += 1
                success_per_episode += int(done)

        exploit_episodes_count = self.args.repeat_task * self.args.repeat_task_traj
        task_returns_per_episode = task_returns_per_episode / exploit_episodes_count
        success_per_episode = success_per_episode / exploit_episodes_count

        return task_returns_per_episode, success_per_episode

    def _explore_task(self, task_id, explore):
        if not self.args.ablation_use_context:
            # Context doesn't exist
            return None
        if not explore:
            # Empty context
            self.transformer.dataset_storage.new_task()
            self.transformer.dataset_storage.new_meta_traj()
            return self._encode_context()

        self.envs.envs[0].unwrapped.reset_task(task_id)

        # Exploit task without any latent task information
        self.transformer.dataset_storage.new_task()
        self.transformer.dataset_storage.new_meta_traj()

        self.transformer.dataset_storage.new_task()
        self.transformer.dataset_storage.new_meta_traj()

        prev_state = utl.reset_meta_traj(self.envs)  # (num_processes, d_obs)
        if len(prev_state.shape) == 1:
            prev_state = prev_state.unsqueeze(0)
        self.transformer.dataset_storage.insert_initial_state(prev_state)

        # The latent prior
        traj_latent = self._encode_meta_traj(is_exploration_policy=True,
                                             return_prior=True, starting_state=prev_state)  # (num_processes, d_z)

        meta_traj_done_flag = False
        while not meta_traj_done_flag:
            _, action = self._select_action(self.exploration_policy,
                                            prev_state,
                                            traj_latent,
                                            np.array([int(meta_traj_done_flag)]),
                                            exploration_policy=True
                                            )  # (num_processes, 1), (num_processes, action_dim)

            next_state, (rew_raw, _), meta_episode_done, infos = utl.env_step(self.envs, action, self.args)

            if len(next_state.shape) == 1:
                next_state = next_state.unsqueeze(0)
                rew_raw = rew_raw.unsqueeze(0)

            meta_traj_done_flag = max(meta_traj_done_flag, meta_episode_done[0])

            # Add the new step to the dataset storage
            self.transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw, infos,
                                                              np.array([meta_traj_done_flag]))

            # Update the latent with the new step
            traj_latent = self._encode_meta_traj(is_exploration_policy=True)  # (num_processes, d_z)

            prev_state = next_state

            traj_done = next_state[:, -1].cpu()
            traj_done_indices = torch.argwhere(traj_done == 1).cpu()
            if traj_done > 0:
                proc = 0

                # The dataset should expect a new trajectory for this meta-trajectory and task
                self.transformer.dataset_storage.new_traj(indices=traj_done_indices)

                traj_latent = self._encode_meta_traj(is_exploration_policy=True)  # (num_processes, d_z)

                if (not meta_traj_done_flag) and traj_done:
                    prev_state[proc] = torch.from_numpy(infos[proc]['start_state']).to(global_device())
                    self.transformer.dataset_storage.insert_initial_state(prev_state[proc], proc)

        return self._encode_context()

    def _encode_context(self):
        # Get all trajectories collected by the exploration policy
        exploration_trajectories = self.transformer.dataset_storage.get_batch(
            batch_size=len(self.transformer.dataset_storage))  # (p, 1, d, m*H)

        # Task context is given by the (composed) latent of the exploration meta-trajectories
        task_context, _, _, _ = self.transformer.forward(exploration_trajectories,
                                                         compute_shared_latent=False,
                                                         use_static_shared_latent=True)  # (p, 1, H*d_z, m)

        assert task_context.shape[1] == 1, f'Shape: {task_context.shape[1]}'
        # FIXME Assumes num_processes == 1:
        assert task_context.shape[0] == 1, f'Shape: {task_context.shape[0]}'

        task_context = task_context.squeeze(1)  # (p, H*d_z, m)

        if self.args.context_type != 'full':
            task_context = task_context.reshape(
                task_context.shape[0], self.args.horizon, -1, task_context.shape[-1])  # (p, H, d_z, m)
            if self.args.context_type == 'traj_latest':
                task_context = task_context[:, -1, :, :]  # (p, d_z, m)
            elif self.args.context_type == 'latest':
                task_context = task_context[:, -1, :, -1]  # (p, d_z)
        return task_context.reshape(task_context.shape[0], -1)  # (p, H*d_z*m / d_z*m / d_z)

class MetaWorldEvaluator(MetaWorldTester):
    @property
    def task_mode(self):
        return 'train'

    def _get_tasks_for_check(self, envs):
        return envs.venv.train_tasks

    def run_eval(self):
        return self.run_test()

    def _run_test(self):
        assert not self.args.ablation_use_history

        task_returns_per_task, success_per_task = 0, 0
        for task_id in trange(len(self.envs.envs[0].unwrapped.tasks_for_mode)):
            returns, success = self._exploit_task(task_id, explore=True)
            task_returns_per_task += returns
            success_per_task += success

        task_returns = task_returns_per_task / len(self.envs.envs[0].unwrapped.tasks_for_mode)
        success = success_per_task / len(self.envs.envs[0].unwrapped.tasks_for_mode)

        return {
            'eval/success_mean': success,
            'eval/return': task_returns,
        }
