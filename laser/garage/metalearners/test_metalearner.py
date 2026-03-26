import copy
import os

import numpy as np
import torch
from laser.garage.envs._utils import gym_to_akro
from laser.garage.envs.parallel_envs import make_vec_envs
from laser.garage.torch._functions import set_gpu_mode, global_device
from laser.garage.utils import helpers as utl
from laser.garage.utils import metalearner_helpers as mlutl
from tqdm import trange


class TestMetaLearner:
    def __init__(
            self,
            args,
            envs,
            full_output_folder,
            transformer,
            exploration_policy,
            task_policy,
            shared_latent,
            verbose=False,
    ):
        self.args = args
        self.envs = envs
        self.full_output_folder = full_output_folder
        self.transformer = transformer
        self.exploration_policy = exploration_policy
        self.task_policy = task_policy
        self.shared_latent = shared_latent
        self.verbose = verbose

        self.iter_idx = -1
        self.measure_mid_episodes = True

        # Update args to the input/output dimensions expected by the policy (i.e. state and action dimensions)
        mlutl.update_policy_input_dims(args, self.envs)

        # Time some of the methods
        self.timer_exploration_collect, self.timer_task_collect = 0, 0

        # We use akro-type environments
        self.envs = gym_to_akro(self.envs)

        self.eval_iterations = None

        if not self.args.ablation_use_context:
            self.task_policy.actor_critic.pass_task_context_to_policy = False

    def eval(self, total_tasks, save_results=True, measure_mid_episodes=True, random_index=False):
        self.measure_mid_episodes = measure_mid_episodes
        with torch.no_grad():
            returns_per_test, exploration_returns_per_test = self._eval(total_tasks, random_index)

        self.envs.close()

        returns_per_test = returns_per_test.cpu().detach().numpy()
        if exploration_returns_per_test is not None:
            exploration_returns_per_test = exploration_returns_per_test.cpu().detach().numpy()
        results = {'returns': returns_per_test,
                   'avg_returns': np.mean(returns_per_test, axis=0),
                   'exploration_returns': exploration_returns_per_test,
                   'exploration_avg_returns': np.mean(exploration_returns_per_test, axis=0)
                   if exploration_returns_per_test is not None else None}
        self._print(results['avg_returns'])

        if save_results:
            self._save_results(results)
        return results

    def _eval(self, total_tasks, random_index):
        """ Main Meta-Training loop """

        # reset environments
        _ = utl.reset_env(self.envs, self.args)

        # log once before training
        self.log(None)

        self.eval_iterations = int(np.ceil(self.args.repeat_task / total_tasks)) * total_tasks

        returns_per_test, exploration_returns_per_test = [], []
        for self.iter_idx in trange(self.eval_iterations):
            self._print(f'{50 * "="} NEW TASK {50 * "="}')
            self._print(f'Iteration: {self.iter_idx}')
            # FIXME Make sure to collect data from all given (curated) tasks
            if not random_index and (self.args.env_name == 'MEWASymbolic-v0' or self.args.env_name == 'MEWACurated-v0'):
                task_index = self.iter_idx % total_tasks
            elif random_index or self.args.env_name == 'MEWAEval-v0':
                task_index = np.random.randint(0, 1000)
            returns_per_episode, exploration_returns_per_episode = self._evaluate(
                iter_idx=self.iter_idx,
                task_index=task_index,
            )

            returns_per_test.append(returns_per_episode)
            exploration_returns_per_test.append(exploration_returns_per_episode)
            self.transformer.dataset_storage.clear()

        returns_per_test = torch.cat(returns_per_test, dim=0)
        if self.args.ablation_true_task:
            return returns_per_test, None
        exploration_returns_per_test = torch.cat(exploration_returns_per_test, dim=0)
        return returns_per_test, exploration_returns_per_test

    def _evaluate(self,
                  iter_idx,
                  task_index,
                  ):
        """
        Collect data for a single task. This includes m exploration trajs and several task trajs

        :param iter_idx:
        :param task_index:
        :return: A list of m+1 performance scores. One for each of the m exploration trajs,
            and one for the average of all task performance trajs
        """

        env_name = self.args.env_name
        if hasattr(self.args, 'test_env_name'):
            env_name = self.args.test_env_name

        # Create environments that have a single task (same task in all envs)
        exploration_envs = self._make_vec_eval_envs(env_name, iter_idx, task_index)
        task_envs = self._make_vec_eval_envs(env_name, iter_idx + int(self.eval_iterations), task_index)

        # FIXME Make sure all tasks are being used
        task_returns_per_episode, exploration_returns_per_episode = self._explore(exploration_envs, task_envs)

        exploration_envs.close()
        task_envs.close()

        if self.args.ablation_true_task:
            assert exploration_returns_per_episode is None
        else:
            assert exploration_returns_per_episode.shape[-1] == (task_returns_per_episode.shape[-1] - 1)
        return task_returns_per_episode, exploration_returns_per_episode

    def _explore(self, exploration_envs, task_envs):
        if self.args.ablation_true_task:
            task_returns_per_episode = torch.zeros((self.args.num_processes, 2)).to(global_device())  # (num_proc, 2)

            task_returns_per_episode[range(self.args.num_processes), 0] \
                = self._exploit(task_envs,
                                None,
                                None,
                                true_context=True,
                                do_update_context=False)

            task_returns_per_episode[range(self.args.num_processes), -1] \
                = self._exploit(task_envs,
                                None,
                                None,
                                true_context=True)
            return task_returns_per_episode, None

        assert self.transformer.dataset_storage.is_empty()

        # for each process, we log the returns during the first, second, ... episode (such that we have a minimum of
        #   [num_episodes]);
        # the first traj_per_meta_traj are for exploration trajs
        # the second to last column is for the task solving traj
        # the last column is for any overflow and will be discarded at the end, because we need to wait
        #   until all processes have at least [num_episodes] many episodes
        exploration_returns_per_episode = torch.zeros((self.args.num_processes, self.args.traj_per_meta_traj + 1)) \
            .to(global_device())  # (num_proc, m+1)
        task_returns_per_episode = torch.zeros((self.args.num_processes, self.args.traj_per_meta_traj + 1)) \
            .to(global_device())  # (num_proc, m+1)
        meta_traj_index = torch.zeros(self.args.num_processes, dtype=torch.long).to(global_device())  # (num_proc)

        # Exploit task without any latent task information
        self.transformer.dataset_storage.new_task()
        self.transformer.dataset_storage.new_meta_traj()
        task_returns_per_episode[range(self.args.num_processes), 0] = self._exploit(task_envs, None, None)

        self.transformer.dataset_storage.new_task()
        self.transformer.dataset_storage.new_meta_traj()

        # Reset the environment to new tasks (i.e. the only task currently available in the env)
        utl.reset_env(exploration_envs, self.args)
        exploration_tasks, exploration_task_ids = None, None

        # Start collecting data
        # FIXME When not using the context, it will return 0 for all episodes besides first and last.
        #  Fix this (0 can be seen as a very high return in MEWA)
        if self.args.ablation_use_context:
            prev_state = utl.reset_meta_traj(exploration_envs)  # (num_processes, d_obs)
            if len(prev_state.shape) == 1:
                prev_state = prev_state.unsqueeze(0)
            self.transformer.dataset_storage.insert_initial_state(prev_state)

            # The latent prior
            traj_latent = self._encode_meta_traj(is_exploration_policy=True,
                                                 return_prior=True, starting_state=prev_state)  # (num_processes, d_z)

            meta_traj_done_flag = np.zeros(self.args.num_processes, dtype=int)

            self._print(f'{50 * "="} NEW EXPLORATION TRAJECTORY {50 * "="}')
            while meta_traj_done_flag.sum() != self.args.num_processes:
                # sample actions from the exploration policy (using the latent traj)
                _, action = self._select_action(self.exploration_policy,
                                                prev_state,
                                                traj_latent,
                                                meta_traj_done_flag,
                                                exploration_policy=True
                                                )  # (num_processes, 1), (num_processes, action_dim)

                # Take step in the environment
                #   meta_episode_done is true only when the last traj of the meta-traj ends (default 10 trajs)
                #   next_state will contain the episode_done flag appended at the end of the next_state features
                # If the current trajectory is finished, env_step will automatically reset the environment
                # (but NOT the task) and return:
                #   the last state of the previous traj in next_state
                #   the first state of the new traj in info['start_state']
                next_state, (rew_raw, rew_normalised), meta_episode_done, infos \
                    = utl.env_step(exploration_envs, action, self.args)

                if len(next_state.shape) == 1:
                    next_state = next_state.unsqueeze(0)
                    rew_raw = rew_raw.unsqueeze(0)

                # add rewards
                for proc_idx, info in enumerate(infos):
                    exploration_returns_per_episode[proc_idx, meta_traj_index[proc_idx]] += infos[proc_idx]['reward_raw']

                if exploration_task_ids is None:
                    exploration_task_ids = [info['task_id'] for info in infos]
                    exploration_tasks = [info['task'] for info in infos]
                assert [info['task_id'] for info in infos] == exploration_task_ids \
                       and [info['task'] for info in infos] == exploration_tasks

                meta_traj_done_flag = np.array(
                    [max(meta_traj_done_flag[i], meta_episode_done[i]) for i in range(self.args.num_processes)], dtype=int)

                # Add the new step to the dataset storage
                self.transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw, infos,
                                                                  meta_traj_done_flag)

                # Update the latent with the new step
                traj_latent = self._encode_meta_traj(is_exploration_policy=True)  # (num_processes, d_z)

                traj_done = next_state[:, -1].cpu()
                traj_done_indices = torch.argwhere(traj_done == 1).cpu()
                done_idxs = traj_done_indices.squeeze(-1)

                prev_state = next_state

                if len(traj_done_indices) > 0:
                    # The dataset should expect a new trajectory for this meta-trajectory and task
                    self.transformer.dataset_storage.new_traj(indices=traj_done_indices)
                    meta_traj_index[traj_done_indices.squeeze(-1)] += 1

                    if self.measure_mid_episodes:
                        exploration_state = self.transformer.dataset_storage.state.clone()
                        exploration_meta_traj_index = copy.deepcopy(self.transformer.dataset_storage.meta_traj_index)

                        if self.args.ablation_use_history:
                            saved_storage = self.transformer.dataset_storage.save()

                        # When a trajectory is done, evaluate the task policy
                        # (using current exploration policies as context)
                        exploit_results = self._exploit(task_envs,
                                                        exploration_task_ids,
                                                        exploration_tasks)
                        task_returns_per_episode[done_idxs, meta_traj_index[done_idxs]] = exploit_results[done_idxs]

                        if self.args.ablation_use_history:
                            self.transformer.dataset_storage.load(saved_storage)

                        assert torch.all(torch.eq(exploration_state, self.transformer.dataset_storage.state)) \
                               and exploration_meta_traj_index == self.transformer.dataset_storage.meta_traj_index, \
                            'The dataset storage no longer contains exploration data'

                    traj_latent = self._encode_meta_traj(is_exploration_policy=True)  # (num_processes, d_z)

                    for proc in range(self.args.num_processes):
                        if meta_traj_done_flag[proc] == 0 and traj_done[proc]:
                            prev_state[proc] = torch.from_numpy(infos[proc]['start_state']).to(global_device())
                            self.transformer.dataset_storage.insert_initial_state(prev_state[proc], proc)

        task_returns_per_episode[range(self.args.num_processes), -1] \
            = self._exploit(task_envs,
                            exploration_task_ids,
                            exploration_tasks)

        exploration_returns_per_episode = exploration_returns_per_episode[:, :-1]
        return task_returns_per_episode, exploration_returns_per_episode

    def _exploit(self, task_envs, exploration_task_ids, exploration_tasks,
                 true_context=False, do_update_context=True):
        task_returns_per_episode = torch.zeros(self.args.num_processes).to(global_device())  # (num_proc)

        if self.args.ablation_use_context and not true_context:
            # Get all trajectories collected by the exploration policy
            exploration_trajectories = self.transformer.dataset_storage.get_batch(
                batch_size=len(self.transformer.dataset_storage))  # (p, 1, d, m*H)

            # Task context is given by the (composed) latent of the exploration meta-trajectories
            task_context, _, _, _ = self.transformer.forward(exploration_trajectories,
                                                             compute_shared_latent=False,
                                                             use_static_shared_latent=True)  # (p, 1, H*d_z, m)
            assert task_context.shape[1] == 1, f'Shape: {task_context.shape[1]}'
            assert task_context.shape[0] == self.args.num_processes, \
                f'Shape: {task_context.shape[0]} != {self.args.num_processes}'
            task_context = task_context.squeeze(1)  # (p, H*d_z, m)

            if self.args.context_type != 'full':
                task_context = task_context.reshape(
                    task_context.shape[0], self.args.horizon, -1, task_context.shape[-1])  # (p, H, d_z, m)
                if self.args.context_type == 'traj_latest':
                    task_context = task_context[:, -1, :, :]  # (p, d_z, m)
                elif self.args.context_type == 'latest':
                    task_context = task_context[:, -1, :, -1]  # (p, d_z)
            task_context = task_context.reshape(task_context.shape[0], -1)  # (p, H*d_z*m / d_z*m / d_z)
        else:
            task_context = None

        self._print(f'{50 * "="} EXPLOITING {50 * "="}')
        for _ in range(self.args.repeat_task_traj):
            if self.args.ablation_use_history:
                self.transformer.dataset_storage.clear()
                self.transformer.dataset_storage.new_task()
                self.transformer.dataset_storage.new_meta_traj()

            # Reset the environment to the same (single) task
            utl.reset_env(task_envs, self.args)

            if not true_context:
                context = task_context if task_context is not None else None  # (p, H*d_z*m)
                if (context is not None) and (len(context.shape) == 1):
                    context = context.unsqueeze(0)
            else:
                context = torch.zeros((self.args.num_processes, int(np.prod(task_envs.action_space.shape))),
                                      device=global_device())

            # Reset the whole BAMDP environment, but to the same task
            prev_state = utl.reset_meta_traj(task_envs)  # (num_processes, d_obs)
            if len(prev_state.shape) == 1:
                prev_state = prev_state.unsqueeze(0)
            if self.args.ablation_use_history:
                self.transformer.dataset_storage.insert_initial_state(prev_state)

            # The latent prior
            traj_latent = self._encode_meta_traj(is_exploration_policy=False,
                                                 return_prior=True, starting_state=prev_state)  # (num_processes, d_z)

            # Run for a single trajectory per task
            traj_done_flag = np.zeros(self.args.num_processes, dtype=int)
            updated_context = False
            while traj_done_flag.sum() != self.args.num_processes:
                # sample actions from the task policy (using the latent traj and task context)
                _, action = self._select_action(self.task_policy,
                                                prev_state,
                                                traj_latent,
                                                traj_done_flag,
                                                task_context=context,
                                                )  # (num_processes, 1), (num_processes, action_dim)

                # Take step in the environment
                #   meta_episode_done is true only when the last traj of the meta-traj ends (default 10 trajs)
                #   next_state will contain the episode_done flag appended at the end of the next_state features
                # If the current trajectory is finished, env_step will automatically reset the environment
                # (but NOT the task) and return:
                #   the last state of the previous traj in next_state
                #   the first state of the new traj in info['start_state']
                next_state, (rew_raw, rew_normalised), _, infos = utl.env_step(task_envs, action, self.args)

                if not true_context and not updated_context and do_update_context and self.args.ablation_use_history:
                    context = []
                    for proc in range(self.args.num_processes):
                        task = infos[proc]['task']
                        # FIXME Replace with a for loop
                        context.append(torch.tensor([task[0][0], task[1][0], task[2][0], task[3][0]]).unsqueeze(0))
                    context = torch.cat(context, dim=0)
                    context = context.to(global_device())
                    updated_context = True

                if len(next_state.shape) == 1:
                    next_state = next_state.unsqueeze(0)
                    rew_raw = rew_raw.unsqueeze(0)
                    rew_normalised = rew_normalised.unsqueeze(0)

                # add rewards
                task_returns_per_episode[range(self.args.num_processes)] += infos[0]['reward_raw']

                # Make sure we are using the same tasks as during exploration
                if exploration_task_ids is not None and exploration_tasks is not None:
                    assert [info['task_id'] for info in infos] == exploration_task_ids \
                           and [info['task'] for info in infos] == exploration_tasks

                # A flag for whether the current traj (not meta-traj) is done
                traj_done = next_state[:, -1].cpu()
                traj_done_flag = np.array(
                    [max(traj_done_flag[i], traj_done[i]) for i in range(self.args.num_processes)], dtype=int)

                # Add the new step to the dataset storage
                if self.args.ablation_use_history:
                    self.transformer.dataset_storage.insert_meta_step(next_state, action, next_state, rew_raw,
                                                                      infos, traj_done_flag)

                # Update the latent with the new step
                traj_latent = self._encode_meta_traj(is_exploration_policy=False)  # (num_processes, d_z)

                prev_state = next_state

        if self.args.ablation_use_history:
            self.transformer.dataset_storage.clear()
        task_returns_per_episode = task_returns_per_episode / self.args.repeat_task_traj
        return task_returns_per_episode

    def _encode_meta_traj(self, is_exploration_policy,
                          return_latest_latent=True, return_prior=False, starting_state=None):
        if not is_exploration_policy and not self.args.ablation_use_history:
            return None
        # FIXME This won't work if the exploration policy was trained without the latent history

        if return_prior:
            return self.transformer.forward_exploration(
                prior=True,
                num_tasks=self.args.num_processes,
                return_latest_latent=True,
                starting_state=starting_state).squeeze(1)  # (num_processes, d_z)
        assert starting_state is None

        # Get a dataset for the current tasks for which we collect data
        # This dataset only includes the current meta-trajectory. This is similar to the meta-testing case, where the
        # policy will not have additional meta-trajectories (i.e. one meta-trajectory per task)
        current_batch = self.transformer.dataset_storage.get_working_meta_traj()  # (num_processes, d, m*H)
        current_batch = current_batch.unsqueeze(1)  # (num_processes, 1, d, m*H)
        return self.transformer.forward_exploration(
            trajectories=current_batch,
            return_latest_latent=return_latest_latent).squeeze(1)  # (num_processes, d_z)

    def _select_action(self, policy, prev_state, latent, meta_traj_done_flag, task_context=None, exploration_policy=False):
        if latent is not None:
            assert prev_state.shape[0] == latent.shape[0], \
                f'Got shapes: {prev_state.shape}; {latent.shape}'
        if task_context is not None:
            assert prev_state.shape[0] == task_context.shape[0], \
                f'Got shapes: {prev_state.shape}; {task_context.shape}'

        value, action = utl.select_action(
            args=self.args,
            policy=policy,
            state=prev_state,
            # FIXME Maybe use args to set this
            deterministic=not exploration_policy,
            latent=latent,
            task_context=task_context,
        )  # (num_processes, 1), (num_processes, action_dim)

        # If any of the meta-trajectories already have m episodes, then do not return an action for that
        if meta_traj_done_flag.sum() > 0:
            for proc in range(self.args.num_processes):
                if meta_traj_done_flag[proc] == 1:
                    action_padding = torch.from_numpy(utl.padding_action(self.envs.action_space)) \
                        .float().to(global_device())
                    action[proc, :] = 0 * action[proc, :] + action_padding
                    # action[proc, :] *= 0
        return value, action

    def _make_vec_eval_envs(self, env_name, iter_idx, task_index):
        if (self.args.env_name == 'MEWASymbolic-v0'
                or self.args.env_name == 'MEWACurated-v0'
                or self.args.env_name == 'MEWAEval-v0'):
            return make_vec_envs(env_name,
                                 seed=self.args.seed * 42 + iter_idx,
                                 num_processes=self.args.num_processes,
                                 gamma=self.args.task_policy_gamma,
                                 device=global_device(),
                                 rank_offset=self.args.num_processes + 1,
                                 # to use diff tmp folders than main processes
                                 h=self.args.horizon - 1,
                                 episodes_per_task=self.args.traj_per_meta_traj,
                                 normalise_rew=self.args.norm_rew_task,
                                 tasks=None,
                                 task_path=self.args.task,
                                 wide_tasks=self.args.wide if 'wide' in self.args else 1,
                                 narrow_tasks=self.args.narrow,
                                 add_done_info=True,
                                 complex_worker=(self.args.worker == 'complex'),
                                 split_dict=utl.create_split_dict(self.args,
                                                                  train_task_count=0,
                                                                  test_task_count=self.args.narrow),
                                 # this gives the index of the current task being used
                                 env_index=task_index,
                                 )
        if self.args.env_name == 'MetaWorldML1-v0' or self.args.env_name == 'MetaWorldML10-v0':
            raise NotImplementedError
        raise ValueError(f'Unknown environment {self.args.env_name}')

    def _save_results(self, results):
        if 'results_name' in vars(self.args) and self.args.results_name is not None:
            results_name = self.args.results_name
        else:
            results_name = 'results'
            if self.ablations_run:
                results_name = f'ablations_{results_name}'
            if not self.args.ablation_use_context:
                results_name = f'{results_name}_nocontext'
        if os.path.splitext(results_name)[1] != '.npz':
            results_name += '.npz'
        with open(os.path.join(self.full_output_folder, results_name), 'wb') as f:
            np.savez(f, **results)

    def log(self, policy_train_stats):
        with torch.no_grad():
            # TODO
            pass

    def _print(self, msg):
        if self.verbose:
            print(msg)

