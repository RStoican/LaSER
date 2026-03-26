import copy
import os

import numpy as np
import yaml
from laser.garage.envs._utils import from_gym
from laser.garage.envs.env_utils.safe_random import SafeRandom
from laser.garage.utils import metalearner_helpers as mlutl

from laser.garage.envs._utils import gym_to_akro
from laser.garage.envs.mewa.mewa_symbolic import MEWASymbolic
from laser.garage.envs.env_tester import EnvTester

import gymnasium as gym

from laser.garage.metalearners.test_metalearner import TestMetaLearner


class MEWACurated(MEWASymbolic):
    def __init__(self,
                 task_path,
                 wide_tasks,
                 narrow_tasks,
                 complex_worker,

                 seed,
                 i,
                 split_dict=None,
                 tasks=None,
                 h=50,
                 uniform_human=False,

                 verbose=0,
                 log_path=None):
        self.curated_task_path = task_path
        self.curated_task_index = i
        self.total_tasks = None

        # FIXME This assumes a narrow distribution, i.e. all curated tasks have the same descriptor
        # Get the task descriptor path of the curated tasks
        task = self._load_task(self.curated_task_path)
        task = next(iter(task['tasks'][0]))

        super().__init__(task, None, None, complex_worker, seed, split_dict, tasks,
                         h, verbose, log_path)
        self._task_id = None
        self._max_episode_steps = h

    def set_task(self, task):
        self._print(f'set_task({task["description"]})({task["worker_personality"]})', log=True)
        super().set_task(task)

    def sample_tasks(self, task_path, wide_count, narrow_count):
        self._np_random = SafeRandom()
        return self._load_curated_tasks()

    def _load_curated_tasks(self):
        with open(self.curated_task_path, 'r') as f:
            curated_tasks = yaml.load(f, Loader=yaml.FullLoader)

        self.total_tasks = np.sum(curated_tasks['total_tasks'])

        tasks = []
        # print(f'Env index: {self.curated_task_index}')
        for curated_wide_task in curated_tasks['tasks']:
            task_path = list(curated_wide_task.keys())[0]
            task_description = self._load_task(task_path)
            for task_human_behavior in curated_wide_task[task_path]:
                # FIXME Get the sds from the task description
                #  [human_gauss[1] for human_gauss in tasks[index]['worker_personality']]
                human_sds = len(task_human_behavior) * [0]
                worker_personality = [(task_human_behavior[i], human_sds[i]) for i in range(len(human_sds))]

                tasks.append({
                    'description': task_path,
                    'worker_personality': worker_personality,
                    'task': task_description
                })
            self._update_reward_normaliser(tasks[-1])
        tasks = [tasks[self.curated_task_index]]
        # print(f'Created using index {self.curated_task_index}: '
        #       f'({tasks[0]["description"]})({tasks[0]["worker_personality"]})')
        return tasks

    def get_total_tasks(self):
        return self.total_tasks

    def compute_eval_metrics(self, data):
        return None


class MEWACuratedTester(EnvTester):
    def __init__(self, args, envs, repeat_task, repeat_task_traj, transformer, exploration_policy,
                 task_policy, shared_latent, full_output_folder):
        super().__init__(args, envs, repeat_task, repeat_task_traj, full_output_folder, transformer,
                         exploration_policy, task_policy, shared_latent)

        self.args = copy.deepcopy(self.args)
        self.args.repeat_task = repeat_task
        self.args.repeat_task_traj = repeat_task_traj

        self.args.env_name = 'MEWACurated-v0'
        self.args.task = os.path.join(os.path.dirname(self.args.task), 'curated_tasks.yaml')

        self.args.wide = None
        self.args.narrow = None

        env = gym.make(self.args.env_name, seed=0,
                       task_path=self.args.task,
                       wide_tasks=None,
                       narrow_tasks=None,
                       complex_worker=(self.args.worker == 'complex')
                       )
        self.total_tasks = env.get_total_tasks()
        self.args.action_space = None

        self.envs = mlutl.ml_make_vec_envs(args, tasks=None)

        # calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.traj_per_meta_traj

        mlutl.update_policy_input_dims(args, self.envs)
        self.envs = gym_to_akro(self.envs)

        baseline_values_path = os.path.join(os.path.dirname(self.args.task), 'curated_tasks_baselines.yaml')
        self.baselines = self._read_baselines(baseline_values_path)

        self.evaluator = TestMetaLearner(
            self.args,
            self.envs,
            self.full_output_folder,
            self.transformer,
            self.exploration_policy,
            self.task_policy,
            self.shared_latent,
        )

    def run_test(self, simple_stats=False):
        self.task_policy.actor_critic.eval()
        results = self.evaluator.eval(self.total_tasks, save_results=False, measure_mid_episodes=True)
        self.task_policy.actor_critic.train()

        # Per episode stats
        return_per_episode = results['avg_returns']
        return_per_episode = self._normalise(return_per_episode, self.baselines['random'], self.baselines['optimal'])
        last_ep_return = return_per_episode[-1]
        first_ep_return = return_per_episode[0]

        # Per task stats
        return_per_task = self._get_avg_per_task(results)
        last_avg_per_task = self._normalise(
            return_per_task[:, -1], self.baselines['random_per_task'], self.baselines['optimal_per_task'])
        first_avg_per_task = self._normalise(
            return_per_task[:, 0], self.baselines['random_per_task'], self.baselines['optimal_per_task'])

        last_std_per_task = np.std(last_avg_per_task)
        last_min_per_task = np.min(last_avg_per_task)
        last_max_per_task = np.max(last_avg_per_task)

        if simple_stats:
            return {
                'return_mean': last_ep_return,
                'last_first_diff': last_ep_return - first_ep_return,
            }

        return {
            # Per episode
            'return_mean': return_per_episode,
            'last_first_diff': last_ep_return - first_ep_return,

            # Per task
            'return_per_task': last_avg_per_task,
            'last_first_diff_per_task': last_avg_per_task - first_avg_per_task,
            'std_per_task': last_std_per_task,
            'min_per_task': last_min_per_task,
            'max_per_task': last_max_per_task,
        }

    def _normalise(self, array, min_v, max_v):
        return (array - min_v) / (max_v - min_v)

    def _get_avg_per_task(self, results):
        return np.mean(self._get_returns_per_task(results['returns']), axis=0)

    def _get_returns_per_task(self, returns):
        meta_traj_len = returns.shape[-1]
        returns_per_task = returns.reshape(-1, self.total_tasks, meta_traj_len)
        return returns_per_task

    def _read_baselines(self, baseline_values_path):
        try:
            with open(baseline_values_path, 'r') as file:
                data = yaml.safe_load(file)
            return {
                'random': data['random_policy_return'],
                'optimal': data['optimal_return'],
                'random_per_task': np.array(data['random_per_task']),
                'optimal_per_task': np.array(data['optimal_per_task']),
            }
        except FileNotFoundError:
            print(f"Error: The file '{baseline_values_path}' was not found.")
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
