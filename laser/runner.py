import copy
import numpy as np
import torch
import pathlib

from laser.garage.metalearners.exploration_metalearner import ExplorationMetaLearner
from laser.garage.metalearners.task_metalearner import TaskMetaLearner
from laser.garage.metalearners.task_metalearner_ablations import TaskMetaLearnerAblations
from laser.garage.torch._functions import set_gpu_mode
from laser.garage.utils import helpers as utl
from laser.garage.utils.logger import DummyLogger, WandbLogger


class Runner:
    def __init__(
            self,
            args,
            exp_train=True,
            task_train=True,
    ):
        self.args = args
        self.exp_train = exp_train
        self.task_train = task_train

        set_gpu_mode(torch.cuda.is_available(), gpu_id=None)
        utl.seed(args.seed, args.deterministic_execution)

        # The meta-trajectory storage modules we use require an extra step for storing the initial and final states. So,
        # increase the horizon by 1
        args.horizon += 1

        if self.args.ablation_true_task:
            self.args.norm_state_exploration = False

        explore_or_ablation = self.exp_train or self.args.ablation_true_task
        if not explore_or_ablation:
            assert self.args.save_path is not None, \
                ('You must provide a save_path to a pre-trained encoder and exploration policy if you only run task '
                 'training (except when running ablation_true_task)')

        # FIXME Create pre_train_seed and task_train_seed from the start
        self.args.pre_train_seed = self.args.seed if self.exp_train \
            else int(pathlib.Path(self.args.save_path).name.split('_')[-1])

        output_folder = None if explore_or_ablation else self.args.save_path

        logger_class = WandbLogger if self.args.wandb else DummyLogger
        self.logger = logger_class(self.args, self.args.exp_label, output_folder=output_folder)

        if self.exp_train and self.task_train:
            self.args.save_path = self.logger.full_output_folder

        if explore_or_ablation:
            print(f'Results will be saved to {self.logger.full_output_folder}')
        else:
            print(f'Results will be loaded from and saved to {self.logger.full_output_folder}')

    def run(self):
        original_args = copy.deepcopy(self.args)
        if self.exp_train and self.args.ablation_use_context and not self.args.ablation_true_task:
            self.logger.save_config(original_args, 'config.json')
            self._exploration_train()
        if self.task_train:
            self.logger.save_config(original_args, 'config_task.json')
            self._task_train()
        print(f'Results saved in {self.logger.full_output_folder}')
        self.logger.finish()

    def _exploration_train(self):
        assert self.exp_train

        # Disable temporarily
        original_args = copy.deepcopy(self.args)
        self.args.norm_state_task = False
        self.args.norm_rew_task = False

        learner = ExplorationMetaLearner(self.args, self.logger)
        learner.train()

        self.args.norm_state_task = original_args.norm_state_task
        self.args.norm_rew_task = original_args.norm_rew_task

    def _task_train(self):
        assert self.task_train

        rng = np.random.default_rng(seed=self.args.pre_train_seed)
        new_seed = rng.choice(65536)
        self.args.seed = new_seed

        self.args.model_label = None
        if not self.args.ablation_true_task:
            learner = TaskMetaLearner(self.args, self.logger)
        else:
            learner = TaskMetaLearnerAblations(self.args, self.logger)
        learner.train()
