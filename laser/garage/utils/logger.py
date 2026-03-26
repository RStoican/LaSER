from abc import ABC, abstractmethod
import json
import os
from datetime import datetime

from laser.garage.torch._functions import global_device


class Logger(ABC):
    def __init__(self, args, exp_label, output_folder=None, set_config_file=None):
        self.logs_dict = {}
        self.output_name = f'{exp_label}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.seed}'

        try:
            log_dir = args.results_log_dir
        except AttributeError:
            log_dir = args['results_log_dir']

        if log_dir is None:
            dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
            dir_path = os.path.join(dir_path, 'logs')
        else:
            dir_path = log_dir

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except:
                dir_path_head, dir_path_tail = os.path.split(dir_path)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.makedirs(dir_path_head)
                os.makedirs(dir_path)

        if output_folder is None:
            dir_path = args.results_log_dir if hasattr(args, 'results_log_dir') else 'logs'
            self.full_output_folder = os.path.join(os.path.join(dir_path, f'logs_{args.env_name}'), self.output_name)
        else:
            self.full_output_folder = output_folder

        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)

        args.task_log_dir = self.full_output_folder

    def save_config(self, args, filename):
        filepath = os.path.join(self.full_output_folder, filename)
        with open(filepath, 'w') as f:
            try:
                config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            except:
                config = args
            config.update(device=global_device().type)
            json.dump(config, f, indent=2)
        print(f'Config saved to {filepath}')

    @abstractmethod
    def add(self, name, value):
        pass

    @abstractmethod
    def log(self, step):
        pass

    def watch(self, models, log_freq, criterion=None, log='all', idx=None):
        pass

    def clear(self):
        pass

    def finish(self):
        pass


class WandbLogger(Logger):
    def __init__(self, args, exp_label, output_folder=None, login=True):
        super().__init__(args, exp_label, output_folder)

        import wandb
        if login:
            # Start a new wandb run to track this script
            wandb.init(project=exp_label, config=vars(args), settings=wandb.Settings(start_method="fork"))

    def add(self, name, value):
        self.logs_dict[name] = value

    def log(self, step):
        import wandb
        wandb.log(self.logs_dict, step=step)
        self.clear()

    def watch(self, models, log_freq, criterion=None, log='all', idx=None):
        import wandb
        wandb.watch(models, criterion, log_freq=log_freq, log=log, idx=idx)

    def clear(self):
        self.logs_dict = {}

    def finish(self):
        import wandb
        wandb.finish()


class DummyLogger(Logger):
    def add(self, name, value):
        pass

    def log(self, step):
        pass
