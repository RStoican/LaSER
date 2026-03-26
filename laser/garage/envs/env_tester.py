from abc import ABC, abstractmethod


class EnvTester(ABC):
    def __init__(
            self,
            args,
            envs,
            repeat_task,
            repeat_task_traj,
            full_output_folder,

            transformer,
            exploration_policy,
            task_policy,
            shared_latent,
    ):
        self.args = args
        self.envs = envs
        self.repeat_task = repeat_task
        self.repeat_task_traj = repeat_task_traj
        self.full_output_folder = full_output_folder
        self.transformer = transformer
        self.exploration_policy = exploration_policy
        self.task_policy = task_policy
        self.shared_latent = shared_latent

    @abstractmethod
    def run_test(self):
        pass
