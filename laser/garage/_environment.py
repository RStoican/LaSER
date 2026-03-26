"""Base Garage Environment API."""

from dataclasses import dataclass

import laser.garage.envs.akro_replace.space as akro


@dataclass(frozen=True)
class InOutSpec:
    """Describes the input and output spaces of a primitive or module."""
    input_space: akro.Space
    output_space: akro.Space


@dataclass(frozen=True, init=False)
class EnvSpec(InOutSpec):
    """Describes the observations, actions, and time horizon of an MDP.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.
        max_episode_length (int): The maximum number of steps allowed in an
            episode.

    """

    def __init__(self,
                 observation_space,
                 action_space,
                 max_episode_length=None):
        object.__setattr__(self, 'max_episode_length', max_episode_length)
        super().__init__(input_space=action_space,
                         output_space=observation_space)

    max_episode_length: int or None = None

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space of the env.

        Returns:
            akro.Space: Observation space.

        """
        return self.output_space

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._input_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._output_space = observation_space
